"""
Invariants the testbed's contracts promise. Synthetic data on the CPU, so these
run in seconds with no downloads, GPU or MLflow.

  uv run pytest tests
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest
import torch
from fakes import fake_data, random_experts

from testbed.communication import ChannelConfig, Exchange, Message, build_ensemble
from testbed.communication.channel import (
  LinearTaperDecoder,
  LinearTaperEncoder,
  LinearTaperGELUDecoder,
  LinearTaperGELUEncoder,
  PooledEncoder,
  build_channel,
)
from testbed.data import Weighted
from testbed.experts import (
  Budget,
  ClassSkew,
  NegativeCorrelation,
  Redundant,
  SeedOnly,
  get_pool,
)
from testbed.run import extra_tags, parse_args
from testbed.seeds import derive, generator
from testbed.train import loss_fn


def test_seed_streams_are_named_and_stable():
  assert derive(0, "pool") == derive(0, "pool")
  assert derive(0, "pool") != derive(0, "group")
  assert derive(0, "a", "b") != derive(0, "b", "a")
  assert derive(1, "expert", 0) != derive(0, "expert", 1)  # no `seed + i` collisions


def test_class_skew_draws_follow_each_experts_mix():
  data = fake_data(n=2000)
  (recipe,) = ClassSkew(alpha=1.0).recipes(data, 1, seed=3)
  draws = Weighted(recipe.sampler.weights, 200_000).epoch(generator(0))
  freq = torch.bincount(data.expert.y[draws], minlength=data.num_classes).float() / len(
    draws
  )
  assert torch.allclose(freq, torch.tensor(recipe.profile["class_mix"]), atol=0.01)


def test_redundant_copies_share_a_mix_but_not_a_seed():
  recipes = Redundant(ClassSkew(1.0), copies=2).recipes(fake_data(), 4, seed=0)
  assert recipes[0].profile["class_mix"] == recipes[1].profile["class_mix"]
  assert recipes[0].profile["class_mix"] != recipes[2].profile["class_mix"]
  assert recipes[0].seed != recipes[1].seed


GELU = ChannelConfig("linear_taper_gelu", decoder="linear_taper_gelu")


@pytest.mark.parametrize(
  "exchanges, rounds, config",
  [
    ([Exchange("stage1", "stage2")], 1, ChannelConfig()),
    ([Exchange("stage2", "stage1")], 2, ChannelConfig()),
    ([Exchange("stage1", "stage3")], 1, ChannelConfig("linear_taper_gelu")),
    ([Exchange("stage3", "stage1")], 3, ChannelConfig(decoder="linear_taper_gelu")),
    ([Exchange("stage2", "stage3")], 1, GELU),
    ([Exchange("stage3", "stage2")], 2, GELU),
    ([Exchange("stage2", "stage3")], 1, ChannelConfig(mixer="self")),
    ([Exchange("stage3", "stage2")], 2, ChannelConfig(mixer="mean")),
    ([Exchange("stage1", "stage2")], 1, ChannelConfig("pooled", decoder="broadcast")),
    ([Exchange("stage1", "stage3")], 1, ChannelConfig("spatial", decoder="spatial")),
    ([Exchange("stage2", "stage1")], 1, ChannelConfig("pooled", decoder="broadcast")),
    ([Exchange("stage3", "stage1")], 3, ChannelConfig("spatial", decoder="spatial")),
    ([], 1, ChannelConfig()),
  ],
)
def test_untrained_channel_is_bagging(exchanges, rounds, config):
  model = build_ensemble(
    random_experts(), exchanges, config, (1, 28, 28), rounds, seed=0
  )
  x = fake_data().joint_val.x
  with torch.no_grad():
    assert torch.allclose(model(x).log_probs, model.bagging(x).log_probs, atol=1e-6)


def count(module: torch.nn.Module) -> int:
  return sum(p.numel() for p in module.parameters())


def additive(f, a: torch.Tensor, b: torch.Tensor) -> bool:
  """f(a + b) - f(0) == (f(a) - f(0)) + (f(b) - f(0)): f is linear up to a constant."""
  zero = f(torch.zeros_like(a))
  return torch.allclose(f(a + b) - zero, f(a) + f(b) - 2 * zero, atol=1e-4)


def test_each_level_only_adds_a_gelu():
  """The linear_taper parts and their linear_taper_gelu versions have the same
  weights. A linear_taper part is linear, up to its biases, and a
  linear_taper_gelu part is not."""
  config = ChannelConfig()
  read, write = (32, 7, 7), (16, 14, 14)
  encoders = [LinearTaperEncoder(config, read), LinearTaperGELUEncoder(config, read)]
  decoders = [LinearTaperDecoder(config, write), LinearTaperGELUDecoder(config, write)]
  # zero patterns would make any decoder linear
  torch.nn.init.normal_(decoders[0].write.weight)
  for linear, gelu in (encoders, decoders):
    gelu.load_state_dict(linear.state_dict())  # strict: the same weights, name for name

  x, y = torch.randn(2, 4, *read)
  m, n = torch.randn(2, 4, 1, config.value_dim)
  features = torch.zeros(4, *write)
  with torch.no_grad():
    assert additive(lambda x: encoders[0](x).value, x, y)
    assert not additive(lambda x: encoders[1](x).value, x, y)
    assert additive(lambda m: decoders[0](m, features), m, n)
    assert not additive(lambda m: decoders[1](m, features), m, n)


def test_channel_size_matches_the_docs():
  """docs/testbed/channel.md: per expert, r (read + write + 2 key + 2 value + 2)
  + 2 key + value weights, so a forward exchange and its backward mirror match."""
  config = ChannelConfig()
  r, k, v = config.width, config.key_dim, config.value_dim
  shapes = {"stage1": (16, 14, 14), "stage2": (32, 7, 7), "stage3": (64, 3, 3)}
  for a, b in (("stage1", "stage2"), ("stage2", "stage3"), ("stage1", "stage3")):
    there = count(build_channel(config, shapes[a], shapes[b], num_experts=1))
    back = count(build_channel(config, shapes[b], shapes[a], num_experts=1))
    size = math.prod(shapes[a]) + math.prod(shapes[b])
    assert there == back == r * (size + 2 * k + 2 * v + 2) + 2 * k + v


ARGS = ["--arm", "a", "--trial-seeds", "0"]


def test_the_command_line_takes_tags_and_full_flag_names_only():
  args = parse_args([*ARGS, "--tag", "study=s", "--tag", "level=3", "--tag", "x=a=b"])
  assert extra_tags(args) == {"study": "s", "level": "3", "x": "a=b"}
  with pytest.raises(SystemExit):
    extra_tags(parse_args([*ARGS, "--tag", "no-value"]))
  with pytest.raises(SystemExit):  # so no abbreviation stands in for --encoder
    parse_args([*ARGS, "--enc", "pooled"])


@pytest.mark.parametrize("mixer", ["self", "mean"])
def test_the_controls_hear_what_they_promise(mixer):
  """self: each expert receives its own message. mean: every expert receives
  the plain average of all of them. Neither has weights to route by, and
  neither trains the query or key."""
  channel = build_channel(
    ChannelConfig(mixer=mixer), (32, 7, 7), (16, 14, 14), num_experts=3
  )
  features = [torch.randn(4, 32, 7, 7) for _ in range(3)]
  reading = channel.read(features)
  sent = reading.sent.value  # (B, M, T, value_dim)
  heard = sent if mixer == "self" else sent.mean(1, keepdim=True).expand_as(sent)
  assert reading.weights is None
  assert torch.allclose(reading.received, heard)
  for decoder in channel.decoders:  # so that the loss reaches the encoders
    torch.nn.init.normal_(decoder.write.weight)
  injection = channel.write(reading.received, [torch.zeros(4, 16, 14, 14)] * 3)
  sum(i.pow(2).sum() for i in injection).backward()
  for encoder in channel.encoders:
    assert encoder.query.weight.grad is None and encoder.key.weight.grad is None
    assert encoder.value.weight.grad.abs().sum() > 0


def test_an_encoder_sends_one_message_per_expert():
  config = ChannelConfig()
  channel = build_channel(config, (16, 14, 14), (32, 7, 7), num_experts=2)

  class TwoMessages(PooledEncoder):
    def forward(self, features):
      m = super().forward(features)
      return Message(*(x.repeat(1, 2, 1) for x in (m.query, m.key, m.value)))

  features = [torch.randn(4, 16, 14, 14)] * 2
  assert channel.read(features).received.shape == (4, 2, 1, config.value_dim)
  channel.encoders[0] = TwoMessages(config, (16, 14, 14))
  with pytest.raises(ValueError):
    channel.read(features)


@pytest.mark.parametrize(
  "exchanges, rounds",
  [([Exchange("stage1", "stage2")], 1), ([Exchange("stage2", "stage1")], 2)],
)
def test_only_the_channel_trains(exchanges, rounds):
  model = build_ensemble(
    random_experts(), exchanges, ChannelConfig(), (1, 28, 28), rounds, seed=0
  )
  experts_before = [p.clone() for p in model.experts.parameters()]
  channel_before = [p.clone() for p in model.channels.parameters()]
  data = fake_data()
  optimizer = torch.optim.Adam(model.channels.parameters(), lr=1e-2)
  model.train()
  for _ in range(
    2
  ):  # the zero-initialised last layer moves first, the layers behind it next
    loss = loss_fn(model(data.joint.x[:64]).logits, data.joint.y[:64], "ensemble")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  assert not model.experts.training
  assert all(p.grad is None for p in model.experts.parameters())
  assert all(
    torch.equal(a, b) for a, b in zip(experts_before, model.experts.parameters())
  )
  assert any(
    not torch.equal(a, b) for a, b in zip(channel_before, model.channels.parameters())
  )


@pytest.mark.parametrize(
  "exchanges, rounds",
  [
    ([Exchange("stage2", "stage2")], 1),  # reads and writes at one point
    ([Exchange("stage1", "stage2"), Exchange("stage2", "stage3")], 1),  # shares a point
    ([Exchange("stage1", "stage3"), Exchange("stage2", "stage3")], 1),  # overlaps
    ([Exchange("stage3", "stage1"), Exchange("stage2", "stage1")], 1),  # two backward
    ([Exchange("stage1", "stage2")], 2),  # rounds on a forward exchange
    ([Exchange("stage1", "head")], 1),  # the head's output isn't a point
  ],
)
def test_invalid_exchanges_are_rejected(exchanges, rounds):
  with pytest.raises(ValueError):
    build_ensemble(
      random_experts(), exchanges, ChannelConfig(), (1, 28, 28), rounds, seed=0
    )


def test_negative_correlation_trains_a_pool_jointly():
  pool = NegativeCorrelation(lam=0.5).train_pool(
    "lenet5", fake_data(), 3, seed=0, budget=Budget(1, 50)
  )
  assert len(pool) == 3
  assert not any(e.model.training for e in pool)
  assert not torch.equal(
    pool[0].model.stages["head"][-1].weight, pool[1].model.stages["head"][-1].weight
  )


def test_pool_cache_returns_the_same_experts(tmp_path):
  data, budget = fake_data(), Budget(1, 50)
  first = get_pool(SeedOnly(), "cnn3", data, 2, seed=0, budget=budget, root=tmp_path)
  again = get_pool(SeedOnly(), "cnn3", data, 2, seed=0, budget=budget, root=tmp_path)
  other = get_pool(
    ClassSkew(0.5), "cnn3", data, 2, seed=0, budget=budget, root=tmp_path
  )
  assert first.key == again.key != other.key
  assert first.digest == again.digest != other.digest
  for a, b in zip(first.experts, again.experts):
    assert all(
      torch.equal(x, y)
      for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values())
    )


@dataclass(frozen=True)
class SlowSeedOnly(SeedOnly):
  """SeedOnly that records every time it trains, and takes long enough that
  two runs asking for the same pool at once overlap."""

  calls: list = field(default_factory=list)

  def train_pool(self, *args):
    self.calls.append(None)
    time.sleep(0.5)
    return super().train_pool(*args)


def test_runs_asking_at_once_share_one_pool(tmp_path):
  scheme, data, budget = SlowSeedOnly(), fake_data(), Budget(1, 50)

  def ask(_):
    return get_pool(scheme, "cnn3", data, 2, seed=0, budget=budget, root=tmp_path)

  with ThreadPoolExecutor(2) as executor:
    first, second = executor.map(ask, range(2))
  assert len(scheme.calls) == 1  # the second waited for the first, then loaded
  assert first.digest == second.digest
  assert [p.name for p in tmp_path.glob("*.pt")] == [f"{first.key}.pt"]
  assert not list(tmp_path.glob("*.tmp"))
