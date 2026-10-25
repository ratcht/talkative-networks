"""
Invariants the testbed's contracts promise. Synthetic data on the CPU, so these
run in seconds with no downloads, GPU or MLflow.

  uv run pytest tests
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest
import torch
from fakes import fake_data, random_experts

from testbed.communication import ChannelConfig, Exchange, Message, build_ensemble
from testbed.communication.channel import PooledEncoder, build_channel
from testbed.data import Weighted
from testbed.experts import (
  Budget,
  ClassSkew,
  NegativeCorrelation,
  Redundant,
  SeedOnly,
  get_pool,
)
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


@pytest.mark.parametrize(
  "exchanges, rounds, encoder, decoder",
  [
    ([Exchange("stage1", "stage2")], 1, "pooled", "broadcast"),
    ([Exchange("stage1", "stage3")], 1, "spatial", "spatial"),
    ([Exchange("stage2", "stage1")], 1, "pooled", "broadcast"),
    ([Exchange("stage3", "stage1")], 3, "spatial", "spatial"),
    ([], 1, "pooled", "broadcast"),
  ],
)
def test_untrained_channel_is_bagging(exchanges, rounds, encoder, decoder):
  config = ChannelConfig(encoder=encoder, decoder=decoder)
  model = build_ensemble(
    random_experts(), exchanges, config, (1, 28, 28), rounds, seed=0
  )
  x = fake_data().joint_val.x
  with torch.no_grad():
    assert torch.allclose(model(x).log_probs, model.bagging(x).log_probs, atol=1e-6)


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
