"""
Invariants the testbed's contracts promise. Synthetic data on the CPU, so these
run in seconds with no downloads, GPU or MLflow.

  uv run pytest tests
"""

import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest
import torch
from fakes import fake_data, random_experts

from testbed.communication import ChannelConfig, Exchange, Message, build_ensemble
from testbed.communication.channel import (
  MIXERS,
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
  DataShard,
  DisjointSubset,
  NegativeCorrelation,
  Redundant,
  SeedOnly,
  Untrained,
  get_pool,
)
from testbed.experts.pool import weights_digest
from testbed.run import (
  check_flags,
  code_version,
  extra_tags,
  group_experts,
  make_scheme,
  parse_args,
)
from testbed.seeds import derive, generator
from testbed.selection import ClonesOfFirst
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


def owner(scheme, data, size=3, seed=0) -> torch.Tensor:
  """Which expert's part each of the expert split's examples is in."""
  labels = data.expert.y.numpy()
  return torch.as_tensor(scheme.parts(labels, data.num_classes, size, seed)[0])


def test_disjoint_subsets_split_the_classes_once_per_pool():
  data = fake_data()
  split = [r.profile["classes"] for r in DisjointSubset().recipes(data, 3, seed=0)]
  assert [len(s) for s in split] == [4, 3, 3]
  assert sorted(c for s in split for c in s) == list(range(data.num_classes))
  again = [r.profile["classes"] for r in DisjointSubset().recipes(data, 3, seed=0)]
  assert again == split
  others = [
    [r.profile["classes"] for r in DisjointSubset().recipes(data, 3, seed=s)]
    for s in range(1, 4)
  ]
  assert any(other != split for other in others)
  parts = owner(DisjointSubset(), data)
  for i, classes in enumerate(split):
    assert torch.equal(parts == i, torch.isin(data.expert.y, torch.tensor(classes)))


def test_data_shards_cut_every_class_in_the_subsets_shares():
  data = fake_data(n=2000)
  y, parts = data.expert.y, owner(DataShard(), data)
  for c in range(data.num_classes):
    in_class = y == c
    for i, share in enumerate((0.4, 0.3, 0.3)):
      count = ((parts == i) & in_class).sum().item()
      assert abs(count - share * in_class.sum().item()) <= 1
  assert not torch.equal(parts, owner(DataShard(), data, seed=1))


@pytest.mark.parametrize(
  "scheme, epsilon",
  [(DisjointSubset(0.0), 0.0), (DisjointSubset(0.25), 0.25), (DataShard(), 0.0)],
)
def test_epsilon_is_the_share_of_draws_outside_an_experts_part(scheme, epsilon):
  data = fake_data()
  parts = owner(scheme, data)
  for i, recipe in enumerate(scheme.recipes(data, 3, seed=0)):
    weights, own = recipe.sampler.weights, parts == i
    assert (weights[~own].sum() / weights.sum()).item() == pytest.approx(epsilon)
    assert torch.unique(weights[own]).numel() == 1  # uniform inside the part
    assert torch.unique(weights[~own]).numel() == 1  # and outside it
    assert recipe.sampler.draws == len(data.expert)


def test_disjoint_parts_start_from_seed_only_weights_and_check_their_settings():
  data = fake_data()
  seeds = [r.seed for r in SeedOnly().recipes(data, 3, seed=5)]
  for scheme in (DisjointSubset(0.1), DataShard()):
    assert [r.seed for r in scheme.recipes(data, 3, seed=5)] == seeds
    with pytest.raises(ValueError):
      scheme.recipes(data, data.num_classes + 1, seed=5)
  for epsilon in (-0.1, 1.0):
    with pytest.raises(ValueError):
      DisjointSubset(epsilon)


def test_disjoint_subsets_from_the_command_line():
  """A pool's cache key hashes the config as JSON, where 0 and 0.0 differ."""
  run = ["--arm", "a", "--trial-seeds", "0"]
  args = parse_args([*run, "--scheme", "disjoint_subset", "--epsilon", "0"])
  assert make_scheme(args).config() == {"name": "disjoint_subset", "epsilon": 0.0}
  assert isinstance(make_scheme(args).config()["epsilon"], float)
  default = parse_args([*run, "--scheme", "disjoint_subset"])
  assert make_scheme(default).config() == DisjointSubset(0).config()
  shard = parse_args([*run, "--scheme", "data_shard"])
  assert make_scheme(shard).config() == {"name": "data_shard"}
  check_flags(shard)
  with pytest.raises(SystemExit):  # the shard control has no epsilon
    check_flags(parse_args([*run, "--scheme", "data_shard", "--epsilon", "0.1"]))
  copies = parse_args([*run, "--scheme", "disjoint_subset", "--copies", "2"])
  recipes = make_scheme(copies).recipes(fake_data(), 4, seed=0)
  assert recipes[0].profile["classes"] == recipes[1].profile["classes"]
  assert recipes[0].profile["classes"] != recipes[2].profile["classes"]


@pytest.mark.parametrize("scheme", [DisjointSubset(0.1), DataShard()])
def test_disjoint_parts_train_and_reload_from_the_cache(scheme, tmp_path):
  data, budget = fake_data(), Budget(1, 50)
  first = get_pool(scheme, "tiny_resnet", data, 3, seed=0, budget=budget, root=tmp_path)
  again = get_pool(scheme, "tiny_resnet", data, 3, seed=0, budget=budget, root=tmp_path)
  assert first.digest == again.digest
  assert [e.profile for e in again.experts] == [e.profile for e in first.experts]


GELU = ChannelConfig("linear_taper_gelu", decoder="linear_taper_gelu")


@pytest.mark.parametrize(
  "exchanges, rounds, config",
  [
    ([Exchange("block1", "block2")], 1, ChannelConfig()),
    ([Exchange("block2", "block1")], 2, ChannelConfig()),
    ([Exchange("block1", "block3")], 1, ChannelConfig("linear_taper_gelu")),
    ([Exchange("block3", "block1")], 3, ChannelConfig(decoder="linear_taper_gelu")),
    ([Exchange("block2", "block3")], 1, GELU),
    ([Exchange("block3", "block2")], 2, GELU),
    ([Exchange("block2", "block3")], 1, ChannelConfig(mixer="self")),
    ([Exchange("block3", "block2")], 2, ChannelConfig(mixer="mean")),
    ([Exchange("block2", "block3")], 1, ChannelConfig(mixer="rmsnorm_attention")),
    ([Exchange("block3", "block1")], 2, ChannelConfig(mixer="rmsnorm_attention")),
    ([Exchange("block1", "block2")], 1, ChannelConfig("pooled", decoder="broadcast")),
    ([Exchange("block1", "block3")], 1, ChannelConfig("spatial", decoder="spatial")),
    ([Exchange("block2", "block1")], 1, ChannelConfig("pooled", decoder="broadcast")),
    ([Exchange("block3", "block1")], 3, ChannelConfig("spatial", decoder="spatial")),
    ([], 1, ChannelConfig()),
  ],
)
def test_untrained_channel_is_bagging(exchanges, rounds, config):
  model = build_ensemble(
    random_experts(), exchanges, config, (3, 32, 32), rounds, seed=0
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
  shapes = {"block1": (16, 14, 14), "block2": (32, 7, 7), "block3": (64, 3, 3)}
  for a, b in (("block1", "block2"), ("block2", "block3"), ("block1", "block3")):
    there = count(build_channel(config, shapes[a], shapes[b], num_experts=1))
    back = count(build_channel(config, shapes[b], shapes[a], num_experts=1))
    size = math.prod(shapes[a]) + math.prod(shapes[b])
    assert there == back == r * (size + 2 * k + 2 * v + 2) + 2 * k + v


ARGS = ["--arm", "a", "--trial-seeds", "0"]


def test_runs_record_the_code_they_ran_from():
  version = code_version()
  assert len(version["git_commit"]) == 40 and version["git_dirty"] in ("true", "false")


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


def mix(mixer, query, key, value=None):
  value = torch.randn(*query.shape[:-1], 64) if value is None else value
  return MIXERS[mixer](ChannelConfig())(query, key, value)


def test_rmsnorm_attention_ignores_the_size_of_each_query_and_key():
  """Stretching every query and key by its own factor moves attention's
  weights and leaves rmsnorm_attention's where they are."""
  torch.manual_seed(0)
  q, k = torch.randn(2, 4, 3, 1, 16)
  stretch_q, stretch_k = 0.1 + 20 * torch.rand(2, 4, 3, 1, 1)
  longer = q * stretch_q, k * stretch_k
  assert torch.allclose(
    mix("rmsnorm_attention", q, k)[1], mix("rmsnorm_attention", *longer)[1], atol=1e-6
  )
  assert not torch.allclose(mix("attention", q, k)[1], mix("attention", *longer)[1])


def test_rmsnorm_attention_is_attention_at_the_length_it_rescales_to():
  torch.manual_seed(0)
  q, k = torch.randn(2, 4, 3, 1, 16)
  q, k = (x / x.norm(dim=-1, keepdim=True) * 16**0.5 for x in (q, k))
  v = torch.randn(4, 3, 1, 64)
  for got, want in zip(mix("rmsnorm_attention", q, k, v), mix("attention", q, k, v)):
    assert torch.allclose(got, want, atol=1e-5)


def test_rmsnorm_attention_can_still_settle_on_one_sender():
  """docs/testbed/channel.md §8: with a key_dim of 16 and 3 experts, a receiver's
  largest weight can reach e^8 / (e^8 + 2), which is about 0.9993, and the
  lowest entropy is about 0.006. So routing as sharp as a fixed wire stays
  possible, and softness isn't forced by construction."""
  u = torch.zeros(16)
  u[0] = 1
  q = u.expand(1, 3, 1, 16)
  k = torch.stack([u, -u, -u])[None, :, None]
  (weights,) = mix("rmsnorm_attention", q, k)[1:]
  top = weights[0, 0, 0, 0, 0].item()
  assert top == pytest.approx(math.exp(8) / (math.exp(8) + 2), abs=1e-6)
  assert top > 0.999
  rest = (1 - top) / 2
  entropy = -top * math.log(top) - 2 * rest * math.log(rest)
  assert entropy == pytest.approx(0.006, abs=1e-3)


def test_rmsnorm_attention_starts_from_attentions_weights():
  """It has no weights of its own, so on the same seed both channels start
  identical, and a difference between the two arms is the normalising."""
  channels = {}
  for mixer in ("attention", "rmsnorm_attention"):
    torch.manual_seed(0)
    channels[mixer] = build_channel(
      ChannelConfig(mixer=mixer), (32, 7, 7), (64, 3, 3), num_experts=3
    ).state_dict()
  plain, normed = channels.values()
  assert plain.keys() == normed.keys()
  assert all(torch.equal(plain[name], normed[name]) for name in plain)


def test_rmsnorm_attention_ignores_the_scale_of_query_and_key():
  features = [torch.randn(4, 32, 7, 7) for _ in range(3)]
  plain, normed = (
    build_channel(ChannelConfig(mixer=m), (32, 7, 7), (16, 14, 14), num_experts=3)
    for m in ("attention", "rmsnorm_attention")
  )
  with torch.no_grad():
    plain_before = plain.read(features).weights
    normed_before = normed.read(features).weights
    for channel in (plain, normed):
      for encoder in channel.encoders:
        for layer in (encoder.query, encoder.key):
          layer.weight.mul_(10)
          layer.bias.mul_(10)
    assert not torch.allclose(plain.read(features).weights, plain_before, atol=1e-5)
    assert torch.allclose(normed.read(features).weights, normed_before, atol=1e-5)


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
  [([Exchange("block1", "block2")], 1), ([Exchange("block2", "block1")], 2)],
)
def test_only_the_channel_trains(exchanges, rounds):
  model = build_ensemble(
    random_experts(), exchanges, ChannelConfig(), (3, 32, 32), rounds, seed=0
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
    ([Exchange("block2", "block2")], 1),  # reads and writes at one point
    ([Exchange("block1", "block2"), Exchange("block2", "block3")], 1),  # shares a point
    ([Exchange("block1", "block3"), Exchange("block2", "block3")], 1),  # overlaps
    ([Exchange("block3", "block1"), Exchange("block2", "block1")], 1),  # two backward
    ([Exchange("block1", "block2")], 2),  # rounds on a forward exchange
    ([Exchange("block1", "head")], 1),  # the head's output isn't a point
  ],
)
def test_invalid_exchanges_are_rejected(exchanges, rounds):
  with pytest.raises(ValueError):
    build_ensemble(
      random_experts(), exchanges, ChannelConfig(), (3, 32, 32), rounds, seed=0
    )


def test_negative_correlation_trains_a_pool_jointly():
  pool = NegativeCorrelation(lam=0.5).train_pool(
    "tiny_vit", fake_data(), 3, seed=0, budget=Budget(1, 50)
  )
  assert len(pool) == 3
  assert not any(e.model.training for e in pool)
  assert not torch.equal(
    pool[0].model.stages["head"].fc.weight, pool[1].model.stages["head"].fc.weight
  )


def test_untrained_experts_are_the_weights_other_schemes_start_from(tmp_path):
  data = fake_data()
  untrained = get_pool(
    Untrained(), "tiny_resnet", data, 3, seed=0, budget=Budget(1, 50), root=tmp_path
  )
  no_training = get_pool(
    SeedOnly(), "tiny_resnet", data, 3, seed=0, budget=Budget(0, 50), root=tmp_path
  )
  trained = get_pool(
    SeedOnly(), "tiny_resnet", data, 3, seed=0, budget=Budget(1, 50), root=tmp_path
  )
  assert untrained.digest == no_training.digest != trained.digest
  assert not any(e.model.training for e in untrained.experts)


def test_a_group_can_hold_one_expert_more_than_once(tmp_path):
  data, budget = fake_data(), Budget(1, 50)
  pool = get_pool(
    SeedOnly(), "tiny_resnet", data, 3, seed=0, budget=budget, root=tmp_path
  )
  group = ClonesOfFirst().select(pool, 3, seed=0)
  assert group == [0, 0, 0]
  experts = group_experts(pool, group, 0, "tiny_resnet", data, pool_seed=0)
  assert len({id(e.model) for e in [*experts, pool.experts[0]]}) == 4  # all its own
  assert weights_digest(experts) == weights_digest([pool.experts[0]] * 3)


def test_untrained_members_replace_only_the_end_of_the_group(tmp_path):
  data, budget = fake_data(), Budget(1, 50)
  pool = get_pool(
    SeedOnly(), "tiny_resnet", data, 3, seed=0, budget=budget, root=tmp_path
  )
  before = get_pool(
    Untrained(), "tiny_resnet", data, 3, seed=0, budget=budget, root=tmp_path
  )
  experts = group_experts(pool, [0, 1, 2], 1, "tiny_resnet", data, pool_seed=0)
  assert weights_digest(experts[:2]) == weights_digest(pool.experts[:2])
  assert weights_digest(experts[2:]) == weights_digest(before.experts[2:])
  assert experts[2].profile == {"untrained": True}
  assert weights_digest(pool.experts) == pool.digest  # the pool itself is untouched


@pytest.mark.parametrize("members", [-1, 3])
def test_a_group_keeps_a_trained_expert(members):
  args = parse_args([*ARGS, "--untrained-members", str(members)])
  with pytest.raises(SystemExit):
    check_flags(args)


@pytest.mark.parametrize(
  "exchanges, rounds",
  [([Exchange("block2", "block3")], 1), ([Exchange("block3", "block1")], 1)],
)
def test_clones_with_identical_channels_have_nothing_to_say(exchanges, rounds):
  """With identical experts and identical encoders and decoders, every expert
  sends the same message, so every mixer delivers it unchanged. All the
  mixers then train to the same answers, and the experts stay identical. Any
  difference between mixers on clones comes from the channels' initialisation."""
  data = fake_data()
  (expert,) = random_experts(1)
  x, y = data.joint.x[:64], data.joint.y[:64]
  answers = {}
  for mixer in ("self", "mean", "attention", "rmsnorm_attention"):
    model = build_ensemble(
      [copy.deepcopy(expert) for _ in range(3)],
      exchanges,
      ChannelConfig(mixer=mixer),
      (3, 32, 32),
      rounds,
      seed=0,
    )
    (channel,) = model.channels
    for parts in (channel.encoders, channel.decoders):
      for part in parts[1:]:
        part.load_state_dict(parts[0].state_dict())
    optimizer = torch.optim.Adam(model.channels.parameters(), lr=1e-3)
    model.train()
    for _ in range(3):
      loss = loss_fn(model(x).logits, y, "ensemble")
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    model.eval()
    with torch.no_grad():
      answers[mixer] = model(x).logits
      assert not torch.allclose(answers[mixer], model.bagging(x).logits, atol=1e-3)
  first = answers["self"]
  assert torch.allclose(first[:, 0], first[:, 2], atol=1e-4)
  assert torch.allclose(first, answers["mean"], atol=1e-4)
  assert torch.allclose(first, answers["attention"], atol=1e-4)
  assert torch.allclose(first, answers["rmsnorm_attention"], atol=1e-4)


def test_pool_cache_returns_the_same_experts(tmp_path):
  data, budget = fake_data(), Budget(1, 50)
  first = get_pool(
    SeedOnly(), "tiny_resnet", data, 2, seed=0, budget=budget, root=tmp_path
  )
  again = get_pool(
    SeedOnly(), "tiny_resnet", data, 2, seed=0, budget=budget, root=tmp_path
  )
  other = get_pool(
    ClassSkew(0.5), "tiny_resnet", data, 2, seed=0, budget=budget, root=tmp_path
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
    return get_pool(
      scheme, "tiny_resnet", data, 2, seed=0, budget=budget, root=tmp_path
    )

  with ThreadPoolExecutor(2) as executor:
    first, second = executor.map(ask, range(2))
  assert len(scheme.calls) == 1  # the second waited for the first, then loaded
  assert first.digest == second.digest
  assert [p.name for p in tmp_path.glob("*.pt")] == [f"{first.key}.pt"]
  assert not list(tmp_path.glob("*.tmp"))
