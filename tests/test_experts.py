"""v4's experts: the residual stream, the families' recipes, augmentation and
the image loading (docs/testbed/experts.md)."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn as nn
from fakes import fake_data, random_experts

from testbed.communication import ChannelConfig, Exchange, build_ensemble
from testbed.communication.channel import positions
from testbed.data import scaled
from testbed.experts import RECIPES, Expert, SeedOnly, build, recipe
from testbed.experts.recipes import Budget, augment, optimiser, rate
from testbed.experts.schemes import new_expert

CIFAR = (3, 32, 32)


@pytest.mark.parametrize(
  "arch, params, stream",
  [("resnet", 896_164, (64, 16, 16)), ("vit", 1_363_012, (96, 65))],
)
def test_the_two_families_at_their_v4_sizes(arch, params, stream):
  """The sizes the v4 selection measured, on CIFAR-100."""
  model = build(arch, CIFAR, 100)
  assert sum(p.numel() for p in model.parameters()) == params
  assert model.points == ["embed"] + [f"block{i}" for i in range(1, 13)]
  assert model.stream_shape(CIFAR) == stream


@pytest.mark.parametrize("arch", ["tiny_resnet", "tiny_vit"])
def test_running_in_pieces_is_running_whole(arch):
  model = random_experts(1, arch)[0]
  x = fake_data().test.x[:8]
  whole = model(x)
  for point in model.points:
    stream = model.run(x, until=point)
    assert torch.allclose(model.run(stream, after=point), whole, atol=1e-5)


def test_every_block_only_adds_to_the_stream():
  """A block whose residual branch outputs zero hands the stream on unchanged."""
  for arch in ("tiny_resnet", "tiny_vit"):
    model = random_experts(1, arch)[0]
    block = model.stages["block2"]
    last = block.conv2 if arch == "tiny_resnet" else None
    if last is not None:
      nn.init.zeros_(last.weight)
    else:
      for layer in (block.attn.proj, block.mlp[-1]):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    stream = model.run(fake_data().test.x[:4], until="block1")
    assert torch.allclose(model.run(stream, after="block1", until="block2"), stream)


def test_the_vit_stream_is_channels_first_with_the_class_token_first():
  model = random_experts(1, "tiny_vit")[0]
  x = fake_data().test.x[:2]
  stream = model.run(x, until="embed")
  assert stream.shape == (2, 32, 65)
  rows = positions(stream)  # one row per token, as for a grid cell
  assert rows.shape == (2, 65, 32)
  cls = model.stages["embed"].cls[0, :, 0] + model.stages["embed"].pos[0, :, 0]
  assert torch.allclose(rows[:, 0], cls.expand(2, -1))


def test_a_channel_refuses_experts_without_a_uniform_stream():
  """Points of different shapes mean some point isn't the residual stream."""
  experts = [
    Expert(
      {
        "a": nn.Conv2d(3, 4, 3, padding=1),
        "b": nn.Conv2d(4, 8, 3, stride=2, padding=1),
        "c": nn.Conv2d(8, 8, 3, padding=1),
        "head": nn.Sequential(nn.Flatten(), nn.LazyLinear(10)),
      }
    )
    for _ in range(2)
  ]
  for e in experts:
    e(torch.zeros(1, *CIFAR))  # materialise the lazy head
  with pytest.raises(ValueError, match="stream"):
    build_ensemble(experts, [Exchange("a", "c")], ChannelConfig(), CIFAR, 1, seed=0)


# ---------------------------------------------------------------- recipes


def test_the_default_budget_is_v3s_recipe():
  budget = Budget()
  assert (budget.optimizer, budget.lr, budget.schedule) == ("adam", 1e-3, "constant")
  assert not budget.augment and not budget.bf16
  assert all(rate(step, 100, budget) == 1e-3 for step in (0, 50, 99))


def test_the_cosine_schedule_warms_up_then_decays_to_zero():
  budget = RECIPES["vit"]
  total = 1000
  warm = int(budget.warmup * total)
  assert rate(0, total, budget) == pytest.approx(budget.lr / warm)
  assert rate(warm - 1, total, budget) == pytest.approx(budget.lr)
  assert rate(warm, total, budget) == pytest.approx(budget.lr)
  middle = warm + (total - warm) // 2
  assert rate(middle, total, budget) == pytest.approx(budget.lr / 2, rel=1e-2)
  assert rate(total - 1, total, budget) < 1e-5 * budget.lr * total
  resnet = RECIPES["resnet"]
  assert rate(0, total, resnet) == resnet.lr  # no warm-up


def test_adamw_leaves_norms_biases_and_the_vits_tokens_out_of_weight_decay():
  model = build("vit", CIFAR, 100)
  opt = optimiser([model], RECIPES["vit"])
  decayed, exempt = ({id(p) for p in group["params"]} for group in opt.param_groups)
  names = {id(p): n for n, p in model.named_parameters()}
  assert opt.param_groups[0]["weight_decay"] == 0.05
  assert opt.param_groups[1]["weight_decay"] == 0.0
  for pid in exempt:
    name = names[pid]
    p = dict(model.named_parameters())[name]
    assert p.ndim < 2 or name.endswith((".cls", ".pos")), name
  assert all(dict(model.named_parameters())[names[pid]].ndim >= 2 for pid in decayed)
  assert len(decayed) + len(exempt) == len(names)


def test_sgd_is_nesterov_with_weight_decay_on_everything():
  model = build("resnet", CIFAR, 100)
  (group,) = optimiser([model], RECIPES["resnet"]).param_groups
  assert group["nesterov"] and group["momentum"] == 0.9
  assert group["weight_decay"] == 5e-4
  assert len(group["params"]) == len(list(model.parameters()))


def test_unregistered_recipes_fall_back_to_the_default():
  assert recipe("tiny_resnet") == Budget()
  assert recipe("resnet").optimizer == "sgd" and recipe("vit").optimizer == "adamw"


@pytest.mark.parametrize("arch", ["tiny_resnet", "tiny_vit"])
def test_an_expert_trains_with_its_familys_recipe(arch):
  """The full recipe, short: cosine, its optimiser, augmentation, bf16."""
  data = fake_data()
  family = RECIPES["resnet" if arch == "tiny_resnet" else "vit"]
  budget = replace(family, epochs=1, batch_size=50)
  (start,) = SeedOnly().recipes(data, 1, seed=0)
  before = new_expert(arch, data, start.seed).state_dict()
  (trained,) = SeedOnly().train_pool(arch, data, 1, seed=0, budget=budget)
  after = trained.model.state_dict()
  assert not trained.model.training
  assert any(not torch.equal(before[k], after[k]) for k in before)
  assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())


# ---------------------------------------------------------------- augmentation


def test_augmentation_crops_and_flips_each_image_on_its_own():
  x = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).view(2, 3, 32, 32)
  fill = torch.full((3, 1, 1), -7.0)
  gen = torch.Generator().manual_seed(0)
  outs = [augment(x, fill, gen) for _ in range(20)]
  assert all(o.shape == x.shape for o in outs)
  # every value is the image's own or the fill, and padding does show up
  for o in outs:
    for i in range(2):
      inside = o[i][o[i] != -7.0]
      assert torch.isin(inside, x[i]).all()
  assert any((o == -7.0).any() for o in outs)
  # the two images are cropped and flipped independently
  assert any(not torch.equal(o[0] - x[0, 0, 0, 0], o[1] - x[1, 0, 0, 0]) for o in outs)


def test_augmentation_is_set_by_the_generator():
  x = torch.randn(4, 3, 32, 32)
  fill = torch.zeros(3, 1, 1)
  a = augment(x, fill, torch.Generator().manual_seed(3))
  b = augment(x, fill, torch.Generator().manual_seed(3))
  assert torch.equal(a, b)


def test_without_crop_or_flip_the_image_is_unchanged():
  """At offset (p, p) and no flip, augment returns the image itself: check
  that some draw over many reproduces it exactly."""
  x = torch.randn(1, 3, 32, 32)
  gen = torch.Generator().manual_seed(0)
  assert any(torch.equal(augment(x, torch.zeros(3, 1, 1), gen), x) for _ in range(400))


# ---------------------------------------------------------------- images


def test_scaled_reads_both_torchvision_layouts():
  gray = torch.randint(0, 256, (5, 28, 28), dtype=torch.uint8)  # MNIST
  rgb = np.random.default_rng(0).integers(
    0, 256, (5, 32, 32, 3), dtype=np.uint8
  )  # CIFAR
  assert scaled(gray).shape == (5, 1, 28, 28)
  out = scaled(rgb)
  assert out.shape == (5, 3, 32, 32)
  assert out[2, 1, 4, 7].item() == pytest.approx(rgb[2, 4, 7, 1] / 255)
  assert 0 <= out.min() and out.max() <= 1


def test_fake_data_pads_with_zero():
  data = fake_data()
  assert data.fill.shape == (3, 1, 1) and not data.fill.any()


def test_the_steps_count_a_partial_last_batch():
  from testbed.experts.recipes import steps_per_epoch

  assert steps_per_epoch(25_000, 128) == math.ceil(25_000 / 128)
  assert steps_per_epoch(256, 128) == 2


class Recording:
  """A sampler that remembers every epoch it drew."""

  def __init__(self, inner):
    self.inner, self.orders = inner, []

  def epoch(self, gen):
    order = self.inner.epoch(gen)
    self.orders.append(order.clone())
    return order

  def __len__(self):
    return len(self.inner)


def test_augmentation_never_shifts_the_batch_order():
  """Crops come from their own seed stream, so turning augmentation on or
  off leaves every epoch's order as it was."""
  from testbed.data import Shuffle
  from testbed.experts.schemes import Recipe, fit

  data = fake_data()
  orders = {}
  for on in (False, True):
    sampler = Recording(Shuffle(len(data.expert)))
    fit("tiny_resnet", data, Recipe(7, sampler, {}), Budget(3, 50, augment=on))
    orders[on] = sampler.orders
  assert len(orders[True]) == 3
  assert all(torch.equal(a, b) for a, b in zip(orders[False], orders[True]))


# ---------------------------------------------------------------- the class token


def test_the_class_token_is_not_a_grid_position():
  from testbed.communication.channel import onto_grid

  vit_model, resnet_model = random_experts(1, "tiny_vit")[0], random_experts(1)[0]
  assert vit_model.tokens == 1 and resnet_model.tokens == 0
  stream = torch.randn(2, 32, 65)
  rows = positions(stream, tokens=1)
  assert rows.shape == (2, 64, 32)
  assert torch.equal(rows, stream[:, :, 1:].transpose(1, 2))
  back = onto_grid(rows, stream, tokens=1)
  assert not back[:, :, 0].any() and torch.equal(back[:, :, 1:], stream[:, :, 1:])


@pytest.mark.parametrize("encoder", ["pooled", "spatial"])
def test_position_encoders_never_read_the_class_token(encoder):
  experts = random_experts(3, "tiny_vit")
  model = build_ensemble(
    experts, [Exchange("block1", "block2")], ChannelConfig(encoder=encoder), CIFAR, 1, 0
  )
  enc = model.channels[0].encoders[0]
  stream = experts[0].run(fake_data().test.x[:4], until="block1")
  changed = stream.clone()
  changed[:, :, 0] += 100.0
  assert torch.allclose(enc(stream).value, enc(changed).value)


@pytest.mark.parametrize("decoder", ["broadcast", "spatial"])
def test_position_decoders_write_nothing_into_the_class_token(decoder):
  experts = random_experts(3, "tiny_vit")
  model = build_ensemble(
    experts, [Exchange("block1", "block2")], ChannelConfig(decoder=decoder), CIFAR, 1, 0
  )
  for channel in model.channels:
    for dec in channel.decoders:
      for p in dec.parameters():  # leave the zero start, so something is written
        torch.nn.init.normal_(p)
  out = model(fake_data().test.x[:4])
  (trace,) = out.traces
  assert trace.tokens == 1
  for injection in trace.injection:
    assert not injection[:, :, 0].any()
    assert injection[:, :, 1:].abs().sum() > 0


def test_the_spatial_injection_metrics_skip_the_class_token():
  """An injection the same at every grid position varies over none of them,
  whatever goes into the class token."""
  from testbed.metrics.exchanges import InjectionSums

  injection = torch.ones(4, 8, 65)
  injection[:, :, 0] = 5.0  # the class token, off the grid
  sums = InjectionSums(tokens=1)
  sums.add(injection)
  assert sums.result()["resid_frac"] == pytest.approx(0.0, abs=1e-12)
  counted = InjectionSums()  # had the token counted as a position
  counted.add(injection)
  assert counted.result()["resid_frac"] > 0.1


# ---------------------------------------------------------------- building pools


def test_pools_only_refuses_a_channel_or_gate():
  from testbed.run import check_flags, parse_args

  base = ["--arm", "pools", "--trial-seeds", "0", "--pools-only"]
  for extra in (["--exchange", "block1", "block2"], ["--gate", "linear_taper_gelu"]):
    with pytest.raises(SystemExit):
      check_flags(parse_args(base + extra))


def test_pools_only_fills_the_cache_and_trains_nothing_else(tmp_path, monkeypatch):
  from functools import partial

  import testbed.run as run
  from testbed.experts import get_pool

  def no_tracker(*args, **kwargs):
    raise AssertionError("--pools-only must not open an MLflow run")

  monkeypatch.setattr(run, "load", lambda *a, **k: fake_data())
  monkeypatch.setattr(run, "get_pool", partial(get_pool, root=tmp_path))
  monkeypatch.setattr(run, "MlflowTracker", no_tracker)
  monkeypatch.setattr(run, "train", no_tracker)
  run.main(
    ["--arm", "pools", "--pools-only", "--arch", "tiny_resnet", "--expert-epochs", "1"]
    + ["--scheme", "class_skew", "--trial-seeds", "0", "1", "--device", "cpu"]
  )
  assert len(list(tmp_path.glob("*.pt"))) == 2  # one pool per trial seed
