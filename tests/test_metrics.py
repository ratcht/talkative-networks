"""
Construction checks for the metrics (docs/testbed/metrics.md): inputs whose
readings are known exactly, the streamed statistics against the same thing
computed in one go, and a short training run whose every logged key must be in
the registry and documented.

  uv run pytest tests/test_metrics.py
"""

import math
import re

import pytest
import torch
from fakes import fake_data, random_experts

from testbed.communication import ChannelConfig, Exchange, build_ensemble
from testbed.communication.ensemble import average
from testbed.data import REPO_ROOT
from testbed.metrics import (
  METRICS,
  joint_answer_shift,
  marginal_answer_shift,
)
from testbed.metrics.answers import answer_shift_kl
from testbed.metrics.exchanges import InjectionSums, participation_ratio, routing
from testbed.metrics.optimisation import OptimisationMeter, component
from testbed.metrics.registry import find
from testbed.train import JointConfig, loss_fn, train

DOCS = REPO_ROOT / "docs" / "testbed" / "metrics.md"


def logits(n=500, m=3, classes=10, seed=0):
  gen = torch.Generator().manual_seed(seed)
  return (
    torch.randn(n, m, classes, generator=gen),
    torch.randint(0, classes, (n,), generator=gen),
  )


def test_net_is_the_change_in_accuracy():
  pre, y = logits(seed=0)
  post, _ = logits(seed=1)
  shift = joint_answer_shift(pre, post, y)
  assert shift["net"] == pytest.approx(shift["post_accuracy"] - shift["pre_accuracy"])
  assert shift["corrected"] <= 1 - shift["pre_accuracy"]
  assert shift["corrupted"] <= shift["pre_accuracy"]


def test_no_communication_shifts_nothing():
  pre, y = logits()
  shift = joint_answer_shift(pre, pre, y)
  assert shift["corrected"] == shift["corrupted"] == shift["net"] == 0
  assert shift["agreement_pre"] == shift["agreement_post"]
  assert all(v == 0 for v in marginal_answer_shift(pre, pre, y).values())
  assert answer_shift_kl(pre, pre) == pytest.approx(0, abs=1e-7)


def test_one_expert_changing_is_its_own_marginal():
  pre, y = logits(seed=0)
  post = pre.clone()
  post[:, 1] = logits(seed=1)[0][:, 1]
  joint, marginal = (
    joint_answer_shift(pre, post, y),
    marginal_answer_shift(pre, post, y),
  )
  for kind in ("corrected", "corrupted", "net"):
    assert marginal[f"{kind}/expert1"] == joint[kind]
    assert marginal[f"{kind}/expert0"] == marginal[f"{kind}/expert2"] == 0


def test_participation_ratio_counts_directions():
  def pr(*energies):  # rows along orthogonal axes, with these squared lengths
    rows = torch.diag(torch.tensor(energies, dtype=torch.float64).sqrt())
    return participation_ratio(rows.T @ rows).item()

  assert pr(1.0) == pytest.approx(1)
  assert pr(1.0, 1.0, 1.0) == pytest.approx(3)
  assert pr(0.9, 0.1) == pytest.approx(1 / 0.82)  # 1.22: nearly one direction
  assert pr(9.0, 1.0) == pytest.approx(pr(0.9, 0.1))  # scale doesn't matter
  assert pr(0.0, 0.0) == 0


def test_streamed_injection_statistics_match_one_pass():
  gen = torch.Generator().manual_seed(0)
  d = torch.randn(300, 4, 5, 5, generator=gen) + torch.randn(4, 5, 5, generator=gen)
  sums = InjectionSums()
  for batch in d.split(64):
    sums.add(batch)
  got = sums.result()

  d = d.double()
  resid = d - d.mean((-2, -1), keepdim=True)
  # v2's form of the rank: a Gram over examples, one channel at a time
  per_channel = resid.flatten(2).transpose(0, 1)  # (C, N, P)
  rank = participation_ratio(per_channel @ per_channel.transpose(1, 2)).mean()
  assert got["norm"] == pytest.approx(d.flatten(1).norm(dim=1).mean().item())
  assert got["spread"] == pytest.approx((d - d.mean(0)).norm().item() / d.norm().item())
  assert got["resid_frac"] == pytest.approx(resid.norm().item() / d.norm().item())
  assert got["rank"] == pytest.approx(rank.item())


@pytest.mark.parametrize(
  "make, expected",
  [
    # nothing injected: every channel's state at step 0
    (lambda g, s: torch.zeros(50, 3, 4, 4), {"spread": 0, "resid_frac": 0, "rank": 0}),
    # the same value at every position (a broadcast decoder): no spatial part
    (
      lambda g, s: g[:, :, None, None].expand(-1, -1, 4, 4),
      {"resid_frac": 0, "rank": 0},
    ),
    # one fixed spatial pattern, its strength set per example: rank exactly 1
    (lambda g, s: g[:, :, None, None] * s, {"rank": 1}),
    # the same injection for every example: a constant shift, no spread
    (lambda g, s: s.expand(50, -1, -1, -1), {"spread": 0}),
  ],
)
def test_injection_statistics_by_construction(make, expected):
  gen = torch.Generator().manual_seed(0)
  g, s = torch.randn(50, 3, generator=gen), torch.randn(3, 4, 4, generator=gen)
  sums = InjectionSums()
  sums.add(make(g, s))
  got = sums.result()
  for key, value in expected.items():
    assert got[key] == pytest.approx(value, abs=1e-9), key


def test_routing_by_construction():
  n, m = 6, 3
  # receiver r always listens most to sender r + 1
  weights = torch.full((n, m, m), 0.1)
  for r in range(m):
    weights[:, r, (r + 1) % m] = 0.8
  # every expert is right before communication, so every sender listened to was right
  y = torch.zeros(n, dtype=torch.long)
  pre = torch.zeros(n, m, 2)
  pre[..., 0] = 1
  metrics, routes = routing(weights, pre, pre, y)
  assert metrics["self_attn_rate"] == 0
  assert metrics["routing_mode_share/joint"] == 1
  assert all(metrics[f"routing_mode_share/expert{r}"] == 1 for r in range(m))
  assert metrics["routed_correct/all"] == 1
  assert "routed_correct/corrected" not in metrics  # nobody's answer changed
  assert "route_flip_rate" not in metrics  # nothing to compare with yet
  assert routing(weights, pre, pre, y, previous=routes)[0]["route_flip_rate"] == 0
  assert (
    routing(weights, pre, pre, y, previous=routes.roll(1, 1))[0]["route_flip_rate"] == 1
  )


def test_gradient_norms_split_the_total():
  model = build_ensemble(
    random_experts(), [Exchange("stage1", "stage2")], ChannelConfig(), (1, 28, 28), 1, 0
  )
  meter = OptimisationMeter(model.channels)
  data = fake_data()
  loss_fn(model(data.joint.x[:64]).logits, data.joint.y[:64], "ensemble").backward()
  meter.before_step()
  params = list(model.channels.parameters())
  before = torch.cat([p.detach().flatten() for p in params]).double()
  with torch.no_grad():
    for p in params:
      p.add_(0.01)  # a known step
  meter.after_step()
  out = meter.result()

  grads = torch.cat([p.grad.flatten() for p in params]).double()
  parts = [v for k, v in out.items() if k.startswith("grad_norm/") and "total" not in k]
  assert out["grad_norm/total"] == pytest.approx(grads.norm().item())
  assert out["grad_norm/total"] ** 2 == pytest.approx(sum(v**2 for v in parts))
  assert out["update_ratio/total"] == pytest.approx(
    0.01 * len(before) ** 0.5 / before.norm().item()
  )


def test_parameters_are_grouped_by_where_they_sit():
  assert component("0.encoders.2.query.weight") == "encoder.query/ex0"
  assert component("0.encoders.1.position") == "encoder.position/ex0"
  assert component("1.decoders.0.mlp.2.bias") == "decoder.mlp/ex1"
  assert component("0.mixer.proj.weight") == "mixer/ex0"


class Recorder:
  def __init__(self):
    self.steps: dict[int, dict[str, float]] = {}

  def log_metrics(self, metrics, step=None):
    self.steps.setdefault(step, {}).update(metrics)

  def log_params(self, params):
    pass

  def set_tags(self, tags):
    pass


ARMS = {
  "forward": ([Exchange("stage1", "stage2")], 1, ChannelConfig()),
  "forward_spatial": (
    [Exchange("stage1", "stage3")],
    1,
    ChannelConfig(encoder="spatial", decoder="spatial"),
  ),
  "backward": ([Exchange("stage3", "stage1")], 2, ChannelConfig()),
  "bagging": ([], 1, ChannelConfig()),
}


@pytest.fixture(scope="module", params=ARMS)
def run(request):
  exchanges, rounds, config = ARMS[request.param]
  data = fake_data(n=400)
  model = build_ensemble(
    random_experts(), exchanges, config, data.input_shape, rounds, 0
  )
  recorder = Recorder()
  history = train(model, data, JointConfig(epochs=2, batch_size=50), 0, recorder)
  return request.param, model, data, recorder, history


def test_every_logged_key_is_in_the_registry(run):
  _, _, _, recorder, history = run
  assert [recorder.steps[s] for s in range(len(history))] == history
  undocumented = {key for step in history for key in step if find(key) is None}
  assert not undocumented


def test_step_zero_is_bagging(run):
  arm, model, data, _, history = run
  first = history[0]
  with torch.no_grad():
    bagging = average(model.bagging(data.test.x).logits).argmax(-1) == data.test.y
  assert first["test/accuracy"] == pytest.approx(bagging.float().mean().item())
  for split in ("joint_val", "test"):
    assert first[f"{split}/joint_answer_shift/net"] == 0
  assert first["joint_val/answer_shift_kl"] == pytest.approx(0, abs=1e-7)
  for key, value in first.items():
    if key.startswith(("msg_norm/injection", "inj_")):
      assert value == 0, key
    assert not key.startswith(("train/", "route_flip_rate", "grad_norm")), key
  assert len(history) == (1 if arm == "bagging" else 3)


def test_train_answers_line_up_with_their_examples(run):
  arm, model, data, _, history = run
  if arm == "bagging":
    return
  with torch.no_grad():
    bagging = average(model.bagging(data.joint.x).logits).argmax(-1) == data.joint.y
  for step in history[1:]:
    # the epoch's examples are shuffled; pre must be shuffled with them
    assert step["train/joint_answer_shift/pre_accuracy"] == pytest.approx(
      bagging.float().mean().item()
    )


def test_each_exchange_is_measured_where_it_ran(run):
  arm, _, _, _, history = run
  sites = {
    re.search(r"ex\d+(/round\d+)?$", key)[0]
    for key in history[-1]
    if key.startswith("msg_norm/")
  }
  expected = {
    "forward": {"ex0"},
    "forward_spatial": {"ex0"},
    "backward": {"ex0/round0", "ex0/round1"},
    "bagging": set(),
  }[arm]
  assert sites == expected
  last = history[-1]
  for site in sites:
    assert last[f"msg_norm/injection/{site}"] > 0  # the channel learned to write
    assert f"route_flip_rate/{site}" in last
    if arm == "forward_spatial":
      assert last[f"inj_resid_frac/{site}"] > 0
    else:  # broadcast: one value per channel, the same at every position
      assert last[f"inj_resid_frac/{site}"] < 1e-12
      assert last[f"inj_rank/{site}"] == 0


def test_every_registry_metric_is_documented():
  headings = [line for line in DOCS.read_text().splitlines() if line.startswith("#")]
  for metric in METRICS:
    assert any(f"`{metric.name}`" in h for h in headings), metric.name


def test_attention_stays_in_bounds(run):
  _, model, _, _, history = run
  m = len(model.experts)
  for step in history:
    for key, value in step.items():
      if key.startswith("attn_entropy/"):  # 0 for one sender, log M for all equally
        assert -1e-9 <= value <= math.log(m) + 1e-6, key
      if key.startswith("attn_top1/"):  # 1/M for all equally, 1 for one sender
        assert 1 / m - 1e-6 <= value <= 1 + 1e-6, key
