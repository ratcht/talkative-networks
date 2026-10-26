"""
The soft-gated MoE baseline (testbed/moe.py): it starts as bagging, its
answer is the weighted average the metrics recompute, and only its gate
trains.

  uv run pytest tests/test_moe.py
"""

import pytest
import torch
from fakes import fake_data, random_experts

from testbed.communication.ensemble import average
from testbed.moe import GateConfig, build_gated
from testbed.run import check_flags, parse_args
from testbed.train import loss_fn


@pytest.mark.parametrize("read", [None, "stage1", "stage2", "stage3"])
def test_an_untrained_gate_is_bagging(read):
  model = build_gated(random_experts(), GateConfig(read=read), (1, 28, 28), seed=0)
  assert model.read == (read or "stage3")  # the experts' last point by default
  x = fake_data().joint_val.x
  with torch.no_grad():
    out = model(x)
    assert torch.allclose(out.log_probs, model.bagging(x).log_probs, atol=1e-6)
    assert torch.allclose(out.weights, torch.full_like(out.weights, 1 / 3))


def train_a_little(model, steps: int = 20) -> None:
  data = fake_data()
  optimizer = torch.optim.Adam(model.trainable.parameters(), lr=1e-2)
  model.train()
  for _ in range(steps):
    out = model(data.joint.x[:64])
    loss = loss_fn(out.logits, data.joint.y[:64], "ensemble", out.weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_only_the_gate_trains_and_it_moves_off_equal_weights():
  model = build_gated(random_experts(), GateConfig(), (1, 28, 28), seed=0)
  experts_before = [p.clone() for p in model.experts.parameters()]
  train_a_little(model)
  assert all(p.grad is None for p in model.experts.parameters())
  assert all(
    torch.equal(a, b) for a, b in zip(experts_before, model.experts.parameters())
  )
  with torch.no_grad():
    weights = model(fake_data().joint_val.x).weights
  assert torch.allclose(weights.sum(-1), torch.ones(len(weights)))
  assert (weights - 1 / 3).abs().max() > 1e-3


def test_the_metrics_recompute_the_models_answer():
  """The answer metrics rebuild the ensemble's answer from the experts' logits
  and the weights; it must be the one the model gave."""
  model = build_gated(random_experts(), GateConfig(), (1, 28, 28), seed=0)
  train_a_little(model)
  with torch.no_grad():
    out = model(fake_data().joint_val.x)
  assert torch.allclose(average(out.logits, out.weights), out.log_probs, atol=1e-5)
  # with equal weights it is the plain average, exactly as without weights
  equal = torch.full_like(out.weights, 1 / 3)
  assert torch.allclose(average(out.logits, equal), average(out.logits), atol=1e-6)


def test_the_gates_size():
  """docs/testbed/moe.md: M readings of width r of every expert's map, then a
  score for each expert from all M r readings."""
  experts, r = random_experts(), GateConfig().width
  for read, size in {"stage1": 3136, "stage2": 1568, "stage3": 576}.items():
    gate = build_gated(experts, GateConfig(read=read), (1, 28, 28), seed=0).gate
    m = len(experts)
    assert (
      sum(p.numel() for p in gate.parameters()) == m * (size + 1) * r + (m * r + 1) * m
    )


def test_a_gate_reads_only_at_a_point():
  with pytest.raises(ValueError):
    build_gated(random_experts(), GateConfig(read="head"), (1, 28, 28), seed=0)


ARGS = ["--arm", "a", "--trial-seeds", "0"]


@pytest.mark.parametrize(
  "flags",
  [
    ["--gate", "linear_taper_gelu", "--exchange", "stage2", "stage3"],
    ["--gate", "linear_taper_gelu", "--loss", "per_expert"],
    ["--gate-read", "stage3"],  # no gate to read for
  ],
)
def test_the_command_line_refuses_what_a_gate_cant_be(flags):
  # each flag parses on its own; only the combination is refused
  args = parse_args([*ARGS, *flags])
  with pytest.raises(SystemExit):
    check_flags(args)
