"""
Train what combines the frozen experts: a communicating ensemble's channels,
or a soft-gated MoE's gate (testbed/moe.py). The experts stay frozen.

The loss is still an open choice (docs/docket.md):
  ensemble     NLL of the ensemble's prediction, the mean expert probability (as in v2)
  per_expert   mean of each expert's own cross-entropy

Every metric in docs/testbed/metrics.md is logged at step 0, before any
training, and again after every epoch, under the split it was measured on:
  joint      the split the channel trains on, over the epoch's batches, as
             the channel was at each batch
  joint_val  a full pass after the epoch, with every exchange's metrics and,
             for a backward exchange, the answers after each round
  test       a full pass after the epoch
Nothing here picks an epoch. Afterwards each run is boiled down to one number
per metric, its median over the 5 epochs around its best joint_val epoch
(testbed/results.py). Nothing selects on test.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

from testbed.communication.ensemble import CommunicatingEnsemble, Output, average
from testbed.data import Data, Shuffle
from testbed.moe import GatedEnsemble
from testbed.metrics import (
  ExchangeMeter,
  OptimisationMeter,
  accuracy,
  answer_grid,
  answer_shift_kl,
  expert_accuracy,
  joint_answer_shift,
  marginal_answer_shift,
)
from testbed.seeds import derive, generator
from testbed.tracking import Tracker


@dataclass(frozen=True)
class JointConfig:
  epochs: int = 10
  batch_size: int = 128
  lr: float = 1e-3
  loss: str = "ensemble"


def loss_fn(
  logits: torch.Tensor,
  y: torch.Tensor,
  kind: str,
  weights: torch.Tensor | None = None,
) -> torch.Tensor:
  """logits (B, M, classes), each expert's answer, and weights (B, M) when
  the ensemble weighs its experts' answers unequally."""
  if kind == "ensemble":
    return F.nll_loss(average(logits, weights), y)
  if kind == "per_expert":
    if weights is not None:
      raise ValueError(
        "per_expert can't train a gate: no expert's answer depends on it"
      )
    return F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(logits.shape[1]))
  raise ValueError(f"unknown loss {kind!r}")


def train(
  model: CommunicatingEnsemble | GatedEnsemble,
  data: Data,
  config: JointConfig,
  seed: int,
  tracker: Tracker,
) -> list[dict[str, float]]:
  """Train the model's trainable part. Returns the metrics logged at each step,
  step 0 first."""
  # the experts are frozen, so their answers without communication are fixed
  pre = {
    split: predict(model.bagging, getattr(data, split).x)[0]
    for split in ("joint", "joint_val", "test")
  }
  routes: dict[str, torch.Tensor | None] = {}  # for route_flip_rate
  history = []

  def log(metrics: dict[str, float]) -> None:
    tracker.log_metrics(metrics, step=len(history))
    history.append(metrics)

  log(evaluate(model, data, pre, config.loss, routes))
  params = list(model.trainable.parameters())
  if not params:  # no exchange: the ensemble is bagging, nothing to train
    return history

  optimizer = torch.optim.Adam(params, lr=config.lr)
  optimisation = OptimisationMeter(model.trainable)
  gen = generator(derive(seed, "batches"), data.device)
  for _ in range(config.epochs):
    model.train()
    order = Shuffle(len(data.joint)).epoch(gen)
    logits, weights = [], []
    for idx in order.split(config.batch_size):
      out = model(data.joint.x[idx])
      loss = loss_fn(out.logits, data.joint.y[idx], config.loss, out.weights)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimisation.before_step()
      optimizer.step()
      optimisation.after_step()
      logits.append(out.logits.detach())
      if out.weights is not None:
        weights.append(out.weights.detach())
    post, y = torch.cat(logits), data.joint.y[order]
    w = torch.cat(weights) if weights else None
    joint_metrics = prefix(
      "joint",
      {
        "loss": loss_fn(post, y, config.loss, w).item(),
        "accuracy": accuracy(post, y, w),
      }
      | prefix("answer_grid", answer_grid(post, y, w))
      | prefix(
        "joint_answer_shift", joint_answer_shift(pre["joint"][order], post, y, w)
      ),
    )
    log(
      joint_metrics
      | optimisation.result()
      | evaluate(model, data, pre, config.loss, routes)
    )
  return history


def evaluate(
  model: CommunicatingEnsemble | GatedEnsemble,
  data: Data,
  pre: dict[str, torch.Tensor],
  loss: str,
  routes: dict[str, torch.Tensor | None],
) -> dict[str, float]:
  """A full pass of joint_val, with every exchange's metrics, then of test.
  `routes` holds each exchange's routing from the previous call and is updated."""
  model.eval()
  out = {}
  for split in ("joint_val", "test"):
    x, y = getattr(data, split).x, getattr(data, split).y
    measure = split == "joint_val"
    run = partial(model, answer_each_round=True) if measure else model
    post, weights, meters = predict(run, x, measure)
    metrics = {
      "loss": loss_fn(post, y, loss, weights).item(),
      "accuracy": accuracy(post, y, weights),
      "expert_accuracy": expert_accuracy(post, y),
    }
    metrics |= prefix("answer_grid", answer_grid(post, y, weights))
    metrics |= prefix(
      "joint_answer_shift", joint_answer_shift(pre[split], post, y, weights)
    )
    if split == "joint_val":
      metrics |= prefix(
        "marginal_answer_shift", marginal_answer_shift(pre[split], post, y)
      ) | {"answer_shift_kl": answer_shift_kl(pre[split], post)}
    out |= prefix(split, metrics)
    for site, meter in meters.items():
      exchange_metrics, routes[site] = meter.result(
        pre[split], post, y, routes.get(site)
      )
      out |= {f"{key}/{site}": v for key, v in exchange_metrics.items()}
  return out


@torch.no_grad()
def predict(
  run: Callable[[torch.Tensor], Output],
  x: torch.Tensor,
  measure: bool = False,
  batch_size: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, ExchangeMeter]]:
  """Logits (N, M, classes) on x, the ensemble's weights on its experts (N, M)
  if it has any, and with `measure` a meter per exchange site that has read
  every batch's trace."""
  logits, weights, meters = [], [], {}
  for batch in x.split(batch_size):
    out = run(batch)
    logits.append(out.logits)
    if out.weights is not None:
      weights.append(out.weights)
    if measure:
      for trace in out.traces:
        meters.setdefault(trace.site, ExchangeMeter()).add(trace)
  return torch.cat(logits), torch.cat(weights) if weights else None, meters


def prefix(name: str, metrics: dict[str, float]) -> dict[str, float]:
  return {f"{name}/{key}": v for key, v in metrics.items()}
