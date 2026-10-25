"""
Train the channel of a communicating ensemble. The experts stay frozen.

The loss is still an open choice (docs/docket.md):
  ensemble     NLL of the ensemble's prediction, the mean expert probability (as in v2)
  per_expert   mean of each expert's own cross-entropy

Every metric in docs/testbed/metrics.md is logged at step 0, before any
training, and again after every epoch:
  train      over the epoch's batches, as the channel was at each batch
  joint_val  a full pass after the epoch, with every exchange's metrics
  test       a full pass after the epoch
Nothing here selects on test. Which epoch a run is scored at is decided
afterwards, from joint_val (docs/docket.md).
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from testbed.communication.ensemble import CommunicatingEnsemble, Output, average
from testbed.data import Data, Shuffle
from testbed.metrics import (
  ExchangeMeter,
  OptimisationMeter,
  accuracy,
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


def loss_fn(logits: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
  """logits (B, M, classes), each expert's answer."""
  if kind == "ensemble":
    return F.nll_loss(average(logits), y)
  if kind == "per_expert":
    return F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(logits.shape[1]))
  raise ValueError(f"unknown loss {kind!r}")


def train(
  model: CommunicatingEnsemble,
  data: Data,
  config: JointConfig,
  seed: int,
  tracker: Tracker,
) -> list[dict[str, float]]:
  """Train the channel. Returns the metrics logged at each step, step 0 first."""
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
  params = list(model.channels.parameters())
  if not params:  # no exchange: the ensemble is bagging, nothing to train
    return history

  optimizer = torch.optim.Adam(params, lr=config.lr)
  optimisation = OptimisationMeter(model.channels)
  gen = generator(derive(seed, "batches"), data.device)
  for _ in range(config.epochs):
    model.train()
    order = Shuffle(len(data.joint)).epoch(gen)
    logits = []
    for idx in order.split(config.batch_size):
      out = model(data.joint.x[idx])
      loss = loss_fn(out.logits, data.joint.y[idx], config.loss)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimisation.before_step()
      optimizer.step()
      optimisation.after_step()
      logits.append(out.logits.detach())
    post, y = torch.cat(logits), data.joint.y[order]
    train_metrics = prefix(
      "train",
      {"loss": loss_fn(post, y, config.loss).item(), "accuracy": accuracy(post, y)}
      | prefix("joint_answer_shift", joint_answer_shift(pre["joint"][order], post, y)),
    )
    log(
      train_metrics
      | optimisation.result()
      | evaluate(model, data, pre, config.loss, routes)
    )
  return history


def evaluate(
  model: CommunicatingEnsemble,
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
    post, meters = predict(model, x, measure=split == "joint_val")
    metrics = {
      "loss": loss_fn(post, y, loss).item(),
      "accuracy": accuracy(post, y),
      "expert_accuracy": expert_accuracy(post, y),
    } | prefix("joint_answer_shift", joint_answer_shift(pre[split], post, y))
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
) -> tuple[torch.Tensor, dict[str, ExchangeMeter]]:
  """Logits (N, M, classes) on x, and with `measure` a meter per exchange site
  that has read every batch's trace."""
  logits, meters = [], {}
  for batch in x.split(batch_size):
    out = run(batch)
    logits.append(out.logits)
    if measure:
      for trace in out.traces:
        meters.setdefault(trace.site, ExchangeMeter()).add(trace)
  return torch.cat(logits), meters


def prefix(name: str, metrics: dict[str, float]) -> dict[str, float]:
  return {f"{name}/{key}": v for key, v in metrics.items()}
