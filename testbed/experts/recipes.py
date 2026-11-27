"""
How an expert trains: its family's usual recipe, as one Budget, and the
pieces every scheme shares to follow it.

A Budget names the optimiser, the schedule, the regularisers and the
augmentation, so the pool's cache key, which includes it, names the whole
recipe. Its defaults are v3's recipe: Adam at a constant 1e-3, no
augmentation.

  schedule  constant  the rate stays at lr
            cosine    a linear warm-up over a share `warmup` of the steps
                      (none by default), then a cosine decay to zero at the
                      last step
  optimizer sgd       SGD with Nesterov momentum 0.9, weight decay on every
                      parameter
            adamw     AdamW (0.9, 0.999); no weight decay on norms, biases or
                      the parameters an architecture lists in
                      Expert.no_weight_decay
            adam      plain Adam

Augmentation is a random crop with zero padding of side/8 (black, in the
normalised data's terms) and a horizontal flip, drawn per image on the GPU
from the expert's own augmentation stream, separate from its batch order.
Expert training runs under bfloat16 autocast when `bf16` is set; joint
training is untouched by any of this.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Budget:
  """How long and how every expert trains. The defaults are v3's recipe; an
  architecture's own is in architectures.RECIPES."""

  epochs: int = 10
  batch_size: int = 128
  optimizer: str = "adam"
  lr: float = 1e-3
  weight_decay: float = 0.0
  schedule: str = "constant"  # or "cosine"
  warmup: float = 0.0  # cosine: the share of all steps spent warming up linearly
  label_smoothing: float = 0.0
  grad_clip: float | None = None  # clip the gradient's total norm to this
  augment: bool = False  # random crop and horizontal flip
  bf16: bool = False  # bfloat16 autocast in the forward pass


def rate(step: int, total: int, budget: Budget) -> float:
  """The learning rate at `step` (from 0) of `total`."""
  if budget.schedule == "constant":
    return budget.lr
  if budget.schedule != "cosine":
    raise ValueError(f"unknown schedule {budget.schedule!r}")
  warm = int(budget.warmup * total)
  if step < warm:
    return budget.lr * (step + 1) / warm
  return (
    budget.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
  )


def optimiser(models: Iterable[nn.Module], budget: Budget) -> torch.optim.Optimizer:
  models = list(models)
  if budget.optimizer == "sgd":
    params = [p for m in models for p in m.parameters()]
    return torch.optim.SGD(
      params,
      lr=budget.lr,
      momentum=0.9,
      nesterov=True,
      weight_decay=budget.weight_decay,
    )
  if budget.optimizer == "adam":
    params = [p for m in models for p in m.parameters()]
    return torch.optim.Adam(params, lr=budget.lr, weight_decay=budget.weight_decay)
  if budget.optimizer == "adamw":
    decay, rest = [], []
    for m in models:
      skip = getattr(m, "no_weight_decay", frozenset())
      for name, p in m.named_parameters():
        exempt = p.ndim < 2 or name.rsplit(".", 1)[-1] in skip
        (rest if exempt else decay).append(p)
    return torch.optim.AdamW(
      [
        {"params": decay, "weight_decay": budget.weight_decay},
        {"params": rest, "weight_decay": 0.0},
      ],
      lr=budget.lr,
      betas=(0.9, 0.999),
    )
  raise ValueError(f"unknown optimizer {budget.optimizer!r}")


class Trainer:
  """One optimiser, its schedule and its clipping, for `total` steps."""

  def __init__(self, models: Iterable[nn.Module], budget: Budget, total: int):
    self.models = list(models)
    self.budget = budget
    self.total = total
    self.step_count = 0
    self.optimizer = optimiser(self.models, budget)

  def autocast(self, device: torch.device):
    return torch.autocast(device.type, torch.bfloat16, enabled=self.budget.bf16)

  def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
      logits.float(), y, label_smoothing=self.budget.label_smoothing
    )

  def step(self, loss: torch.Tensor) -> None:
    for group in self.optimizer.param_groups:
      group["lr"] = rate(self.step_count, self.total, self.budget)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if self.budget.grad_clip:
      nn.utils.clip_grad_norm_(
        [p for m in self.models for p in m.parameters()], self.budget.grad_clip
      )
    self.optimizer.step()
    self.step_count += 1


def steps_per_epoch(draws: int, batch_size: int) -> int:
  """Batches in an epoch of `draws` examples; the last one may be partial."""
  return -(-draws // batch_size)


def augment(x: torch.Tensor, fill: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
  """Random crop with padding of side/8 filled with `fill` (C, 1, 1), and a
  horizontal flip with probability 1/2, per image."""
  b, c, h, w = x.shape
  p = h // 8
  padded = fill.expand(b, c, h + 2 * p, w + 2 * p).clone()
  padded[:, :, p : p + h, p : p + w] = x
  dy = torch.randint(0, 2 * p + 1, (b,), device=x.device, generator=gen)
  dx = torch.randint(0, 2 * p + 1, (b,), device=x.device, generator=gen)
  rows = (dy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
  cols = (dx[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
  batch = torch.arange(b, device=x.device)[:, None, None, None]
  chans = torch.arange(c, device=x.device)[None, :, None, None]
  out = padded[batch, chans, rows, cols]
  flip = torch.rand(b, device=x.device, generator=gen) < 0.5
  return torch.where(flip[:, None, None, None], out.flip(3), out)
