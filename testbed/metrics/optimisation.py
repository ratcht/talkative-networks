"""
Training dynamics: whether each part of the channel gets gradient, and whether
its weights move. Definitions: docs/testbed/metrics.md.
"""

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


def component(name: str) -> str:
  """Where a parameter of an ensemble's channels sits, as a key suffix:
  '0.encoders.2.query.weight' -> 'encoder.query/ex0'. Read off the module
  tree, so a new part gets its own series without a list to update."""
  exchange, group, *rest = name.split(".")
  if group in ("encoders", "decoders"):
    group = f"{group.removesuffix('s')}.{rest[1]}"  # rest[0] is the expert; pooled over experts
  return f"{group}/ex{exchange}"


class OptimisationMeter:
  """Gradient norm and update ratio per component, averaged over the steps
  since the last result(). Call before_step() between backward() and the
  optimiser's step(), and after_step() right after it.

  The channel's weights are handled as one flat vector, with the series each
  weight belongs to alongside, so a step costs a few tensor operations however
  many parameters the channel has, and nothing waits on the GPU."""

  def __init__(self, channels: nn.ModuleList):
    names, self.params = zip(*channels.named_parameters())
    components = [component(name) for name in names]
    self.names = ["total", *dict.fromkeys(components)]
    sizes = torch.tensor([p.numel() for p in self.params])
    series = torch.tensor([self.names.index(c) for c in components])
    self.device = self.params[0].device
    self.series = series.repeat_interleave(sizes).to(self.device)  # one per weight
    self.reset()

  def reset(self) -> None:
    self.grad_norm = torch.zeros(
      len(self.names), dtype=torch.float64, device=self.device
    )
    self.update_ratio = torch.zeros_like(self.grad_norm)
    self.steps = 0

  def _norms(self, flat: torch.Tensor) -> torch.Tensor:
    """flat: one value per weight -> its norm over each series, total first."""
    squares = flat.double().pow(2)
    sums = torch.zeros_like(self.grad_norm).index_add_(0, self.series, squares)
    sums[0] = squares.sum()
    return sums.sqrt()

  @torch.no_grad()
  def before_step(self) -> None:
    grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in self.params]
    self.grad_norm += self._norms(parameters_to_vector(grads))
    self.before = parameters_to_vector(self.params)

  @torch.no_grad()
  def after_step(self) -> None:
    moved = self._norms(parameters_to_vector(self.params) - self.before)
    self.update_ratio += moved / self._norms(self.before).clamp_min(1e-12)
    self.steps += 1

  def result(self) -> dict[str, float]:
    out = {}
    for name, g, u in zip(
      self.names,
      (self.grad_norm / self.steps).tolist(),
      (self.update_ratio / self.steps).tolist(),
    ):
      out[f"grad_norm/{name}"] = g
      out[f"update_ratio/{name}"] = u
    self.reset()
    return out
