"""Spatial mappings at the two communication ports (Study 02).

The channel's adapters each sit against a spatially structured activation
tensor and have, until now, discarded that structure: the encoder averaged
`(C, H, W)` to `(C,)` and the decoder broadcast `(C,)` back over every
position. These modules are the levels of the two ladders in Study 02 §3.2 —
what each adapter is allowed to *see* at the read port and to *say* at the
write port.

Every reduce maps `(b, C, H, W) -> (b, C)` and feeds an unchanged trunk, so
the encoder levels differ only in how space is contracted. Every expand maps
`(b, C) -> (b, C, H, W)`, except `convt`, which replaces the decoder outright
because its dense stage emits a spatial grid rather than one value per
channel.

Parameter counts at the study's default ports are fixed in §7 and asserted by
scripts/check_arch_params.py — they are not free to drift, since §6.6's R7 reads E4
against E3 as a contrast between priors and is only valid while the two are
matched.
"""

from __future__ import annotations

import math

import torch as t
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

# (C, H, W) of a port's activation tensor
Shape = tuple[int, int, int]


# ---------------------------------------------------------------------------
# read port: (b, C, H, W) -> (b, C)
# ---------------------------------------------------------------------------

def gap_reduce(shape: Shape) -> nn.Module:
  """E1 — spatial mean. No spatial access: channel means only."""
  return Reduce("b c h w -> b c", "mean")


class PoolReduce(nn.Module):
  """E2 — learned per-channel mask, input-independent.

  f_c = sum_hw a_c[h,w] * A_c[h,w], with `a` initialised to 1/HW so the module
  begins numerically identical to E1 and the arm starts where its reference
  arm sits. Where it reads is fixed once trained: the same weighting applies
  to every input.
  """

  def __init__(self, shape: Shape):
    super().__init__()
    c, h, w = shape
    self.mask = nn.Parameter(t.full((c, h, w), 1.0 / (h * w)))

  def forward(self, x: t.Tensor) -> t.Tensor:
    return (x * self.mask).sum(dim=(-2, -1))


class ConvReduce(nn.Module):
  """E3 — convolutional stack, strided, into a dense contraction.

  Downsampling happens inside the stack rather than by a spatial mean, so the
  flatten reaches the trunk at `width * (H/4)^2` instead of the C*H*W a full
  flatten would present. Input-conditioned, with a learned convolutional
  prior.
  """

  def __init__(self, shape: Shape, width: int = 32):
    super().__init__()
    c, h, w = shape
    self.stack = nn.Sequential(
      nn.Conv2d(c, width, 3, stride=2, padding=1), nn.GELU(),
      nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
      nn.Flatten(),
    )
    flat = width * _after_stride2(h, 2) * _after_stride2(w, 2)
    self.out = nn.Linear(flat, c)

  def forward(self, x: t.Tensor) -> t.Tensor:
    return self.out(self.stack(x))


class FlatReduce(nn.Module):
  """E4 — fixed average pool, then dense layers. Control for E3.

  Input-conditioned like E3, and differing from it only in the prior: a fixed
  2x2 average pool asserts that the positions within a block are
  interchangeable, where a convolution learns how neighbours relate. The
  nonlinearity before the contraction is a requirement rather than a detail
  (§3.2) — without it the effective spatial weighting would not depend on the
  input and the arm would differ from E3 in conditioning as well as prior.
  """

  def __init__(self, shape: Shape, pool: int = 2, hidden: int = 32):
    super().__init__()
    c, h, w = shape
    self.net = nn.Sequential(
      nn.AvgPool2d(pool), nn.Flatten(),
      nn.Linear(c * (h // pool) * (w // pool), hidden), nn.GELU(),
      nn.Linear(hidden, c),
    )

  def forward(self, x: t.Tensor) -> t.Tensor:
    return self.net(x)


# ---------------------------------------------------------------------------
# write port: (b, C) -> (b, C, H, W)
# ---------------------------------------------------------------------------

def broadcast_expand(shape: Shape) -> nn.Module:
  """D1 — one value per channel, added at every position."""
  return Rearrange("b c -> b c 1 1")


class MaskExpand(nn.Module):
  """D2 — input-independent spatial shape, message-set amplitude.

  D_c[h,w] = g_c * s_c[h,w], an outer product of the decoded per-channel value
  with a learned mask. `s` is initialised to 1, which both starts the module
  identical to D1 and keeps the zero-init discipline intact: the product's
  factors receive gradient in proportion to one another, so zeroing *both*
  would leave both permanently dead. The decoder's last Linear is the factor
  that starts at zero.
  """

  def __init__(self, shape: Shape):
    super().__init__()
    c, h, w = shape
    self.mask = nn.Parameter(t.ones(c, h, w))

  def forward(self, x: t.Tensor) -> t.Tensor:
    return x[:, :, None, None] * self.mask


def _after_stride2(size: int, times: int) -> int:
  """Spatial extent after `times` applications of Conv2d(3, stride=2, pad=1)."""
  for _ in range(times):
    size = (size - 1) // 2 + 1
  return size


class ConvTDecoder(nn.Module):
  """D3 — dense layers into a transpose-convolution stack.

  Replaces the decoder rather than its expand: emitting (C, H, W) from a dense
  layer directly would cost roughly a million parameters at this message
  width, so the dense stage produces a small grid (C, 4, 4) and transpose
  convolutions double it to the port. Message-conditioned shape, with a
  convolutional prior.

  The final transpose convolution is zero-initialised, which is what makes the
  injection an exact no-op at step 0 — the frozen backbones must not be
  perturbed before the channel has learned anything to say.
  """

  START = 4

  def __init__(self, value_dim: int, shape: Shape):
    super().__init__()
    c, h, w = shape
    if h != w:
      raise ValueError(f"convt decoder expects a square port, got {h}x{w}")
    ups = math.log2(h / self.START)
    if ups != int(ups) or ups < 1:
      raise ValueError(
        f"port extent {h} is not {self.START} doubled a whole number of times"
      )

    self.mlp = nn.Sequential(
      nn.Linear(value_dim, value_dim), nn.GELU(),
      nn.Linear(value_dim, c * self.START * self.START),
    )
    self.to_grid = Rearrange("b (c h w) -> b c h w", c=c, h=self.START, w=self.START)

    layers: list[nn.Module] = []
    for i in range(int(ups)):
      if i:
        layers.append(nn.GELU())
      layers.append(nn.ConvTranspose2d(c, c, 4, stride=2, padding=1))
    self.expand = nn.Sequential(*layers)

    last = self.expand[-1]
    assert isinstance(last, nn.ConvTranspose2d)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

  def forward(self, x: t.Tensor) -> t.Tensor:
    return self.expand(self.to_grid(self.mlp(x)))
