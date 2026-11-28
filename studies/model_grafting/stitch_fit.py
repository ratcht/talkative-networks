"""
Fitted stitches for model grafting: before the channel trains, fit each
obligate's stitch by least squares, so that it turns the anchor's stream at the
read point into that obligate's own stream there, as near as one affine map at
every cell can (Csiszárik et al. 2021 start stitches this way).

This is the one place the obligates see images: once, before training, on the
joint split, to give the fit its targets. Their answers are never read, and in
training and evaluation they still get only the stitched anchor stream.

  random          the stitch starts from random weights (the testbed's own start)
  fitted          the fit is the stitch's start; training moves it as usual
  fitted_frozen   the stitch stays the fit; only the return trains
"""

import torch

from testbed.data import Data
from testbed.grafting import GraftedEnsemble, PointwiseLinear, rows

FITS = ("random", "fitted", "fitted_frozen")


@torch.no_grad()
def fit_stitches(
  model: GraftedEnsemble, data: Data, freeze: bool = False, batch_size: int = 500
) -> list[float]:
  """Fit every stitch on the joint split and freeze it if asked. Returns, per
  obligate, the share of its stream at the read point the fit explains (R^2,
  every cell and channel pooled)."""
  anchor, obligates = model.experts[0], list(model.experts[1:])
  read = model.config.read
  stitches = list(model.channel.stitch)
  if not all(type(s) is PointwiseLinear for s in stitches):
    raise ValueError("fitting needs the plain stitch: --graft-stitch pointwise_linear")
  c = stitches[0].map.in_channels
  device = next(anchor.parameters()).device
  f64 = dict(dtype=torch.float64, device=device)
  xtx = torch.zeros(c + 1, c + 1, **f64)
  xty = [torch.zeros(c + 1, c, **f64) for _ in obligates]
  yy = [torch.zeros((), **f64) for _ in obligates]
  ysum = [torch.zeros(c, **f64) for _ in obligates]
  n = 0
  for x in data.joint.x.split(batch_size):
    u = rows(anchor.run(x, until=read)).reshape(-1, c).double()  # (B * P, C)
    design = torch.cat([u, torch.ones(len(u), 1, **f64)], 1)
    xtx += design.T @ design
    n += len(u)
    for j, obligate in enumerate(obligates):
      y = rows(obligate.run(x, until=read)).reshape(-1, c).double()
      xty[j] += design.T @ y
      yy[j] += (y * y).sum()
      ysum[j] += y.sum(0)
  ridge = 1e-8 * xtx.diagonal().mean() * torch.eye(c + 1, **f64)
  explained = []
  for j, stitch in enumerate(stitches):
    w = torch.linalg.solve(xtx + ridge, xty[j])  # (C + 1, C): y = [u, 1] w
    stitch.map.weight.copy_(w[:c].T.unsqueeze(-1).to(stitch.map.weight.dtype))
    stitch.map.bias.copy_(w[c].to(stitch.map.bias.dtype))
    sse = yy[j] - 2 * (w * xty[j]).sum() + (w * (xtx @ w)).sum()
    sst = yy[j] - (ysum[j] * ysum[j]).sum() / n
    explained.append(float(1 - sse / sst))
    if freeze:
      stitch.requires_grad_(False)
  return explained
