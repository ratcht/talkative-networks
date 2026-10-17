"""Spatial-structure diagnostics for Study 02 (docs/studies/02-spatial-ports.md §5.2).

Six series, all taken from a single forward pass over one fixed validation
batch. They answer a question the existing metrics structurally cannot: whether
an adapter given spatial access *used* it. An arm whose injection is spatially
constant, or whose decoder produces one fixed shape and only scales it, has not
exercised the treatment its label claims, and §6.1 reads every later comparison
for that arm at its reference arm's position instead of its own.

Nothing here touches the training loop's hot path — the pass is run once per
epoch, deliberately on the same batch every epoch and in every arm so the
series are comparable along training and across the grid.
"""

from __future__ import annotations

from contextlib import ExitStack

import torch as t
import torch.nn as nn

from ensemble.communicative.spatial import PoolReduce

# Below this the injection is numerically zero and the ratios are 0/0. That is
# the state of every arm at step 0, the decoders being zero-initialised so the
# injection starts an exact no-op, so it is a routine reading and not an edge
# case. Reported as 0, which is also the right answer on its merits: an
# injection of zero magnitude has no spatially varying part.
ZERO_NORM = 1e-12

# Below this residual fraction there is no spatial structure whose rank could be
# meaningful, and the participation ratio is 0/0. Far above float32 round-off at
# D1, far below any value a trained mask produces.
ZERO_RESID = 1e-6


def _participation_ratio(x: t.Tensor) -> float:
  """Effective rank of `x`, shape (B, D), rows being observations.

  The participation ratio (sum l)^2 / sum l^2 over the eigenvalues of the Gram
  matrix `x @ x.T`, which counts directions in proportion to the magnitude they
  carry: 1 for a rank-one set, r for r equal directions, and barely above 1 for
  a dominant direction with a weak companion. No eigendecomposition is needed,
  since for a symmetric positive semidefinite Gram `sum l = tr G` and
  `sum l^2 = ||G||_F^2`.
  """
  x = x.double()
  gram = x @ x.T
  trace = t.diagonal(gram).sum()
  fro_sq = (gram * gram).sum()
  if fro_sq.item() <= 0.0:
    return 0.0
  return (trace * trace / fro_sq).item()


def _centred_rank(x: t.Tensor) -> float:
  """Participation ratio of `x` with its batch mean removed.

  The centring is load-bearing rather than cosmetic: an uncentred constant
  message dominates the spectrum and would report ~1 for a channel in perfect
  health, since every row would be nearly parallel to the mean.
  """
  return _participation_ratio(x - x.mean(dim=0, keepdim=True))


def _frac(numer: t.Tensor, denom: t.Tensor) -> float:
  """||numer|| / ||denom||, zero where the denominator has underflowed."""
  d = denom.norm().item()
  return 0.0 if d < ZERO_NORM else numer.norm().item() / d


def injection_stats(d: t.Tensor) -> tuple[float, float, float]:
  """(resid_frac, rank, spread) for one decoder's injection, shape (B, C, H, W).

  resid_frac is the share of the injection's magnitude that varies over space;
  it is a property of the injection and not of the message, D2 reaching a high
  value while carrying no spatial information at all. rank is the effective
  number of distinct spatial patterns across the batch, which is what separates
  a shape conditioned on the message from one fixed shape scaled by it. spread
  is the share varying across examples, which catches a decoder emitting a
  large but message-independent constant — something a norm alone cannot
  distinguish from a healthy injection.
  """
  d = d.detach().float()
  resid = d - d.mean(dim=(-2, -1), keepdim=True)
  resid_frac = _frac(resid, d)
  spread = _frac(d - d.mean(dim=0, keepdim=True), d)

  if resid_frac < ZERO_RESID:
    return resid_frac, 0.0, spread

  # Gram over the batch dimension, not over positions: the two share their
  # non-zero spectrum and hence their participation ratio, the batch side is
  # the smaller matrix, and it asks the question directly — how many distinct
  # spatial patterns appear across examples.
  b, c = resid.shape[0], resid.shape[1]
  per_channel = resid.reshape(b, c, -1).transpose(0, 1)  # (C, B, HW)
  rank = sum(_participation_ratio(per_channel[i]) for i in range(c)) / c
  return resid_frac, rank, spread


def pool_mask_dev(model: nn.Module) -> float | None:
  """Mean departure of the learned pooling masks from uniform, or None.

  Scale-invariant and mean-removed, so a mask rescaled uniformly still reads 0:
  what matters is whether the encoder learned to read somewhere in particular,
  not whether the weights moved. An arm still at 0 after training is its
  reference arm with spare parameters.
  """
  devs = []
  for module in model.modules():
    if isinstance(module, PoolReduce):
      a = module.mask.detach().float()
      centred = a - a.mean(dim=(-2, -1), keepdim=True)
      per_channel = [
        _frac(centred[i], a[i]) for i in range(a.shape[0])
      ]
      devs.append(sum(per_channel) / len(per_channel))
  return sum(devs) / len(devs) if devs else None


@t.no_grad()
def spatial_diagnostics(
  model: nn.Module, x: t.Tensor, device: str, k_rounds: int
) -> dict[str, float | list[float]]:
  """Run one forward pass and return §5.2's six series.

  Keys: inj_resid_frac, inj_rank, inj_spread (means over specialists),
  pool_mask_dev (absent unless the encoders carry a learned mask),
  msg_rank_value / msg_rank_aggregated (per specialist, plus a `_mean`), and
  msg_rank_ratio.
  """
  was_training = model.training
  model.eval()

  captured: dict[str, t.Tensor] = {}
  injections: dict[int, t.Tensor] = {}

  def bus_pre(_module, inputs):
    captured["V"] = inputs[2].detach()

  def bus_post(_module, _inputs, output):
    captured["c"] = output[0].detach()

  def decoder_hook(idx):
    def fn(_module, _inputs, output):
      injections[idx] = output.detach()
    return fn

  with ExitStack() as stack:
    stack.callback(lambda: model.train() if was_training else None)
    h = model.bus.register_forward_pre_hook(bus_pre)
    stack.callback(h.remove)
    h = model.bus.register_forward_hook(bus_post)
    stack.callback(h.remove)
    for i, s in enumerate(model.specialists):
      h = s.decoder.register_forward_hook(decoder_hook(i))
      stack.callback(h.remove)
    model(x.to(device), k_rounds=k_rounds)

  out: dict[str, float | list[float]] = {}

  if injections:
    stats = [injection_stats(injections[i]) for i in sorted(injections)]
    for name, values in zip(("inj_resid_frac", "inj_rank", "inj_spread"), zip(*stats)):
      out[name] = sum(values) / len(values)

  dev = pool_mask_dev(model)
  if dev is not None:
    out["pool_mask_dev"] = dev

  for key, name in (("V", "msg_rank_value"), ("c", "msg_rank_aggregated")):
    tensor = captured.get(key)
    if tensor is None:
      continue
    per_specialist = [_centred_rank(tensor[:, i]) for i in range(tensor.shape[1])]
    out[name] = per_specialist
    out[f"{name}_mean"] = sum(per_specialist) / len(per_specialist)

  value, aggregated = out.get("msg_rank_value_mean"), out.get("msg_rank_aggregated_mean")
  if value and aggregated:
    # above 1 means what arrives spans more directions than one sender emits,
    # which needs both diverse senders and aggregation that draws on several of
    # them; read alongside routing_mode_share, which settles which is missing
    out["msg_rank_ratio"] = aggregated / value

  return out
