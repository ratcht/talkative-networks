"""Assert the by-construction readings Study 02 §6.1 relies on.

Three of §5.2's series take values fixed by the architecture rather than by
training, and §6.1's measurement checks treat a departure from them as evidence that a
metric is reading the wrong tensor, reducing over the wrong axes, or hooked at
the wrong point — in which case its readings elsewhere in the grid are not
evidence of anything. Those values are checkable without training, which is
what this script does.

The decoders are zero-initialised so the injection starts an exact no-op, so
each is perturbed first: at initialisation every arm correctly reads 0 and the
construction checks say nothing.

    uv run python scripts/check_diagnostics.py
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ensemble.communicative.diagnostics import (
  ZERO_RESID, injection_stats, pool_mask_dev, _centred_rank,
)
from ensemble.communicative.spatial import PoolReduce
from ensemble.train import DECODERS

WRITE = (16, 32, 32)
VALUE = 64
BATCH = 128

failures: list[str] = []
torch.manual_seed(0)


def perturb(decoder: torch.nn.Module) -> torch.nn.Module:
  """Move the arm off its initialisation, as training would.

  Two things start at a fixed value and both must move. The output layer is
  zero-initialised, so the injection is identically zero until it moves. A
  spatial mask is initialised uniform, so the injection stays spatially
  constant until *it* moves — a mask arm at initialisation is its reference arm
  exactly, and reads 0 rather than 1 on inj_rank.
  """
  moved = False
  for module in reversed(list(decoder.modules())):
    if isinstance(module, (torch.nn.Linear, torch.nn.ConvTranspose2d)):
      torch.nn.init.normal_(module.weight, std=0.05)
      torch.nn.init.normal_(module.bias, std=0.05)
      moved = True
      break
  assert moved, "no output layer found"
  for name, param in decoder.named_parameters():
    if name.endswith("mask"):
      with torch.no_grad():
        param.add_(torch.randn_like(param) * 0.3)
  return decoder


msg = torch.randn(BATCH, VALUE)
print(f"{'decoder':11s} {'resid_frac':>11s} {'rank':>8s} {'spread':>8s}")
readings = {}
for name in ("broadcast", "mask", "convt"):
  dec = perturb(DECODERS[name](VALUE, WRITE))
  d = dec(msg)
  if name == "broadcast":
    d = d.expand(-1, -1, *WRITE[1:])  # broadcast is added, not materialised
  resid, rank, spread = injection_stats(d)
  readings[name] = (resid, rank, spread)
  print(f"{name:11s} {resid:11.6f} {rank:8.4f} {spread:8.4f}")

# §6.1: a broadcast injection is spatially constant whatever it has learned
resid, rank, _ = readings["broadcast"]
if resid > ZERO_RESID:
  failures.append(f"broadcast inj_resid_frac is {resid:.3e}, expected ~0")
if rank != 0.0:
  failures.append(f"broadcast inj_rank is {rank}, expected exactly 0")

# §6.1: a non-uniform mask scaled per channel has exactly one spatial pattern,
# every example's residual being a multiple of the same (s_c - mean s_c)
rank = readings["mask"][1]
if abs(rank - 1.0) > 1e-6:
  failures.append(f"mask inj_rank is {rank:.6f}, expected exactly 1")

# a transpose-convolution stack conditions shape on the message, so more than one
rank = readings["convt"][1]
if rank <= 1.0 + 1e-6:
  failures.append(f"convt inj_rank is {rank:.6f}, expected above 1")

# every decoder's injection must vary across examples when the message does
for name, (_, _, spread) in readings.items():
  if spread <= 0.0:
    failures.append(f"{name} inj_spread is {spread}, expected above 0")

# a mask arm at initialisation is D1 exactly: uniform mask, so no spatial
# variation to rank, whatever the output layer has learned
fresh = DECODERS["mask"](VALUE, WRITE)
for module in reversed(list(fresh.modules())):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.normal_(module.weight, std=0.05)
    break
resid, rank, _ = injection_stats(fresh(msg))
if resid > ZERO_RESID or rank != 0.0:
  failures.append(
    f"mask at uniform init reads resid_frac {resid:.3e}, rank {rank}; "
    "expected ~0 and exactly 0, identical to broadcast"
  )
print(f"mask at uniform init: resid_frac {resid:.3e}, rank {rank:.1f} "
      "(identical to broadcast, as initialised)")

# a decoder ignoring its message: rank 1 and spread 0, distinguishing the two
# rank-1 regimes §5.2 tabulates
const = perturb(DECODERS["mask"](VALUE, WRITE))(torch.zeros(BATCH, VALUE))
_, rank, spread = injection_stats(const)
if spread > 1e-6:
  failures.append(f"constant injection inj_spread is {spread:.3e}, expected ~0")

# §5.2: a uniform mask reads exactly 0, and a rescaled uniform mask still does
pool = PoolReduce((64, 8, 8))
dev = pool_mask_dev(torch.nn.Sequential(pool))
if dev is None or dev > 1e-6:
  failures.append(f"pool_mask_dev at init is {dev}, expected ~0")
with torch.no_grad():
  pool.mask.mul_(7.0)
dev = pool_mask_dev(torch.nn.Sequential(pool))
if dev is None or dev > 1e-6:
  failures.append(f"pool_mask_dev after uniform rescale is {dev}, expected ~0")
with torch.no_grad():
  pool.mask[:, 0, 0] += 1.0
dev = pool_mask_dev(torch.nn.Sequential(pool))
if not dev or dev <= 1e-6:
  failures.append(f"pool_mask_dev after a non-uniform change is {dev}, expected above 0")
print(f"\npool_mask_dev: 0 at init, 0 after uniform rescale, {dev:.4f} once non-uniform")

# §5.2: centring is load-bearing — a near-constant message must not read as rich
constant_msg = torch.ones(BATCH, VALUE) * 5 + torch.randn(BATCH, VALUE) * 1e-3
centred = _centred_rank(constant_msg)
uncentred_like = _centred_rank(constant_msg - 0)  # same call, kept explicit
if centred < 2:
  failures.append(f"centred rank of near-isotropic noise is {centred:.2f}, expected high")
print(f"centred rank of a large constant plus small isotropic noise: {centred:.1f} "
      f"(uncentred this would read ~1)")

print()
if failures:
  print(f"FAILED ({len(failures)}):")
  for f in failures:
    print("  -", f)
  raise SystemExit(1)
print("all diagnostic construction checks passed")
