"""Assert Study 02's architectures against the numbers frozen in its protocol.

docs/studies/02-spatial-ports.md §7 fixes a parameter count for every level of
both ladders, and §6.6's R7 is only a contrast between priors while E3 and E4
remain matched. Those numbers are protocol rather than implementation detail,
so drift in either direction is a deviation and not a refactor.

This also checks the two initialisation guarantees §3.2 relies on: that the
injection is an exact no-op at step 0 under every decoder, and that E2 and D2
begin numerically identical to the reference levels they extend.

    uv run python scripts/check_arch_params.py
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ensemble.communicative.bus import MLPEncoder
from ensemble.communicative.spatial import (
  ConvReduce, ConvTDecoder, FlatReduce, MaskExpand, PoolReduce,
  broadcast_expand, gap_reduce,
)
from ensemble.train import DECODERS, ENCODERS

READ: tuple[int, int, int] = (64, 8, 8)    # output of layer3.0
WRITE: tuple[int, int, int] = (16, 32, 32)  # input to layer1.2
KEY, VALUE = 16, 64

# §7.1 and §7.2, parameters per specialist
EXPECTED_REDUCE = {"gap": 0, "pool": 4_096, "conv": 35_968, "flat": 34_912}
EXPECTED_DECODER = {"broadcast": 5_200, "mask": 21_584, "convt": 33_136}
TRUNK_AND_HEADS = 10_528  # §7 preamble

failures: list[str] = []


def check(label: str, got, want) -> None:
  if got != want:
    failures.append(f"{label}: got {got}, expected {want}")


def n_params(module: torch.nn.Module) -> int:
  return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --- the spatial mappings in isolation -------------------------------------
for name, factory in (("gap", gap_reduce), ("pool", PoolReduce),
                      ("conv", ConvReduce), ("flat", FlatReduce)):
  mod = factory(READ)
  out = mod(torch.zeros(2, *READ))
  check(f"reduce {name!r} output shape", tuple(out.shape[1:]), (READ[0],))
  check(f"reduce {name!r} parameters", n_params(mod), EXPECTED_REDUCE[name])

# --- whole encoders, trunk included ---------------------------------------
for name in ("gap", "pool", "conv", "flat"):
  cls, factory = ENCODERS[name]
  enc = cls(READ[0], KEY, VALUE, factory(READ))
  q, k, v = enc(torch.zeros(2, *READ))
  check(f"encoder {name!r} q/k/v widths",
        (q.shape[-1], k.shape[-1], v.shape[-1]), (KEY, KEY, VALUE))
  check(f"encoder {name!r} parameters",
        n_params(enc), TRUNK_AND_HEADS + EXPECTED_REDUCE[name])

# --- whole decoders --------------------------------------------------------
for name in ("broadcast", "mask", "convt"):
  dec = DECODERS[name](VALUE, WRITE)
  out = dec(torch.zeros(2, VALUE))
  # D1 emits (C, 1, 1) and is broadcast on addition; the others emit the port
  want = (WRITE[0], 1, 1) if name == "broadcast" else WRITE
  check(f"decoder {name!r} output shape", tuple(out.shape[1:]), want)
  check(f"decoder {name!r} parameters", n_params(dec), EXPECTED_DECODER[name])

# --- §3.2: the injection is an exact no-op at step 0 ----------------------
for name in ("broadcast", "mask", "convt"):
  dec = DECODERS[name](VALUE, WRITE)
  out = dec(torch.randn(8, VALUE))
  if out.abs().max().item() != 0.0:
    failures.append(
      f"decoder {name!r} injects {out.abs().max().item():.3e} at init, expected exactly 0"
    )

# --- §3.2: E2 and D2 start identical to the levels they extend -------------
x = torch.randn(8, *READ)
if not torch.allclose(PoolReduce(READ)(x), gap_reduce(READ)(x), atol=1e-6):
  failures.append("PoolReduce at init does not match gap_reduce")

g = torch.randn(8, WRITE[0])
mask_out = MaskExpand(WRITE)(g)
broadcast_out = broadcast_expand(WRITE)(g)
if not torch.allclose(mask_out, broadcast_out.expand_as(mask_out), atol=1e-6):
  failures.append("MaskExpand at init does not match broadcast_expand")

# --- §7.3: per-arm budget --------------------------------------------------
print("per-arm trainable parameters (3 specialists, classification heads frozen):\n")
base = None
for enc in ("gap", "pool", "conv", "flat"):
  for dec in ("broadcast", "mask", "convt"):
    if enc == "flat" and dec != "broadcast":
      continue  # §3.3: E4 is crossed with D1 alone
    per = TRUNK_AND_HEADS + EXPECTED_REDUCE[enc] + EXPECTED_DECODER[dec]
    total = 3 * per
    base = base or total
    print(f"  {enc:5s} x {dec:9s} {per:>8,} per specialist  {total:>9,} per arm"
          f"  {total / base:>5.2f}x")

# --- the convt stack rejects ports it cannot reach by doubling from 4 ------
for bad in ((16, 12, 12), (16, 32, 16)):
  try:
    ConvTDecoder(VALUE, bad)
  except ValueError:
    pass
  else:
    failures.append(f"ConvTDecoder accepted an unreachable port shape {bad}")

print()
if failures:
  print(f"FAILED ({len(failures)}):")
  for f in failures:
    print("  -", f)
  raise SystemExit(1)
print("all architecture checks passed")
