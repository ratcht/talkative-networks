"""Patch attention sweep, one fixed expert group, first 3 joint seeds.

  uv run python experiments/04_patch_attention/run.py              # every arm
  uv run python experiments/04_patch_attention/run.py p2_mlp_add   # just these

Finished arms and seeds are skipped, so rerunning resumes.
"""
import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "experiments/04_patch_attention"
PREFIX = "pa_"

SPLIT_FILE = "splits/three_way_seed42.pt"
MANIFEST = "selection/manifests/encoder_study_group.json"
SEEDS = 3
EPOCHS = 70
PATIENCE = 10
ENCODER = "qkv_eye"
LATE_CH = 64  # channels at the layer3.0 output

# arm grid. patch 0 has one patch, so no patch attention and no replace
PATCHES = (1, 2)
PATTN = (True, False)
DECODERS = ("linear", "mlp")
INJECTS = ("add", "replace")


def arms() -> dict[str, dict]:
  out = {}
  for dec in DECODERS:
    out[f"p0_{dec}_add"] = dict(patch=0, decoder=dec, inject="add", pattn=False)
  for patch, pattn, dec, inject in product(PATCHES, PATTN, DECODERS, INJECTS):
    name = f"p{patch}_{dec}_{inject}" + ("_pattn" if pattn else "")
    out[name] = dict(patch=patch, decoder=dec, inject=inject, pattn=pattn)
  return out


def command(name: str, patch: int, decoder: str, inject: str, pattn: bool) -> list[str]:
  # qkv_eye needs value size = size of one patch
  value_dim = LATE_CH * max(patch, 1) ** 2
  return [
    sys.executable, "scripts/replicate_joint.py",
    "--split-file", SPLIT_FILE,
    "--experiment", f"{PREFIX}{name}",
    "--limit", str(SEEDS),
    "--state-file", f"{EXP_DIR}/state/{PREFIX}{name}.json",
    "--",
    "communicative",
    "--manifest", MANIFEST,
    "--encoder", ENCODER,
    "--value-dim", str(value_dim),
    "--epochs", str(EPOCHS),
    "--patience", str(PATIENCE),
    "--freeze-fc",
    "--output-dir", f"{EXP_DIR}/checkpoints",
    "--patch", str(patch),
    "--decoder", decoder,
    "--inject", inject,
    *(["--patch-attn"] if pattn else []),
  ]


def main():
  all_arms = arms()
  p = argparse.ArgumentParser()
  p.add_argument("arms", nargs="*", help=f"default: all {len(all_arms)}")
  names = p.parse_args().arms or list(all_arms)
  if unknown := set(names) - set(all_arms):
    p.error(f"unknown arms {sorted(unknown)}, choose from {list(all_arms)}")

  for i, name in enumerate(names, 1):
    print(f"[{i}/{len(names)}] {name}", flush=True)
    subprocess.run(command(name, **all_arms[name]), cwd=REPO, check=True)


if __name__ == "__main__":
  main()
