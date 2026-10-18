"""Minimal exchange sweep, one fixed expert group, first 3 joint seeds.

  uv run python experiments/05_minimal_exchange/run.py                       # every arm
  uv run python experiments/05_minimal_exchange/run.py gate_p32_mlp_add      # just these

Finished arms and seeds are skipped, so rerunning resumes.
"""
import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "experiments/05_minimal_exchange"
PREFIX = "me_"

SPLIT_FILE = "splits/three_way_seed42.pt"
MANIFEST = "selection/manifests/encoder_study_group.json"
SEEDS = 3
EPOCHS = 150  # a cap, not a target: patience stops arms that have converged
PATIENCE = 10

# arm grid. proj None = no projection, else grid positions after it (64 of 8 x 8)
MIXES = ("average", "gate", "attention")
PROJS = (None, 64, 32)
DECODERS = ("linear", "mlp")
INJECTS = ("add", "replace")


def arms() -> dict[str, dict]:
  out = {}
  for mix, proj, dec, inject in product(MIXES, PROJS, DECODERS, INJECTS):
    name = f"{mix}_{'none' if proj is None else f'p{proj}'}_{dec}_{inject}"
    out[name] = dict(mix=mix, proj=proj, decoder=dec, inject=inject)
  return out


def command(name: str, mix: str, proj: int | None, decoder: str, inject: str) -> list[str]:
  return [
    sys.executable, "scripts/replicate_joint.py",
    "--split-file", SPLIT_FILE,
    "--experiment", f"{PREFIX}{name}",
    "--limit", str(SEEDS),
    "--state-file", f"{EXP_DIR}/state/{PREFIX}{name}.json",
    "--",
    "minimal",
    "--manifest", MANIFEST,
    "--mix", mix,
    *(["--proj", str(proj)] if proj is not None else []),
    "--decoder", decoder,
    "--inject", inject,
    "--epochs", str(EPOCHS),
    "--patience", str(PATIENCE),
    "--freeze-fc",
    "--output-dir", f"{EXP_DIR}/checkpoints",
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
