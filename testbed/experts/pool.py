"""
Expert pools on disk, keyed by content.

A pool is identified by everything that determines its experts: the scheme's
config, the architecture, the budget, the data partition, the pool size and
the pool seed. Their hash is the file name, so changing alpha, the epoch count
or the split can't silently serve stale experts. A plain directory path can't
tell those apart (the lesson of v2's selection/cache.py, on main).

The key names the recipe, not the weights: retraining never reproduces the
weights exactly, on another GPU or even the same one. That is why a pool is
trained once and loaded from then on, and why a saved pool is never replaced:
  - runs on one machine (one per GPU, say) take a lock per key, so a second
    run asking for a pool still being trained waits, then loads it;
  - every pool carries a digest of its weights, which runs log, so the results
    can check that paired runs used identical experts, wherever they ran.
To share pools between machines, copy without replacing existing files
(rsync --ignore-existing).
"""

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from testbed.data import REPO_ROOT, Data
from testbed.experts.architectures import build
from testbed.experts.schemes import Budget, Scheme, TrainedExpert

CACHE = REPO_ROOT / "cache" / "experts"


@dataclass
class Pool:
  key: str  # the recipe: a hash of everything that determines the experts
  digest: str  # the weights: identical experts give an identical digest
  spec: dict
  experts: list[TrainedExpert]


def get_pool(
  scheme: Scheme,
  arch: str,
  data: Data,
  size: int,
  seed: int,
  budget: Budget,
  root: Path = CACHE,
) -> Pool:
  spec = {
    "scheme": scheme.config(),
    "arch": arch,
    "budget": asdict(budget),
    "data": data.digest,
    "size": size,
    "seed": seed,
  }
  key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
  path = root / f"{key}.pt"
  root.mkdir(parents=True, exist_ok=True)
  with _lock(root / f"{key}.lock"):
    if path.exists():
      experts = _load(path, arch, data)
    else:
      experts = scheme.train_pool(arch, data, size, seed, budget)
      _save(path, spec, experts)
  return Pool(key, weights_digest(experts), spec, experts)


def weights_digest(experts: list[TrainedExpert]) -> str:
  """A hash of every expert's weights, in order. It depends only on the
  weights, not on how they were saved or copied."""
  h = hashlib.sha256()
  for expert in experts:
    for name, tensor in expert.model.state_dict().items():
      h.update(name.encode())
      h.update(tensor.detach().cpu().reshape(-1).numpy().tobytes())
  return h.hexdigest()[:16]


@contextmanager
def _lock(path: Path) -> Iterator[None]:
  """Hold an exclusive lock on `path`, across processes and threads."""
  with open(path, "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(f, fcntl.LOCK_UN)


def _save(path: Path, spec: dict, experts: list[TrainedExpert]) -> None:
  """Write atomically: a killed process must not leave a file that exists
  (so the cache would trust it) but won't load. The temporary name is unique
  to the process, so no two writers ever share one."""
  tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
  torch.save(
    {
      "spec": spec,
      "experts": [
        {"state_dict": e.model.state_dict(), "profile": e.profile} for e in experts
      ],
    },
    tmp,
  )
  os.replace(tmp, path)


def _load(path: Path, arch: str, data: Data) -> list[TrainedExpert]:
  blob = torch.load(path, map_location=data.device, weights_only=True)
  experts = []
  for entry in blob["experts"]:
    model = build(arch, data.input_shape, data.num_classes).to(data.device)
    model.load_state_dict(entry["state_dict"])
    experts.append(TrainedExpert(model.eval(), entry["profile"]))
  return experts
