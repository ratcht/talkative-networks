"""
Datasets split for the pipeline's stages.

The official training set is split per class into
  expert     trains the experts
  joint      trains the communication channel
  joint_val  picks the epochs each run is read at; never seen by any expert
The official test set is kept apart and only used for final scores.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, eq=False)
class Split(Dataset):
  x: torch.Tensor  # (N, C, H, W), normalised
  y: torch.Tensor  # (N,)

  def __len__(self) -> int:
    return len(self.y)

  def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.x[i], self.y[i]


@dataclass(frozen=True, eq=False)
class Data:
  name: str
  num_classes: int
  expert: Split
  joint: Split
  joint_val: Split
  test: Split
  digest: str  # identifies the partition; part of every cache key
  device: str = "cpu"  # where models run; the splits stay on the cpu

  @property
  def input_shape(self) -> tuple[int, ...]:
    return tuple(self.expert.x.shape[1:])


def stratified_split(
  targets: np.ndarray, fracs: tuple[float, float, float], seed: int
) -> dict[str, torch.Tensor]:
  """Partition indices so every class keeps its share in every part."""
  rng = np.random.default_rng(seed)
  parts: dict[str, list[int]] = {"expert": [], "joint": [], "joint_val": []}
  for cls in np.unique(targets):
    idx = rng.permutation(np.flatnonzero(targets == cls))
    n_expert = round(fracs[0] * len(idx))
    n_joint = round(fracs[1] * len(idx))
    parts["expert"] += idx[:n_expert].tolist()
    parts["joint"] += idx[n_expert : n_expert + n_joint].tolist()
    parts["joint_val"] += idx[n_expert + n_joint :].tolist()
  return {k: torch.as_tensor(sorted(v)) for k, v in parts.items()}


def partition_digest(name: str, parts: dict[str, torch.Tensor]) -> str:
  h = hashlib.sha256(name.encode())
  for key in ("expert", "joint", "joint_val"):
    h.update(parts[key].numpy().astype(np.int64).tobytes())
  return h.hexdigest()[:16]
