"""
Datasets held on the GPU as whole tensors, split for the pipeline's stages.

Small image datasets fit in GPU memory, so there is no DataLoader: an epoch is
a tensor of example indices drawn by a Sampler with an explicit generator.
Every draw is tied to a named seed, and an expert's epoch costs a fraction of
a second.

The official training set is split per class into
  expert     trains the experts
  joint      trains the communication channel
  joint_val  picks the epochs each run is read at; never seen by any expert
The official test set is kept apart and only used for final scores.

Images are scaled to [0, 1] and normalised per channel by the expert split's
mean and standard deviation. `fill` is a black pixel in those terms, which
the experts' augmentation pads with.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
  "cifar100": datasets.CIFAR100,
  "cifar10": datasets.CIFAR10,
  "fashion": datasets.FashionMNIST,
  "mnist": datasets.MNIST,
}


@dataclass(frozen=True, eq=False)
class Split:
  x: torch.Tensor  # (N, C, H, W), normalised
  y: torch.Tensor  # (N,)

  def __len__(self) -> int:
    return len(self.y)


@dataclass(frozen=True, eq=False)
class Data:
  name: str
  num_classes: int
  expert: Split
  joint: Split
  joint_val: Split
  test: Split
  digest: str  # identifies the partition; part of every cache key
  fill: torch.Tensor | None = None  # (C, 1, 1): a black pixel, normalised

  def __post_init__(self):
    if self.fill is None:
      object.__setattr__(
        self, "fill", torch.zeros(self.input_shape[0], 1, 1, device=self.device)
      )

  @property
  def input_shape(self) -> tuple[int, ...]:
    return tuple(self.expert.x.shape[1:])

  @property
  def device(self) -> torch.device:
    return self.expert.x.device


def load(
  name: str = "cifar100",
  fracs: tuple[float, float, float] = (0.5, 0.4, 0.1),
  split_seed: int = 0,
  root: Path = REPO_ROOT / "data",
  device: str = "cuda",
) -> Data:
  train = DATASETS[name](root, train=True, download=True)
  test = DATASETS[name](root, train=False, download=True)
  targets = torch.as_tensor(train.targets)
  parts = stratified_split(targets.numpy(), fracs, split_seed)

  images = scaled(train.data)
  reference = images[parts["expert"]]
  mean = reference.mean((0, 2, 3), keepdim=True)[0]  # (C, 1, 1)
  std = reference.std((0, 2, 3), keepdim=True)[0]

  def split(x: torch.Tensor, y: torch.Tensor) -> Split:
    return Split(((x - mean) / std).to(device), y.to(device))

  return Data(
    name=name,
    num_classes=len(train.classes),
    **{k: split(images[idx], targets[idx]) for k, idx in parts.items()},
    test=split(scaled(test.data), torch.as_tensor(test.targets)),
    digest=partition_digest(name, parts),
    fill=((0 - mean) / std).to(device),
  )


def scaled(images) -> torch.Tensor:
  """A dataset's images as (N, C, H, W) floats in [0, 1]: torchvision keeps
  MNIST's as (N, H, W) tensors and CIFAR's as (N, H, W, C) arrays."""
  x = torch.as_tensor(images)
  x = x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
  return x.float().div(255).contiguous()


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


class Sampler(Protocol):
  """Draws the example indices for one epoch; its length is how many."""

  def epoch(self, gen: torch.Generator) -> torch.Tensor: ...

  def __len__(self) -> int: ...


@dataclass(frozen=True)
class Shuffle:
  """Every example once per epoch, in random order."""

  size: int

  def epoch(self, gen: torch.Generator) -> torch.Tensor:
    return torch.randperm(self.size, generator=gen, device=gen.device)

  def __len__(self) -> int:
    return self.size


@dataclass(frozen=True, eq=False)
class Weighted:
  """`draws` examples per epoch, with replacement, in proportion to `weights`."""

  weights: torch.Tensor  # one non-negative weight per example
  draws: int

  def epoch(self, gen: torch.Generator) -> torch.Tensor:
    return torch.multinomial(self.weights, self.draws, replacement=True, generator=gen)

  def __len__(self) -> int:
    return self.draws
