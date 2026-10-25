"""Synthetic data and untrained experts on the CPU, shared by the tests."""

import torch

from testbed.data import Data, Split
from testbed.experts.schemes import new_expert


def fake_data(n: int = 200, classes: int = 10) -> Data:
  gen = torch.Generator().manual_seed(0)

  def split(k: int) -> Split:
    return Split(
      torch.randn(k, 1, 28, 28, generator=gen),
      torch.randint(0, classes, (k,), generator=gen),
    )

  return Data(
    "fake", classes, split(n), split(n), split(n // 2), split(n // 2), digest="fake"
  )


def random_experts(m: int = 3, arch: str = "cnn3") -> list:
  data = fake_data()
  return [new_expert(arch, data, seed).eval() for seed in range(m)]
