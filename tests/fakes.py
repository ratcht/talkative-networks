"""Synthetic data and untrained experts on the CPU, shared by the tests.

The experts are small sizes of the two v4 families, registered here under
their own names: points embed, block1, block2, block3, all in one stream.
"""

import torch

from testbed.data import Data, Split
from testbed.experts.architectures import register, resnet, vit
from testbed.experts.schemes import new_expert

register("tiny_resnet", resnet, width=8, depth=3)
register("tiny_vit", vit, width=32, depth=3)


def fake_data(n: int = 200, classes: int = 10) -> Data:
  gen = torch.Generator().manual_seed(0)

  def split(k: int) -> Split:
    return Split(
      torch.randn(k, 3, 32, 32, generator=gen),
      torch.randint(0, classes, (k,), generator=gen),
    )

  return Data(
    "fake", classes, split(n), split(n), split(n // 2), split(n // 2), digest="fake"
  )


def random_experts(m: int = 3, arch: str = "tiny_resnet") -> list:
  data = fake_data()
  return [new_expert(arch, data, seed).eval() for seed in range(m)]
