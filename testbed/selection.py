"""
Which experts of a pool form the communicating group.

With a fresh pool per trial seed, the size of the group (the default), there
is nothing to choose and the whole pool is the group. A larger pool plus a
real selector, e.g. one that picks class mixes that cover each other, slots in
here.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from testbed.experts.pool import Pool


class Selector(Protocol):
  def select(self, pool: Pool, size: int, seed: int) -> list[int]: ...


@dataclass(frozen=True)
class TakeFirst:
  def select(self, pool: Pool, size: int, seed: int) -> list[int]:
    return list(range(size))


@dataclass(frozen=True)
class RandomSubset:
  def select(self, pool: Pool, size: int, seed: int) -> list[int]:
    return sorted(
      np.random.default_rng(seed)
      .choice(len(pool.experts), size, replace=False)
      .tolist()
    )


SELECTORS = {"first": TakeFirst, "random": RandomSubset}
