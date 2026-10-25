"""
Named seed streams.

Every source of randomness gets its own seed, derived from a parent seed and a
path of names: derive(trial_seed, "pool"), derive(expert_seed, "init"). Streams
are keyed by name rather than by position, so adding a consumer never shifts
anyone else's seed, and two consumers can't land on the same stream the way
seed arithmetic does (with `seed + i`, expert 1 of one pool is expert 0 of the
next).

A trial is one full pass of the pipeline for one arm, and its trial seed is
the root of every stream in it:

  trial seed
  ├─ "pool"     pool seed
  │   └─ "expert", i   expert seed: "init", "batches", "class_mix", ...
  ├─ "group"    which experts of the pool communicate
  ├─ "channel"  the channel's initial weights
  └─ "joint"    joint training of the channel: "batches"
"""

import zlib
from contextlib import contextmanager

import numpy as np
import torch


def derive(seed: int, *path: str | int) -> int:
  """Child seed for the stream named by `path` under `seed`."""
  key = tuple(p if isinstance(p, int) else zlib.crc32(p.encode()) for p in path)
  (word,) = np.random.SeedSequence(seed, spawn_key=key).generate_state(1, np.uint64)
  return int(word) >> 1  # 63 bits: valid for torch, numpy and random alike


def generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
  return torch.Generator(device=device).manual_seed(seed)


@contextmanager
def seeded(seed: int):
  """Run a block on its own seed without disturbing the global RNG.

  Module construction draws initial weights from the global CPU generator.
  Forking it means one model's initialisation never depends on how many draws
  whatever ran before it happened to consume.
  """
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    yield
