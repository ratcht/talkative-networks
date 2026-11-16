"""
Expert-generation schemes: how a pool of experts comes to differ.

A scheme turns (architecture, data, pool size, pool seed, budget) into a list
of trained experts. The contract is pool-level on purpose: negative
correlation learning trains all experts together, which a per-expert
"train from a seed" contract can't express.

Most schemes train each expert on its own and differ only in what each expert
sees. Those subclass IndependentScheme and say, per expert, which seed to use
(a Recipe) and what to sample (sampler). The optimiser, epoch loop and budget
are shared, so independent schemes are directly comparable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Sampler, WeightedRandomSampler

from testbed.data import Data
from testbed.experts.architectures import Expert, build
from testbed.seeds import derive, generator, seeded


@dataclass(frozen=True)
class Budget:
  """How long and how hard every expert trains. Short on purpose: we want
  sub-optimal experts (docs/docket.md)."""

  epochs: int = 10
  batch_size: int = 128
  lr: float = 1e-3


@dataclass
class TrainedExpert:
  model: Expert
  profile: dict  # what made this expert different, e.g. its class mix


class Scheme(ABC):
  @abstractmethod
  def config(self) -> dict:
    """Everything that changes the experts; it goes into the pool's cache key."""

  @abstractmethod
  def train_pool(
    self, arch: str, data: Data, size: int, seed: int, budget: Budget
  ) -> list[TrainedExpert]: ...


@dataclass(frozen=True, eq=False)
class Recipe:
  """What one independently trained expert gets."""

  seed: int
  profile: dict


class IndependentScheme(Scheme):
  """Experts train one at a time, alike in everything but their recipes."""

  @abstractmethod
  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]: ...

  @abstractmethod
  def sampler(self, data: Data, profile: dict, gen: torch.Generator) -> Sampler: ...

  def train_pool(
    self, arch: str, data: Data, size: int, seed: int, budget: Budget
  ) -> list[TrainedExpert]:
    experts = []
    for r in self.recipes(data, size, seed):
      sampler = self.sampler(data, r.profile, generator(derive(r.seed, "batches")))
      experts.append(TrainedExpert(fit(arch, data, r.seed, sampler, budget), r.profile))
    return experts


def new_expert(arch: str, data: Data, seed: int) -> Expert:
  with seeded(derive(seed, "init")):
    model = build(arch, data.input_shape, data.num_classes)
  return model.to(data.device)


def fit(arch: str, data: Data, seed: int, sampler: Sampler, budget: Budget) -> Expert:
  model = new_expert(arch, data, seed)
  optimizer = torch.optim.Adam(model.parameters(), lr=budget.lr)
  loader = DataLoader(data.expert, budget.batch_size, sampler=sampler)
  model.train()
  for _ in range(budget.epochs):
    for x, y in loader:
      x, y = x.to(data.device), y.to(data.device)
      loss = F.cross_entropy(model(x), y)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
  return model.eval()


@dataclass(frozen=True)
class SeedOnly(IndependentScheme):
  """Experts differ only by seed: initialisation and batch order."""

  def config(self) -> dict:
    return {"name": "seed_only"}

  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]:
    return [Recipe(derive(seed, "expert", i), {}) for i in range(size)]

  def sampler(self, data: Data, profile: dict, gen: torch.Generator) -> Sampler:
    return RandomSampler(data.expert, generator=gen)


@dataclass(frozen=True)
class ClassSkew(IndependentScheme):
  """Each expert sees the classes in its own proportions, rolled from a
  Dirichlet(alpha). Large alpha ≈ equal mix (SeedOnly), ~1 lopsided, ~0.1
  near-specialists. An epoch is still len(expert split) draws, so every expert
  gets the same number of steps whatever its mix."""

  alpha: float = 1.0

  def config(self) -> dict:
    return {"name": "class_skew", "alpha": self.alpha}

  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]:
    recipes = []
    for i in range(size):
      expert_seed = derive(seed, "expert", i)
      mix = np.random.default_rng(derive(expert_seed, "class_mix")).dirichlet(
        [self.alpha] * data.num_classes
      )
      recipes.append(Recipe(expert_seed, {"class_mix": mix.tolist()}))
    return recipes

  def sampler(self, data: Data, profile: dict, gen: torch.Generator) -> Sampler:
    y = data.expert.y
    counts = torch.bincount(y, minlength=data.num_classes).float()
    weights = torch.tensor(profile["class_mix"])[y] / counts[y]
    return WeightedRandomSampler(weights, len(y), generator=gen)


@dataclass(frozen=True)
class Redundant(IndependentScheme):
  """Experts in groups of `copies` that share a recipe and differ only by seed,
  e.g. two experts with the same class mix. For testing whether communication
  copes with experts that know the same things."""

  inner: IndependentScheme
  copies: int = 2

  def config(self) -> dict:
    return {"name": "redundant", "copies": self.copies, "inner": self.inner.config()}

  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]:
    base = self.inner.recipes(data, -(-size // self.copies), seed)
    return [
      Recipe(derive(r.seed, "copy", c), {**r.profile, "recipe": i, "copy": c})
      for i, r in enumerate(base)
      for c in range(self.copies)
    ][:size]

  def sampler(self, data: Data, profile: dict, gen: torch.Generator) -> Sampler:
    return self.inner.sampler(data, profile, gen)


@dataclass(frozen=True)
class NegativeCorrelation(Scheme):
  """Negative correlation learning (Liu & Yao 1999). All experts train together
  on the same batches, each penalised for agreeing with the ensemble mean:

    loss = mean_i CE_i  -  lam * mean_i ||p_i - p_mean||^2

  with p the experts' probabilities. lam = 0 is independent training on shared
  batches, from the same initialisations SeedOnly would use.
  """

  lam: float = 0.5

  def config(self) -> dict:
    return {"name": "negative_correlation", "lam": self.lam}

  def train_pool(
    self, arch: str, data: Data, size: int, seed: int, budget: Budget
  ) -> list[TrainedExpert]:
    models = [new_expert(arch, data, derive(seed, "expert", i)) for i in range(size)]
    optimizer = torch.optim.Adam(
      [p for m in models for p in m.parameters()], lr=budget.lr
    )
    gen = generator(derive(seed, "batches"))
    loader = DataLoader(data.expert, budget.batch_size, shuffle=True, generator=gen)
    for m in models:
      m.train()
    for _ in range(budget.epochs):
      for x, y in loader:
        x, y = x.to(data.device), y.to(data.device)
        logits = torch.stack([m(x) for m in models])  # (M, B, C)
        probs = logits.softmax(-1)
        spread = (probs - probs.mean(0)).pow(2).sum(-1).mean()
        loss = F.cross_entropy(logits.flatten(0, 1), y.repeat(size)) - self.lam * spread
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return [TrainedExpert(m.eval(), {"lam": self.lam}) for m in models]
