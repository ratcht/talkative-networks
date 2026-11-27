"""
Expert-generation schemes: how a pool of experts comes to differ.

A scheme turns (architecture, data, pool size, pool seed, budget) into a list
of trained experts. The contract is pool-level on purpose: negative
correlation learning trains all experts together, which a per-expert
"train from a seed" contract can't express.

Most schemes train each expert on its own and differ only in what each expert
sees. Those subclass IndependentScheme and say, per expert, which seed and
which sampler to use (a Recipe). The optimiser, epoch loop and budget are
shared, so independent schemes are directly comparable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from testbed.data import Data, Sampler, Shuffle, Weighted
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
  sampler: Sampler
  profile: dict


class IndependentScheme(Scheme):
  """Experts train one at a time, alike in everything but their recipes."""

  @abstractmethod
  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]: ...

  def train_pool(
    self, arch: str, data: Data, size: int, seed: int, budget: Budget
  ) -> list[TrainedExpert]:
    return [
      TrainedExpert(fit(arch, data, r, budget), r.profile)
      for r in self.recipes(data, size, seed)
    ]


def new_expert(arch: str, data: Data, seed: int) -> Expert:
  with seeded(derive(seed, "init")):
    model = build(arch, data.input_shape, data.num_classes)
  return model.to(data.device)


def fit(arch: str, data: Data, recipe: Recipe, budget: Budget) -> Expert:
  model = new_expert(arch, data, recipe.seed)
  optimizer = torch.optim.Adam(model.parameters(), lr=budget.lr)
  gen = generator(derive(recipe.seed, "batches"), data.device)
  model.train()
  for _ in range(budget.epochs):
    for idx in recipe.sampler.epoch(gen).split(budget.batch_size):
      loss = F.cross_entropy(model(data.expert.x[idx]), data.expert.y[idx])
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
  return model.eval()


def untrained_expert(arch: str, data: Data, pool_seed: int, i: int) -> TrainedExpert:
  """Expert i of a pool with no training at all. SeedOnly, ClassSkew,
  DisjointSubset, DataShard and NegativeCorrelation start expert i from these
  very weights."""
  model = new_expert(arch, data, derive(pool_seed, "expert", i))
  return TrainedExpert(model.eval(), {"untrained": True})


@dataclass(frozen=True)
class SeedOnly(IndependentScheme):
  """Experts differ only by seed: initialisation and batch order."""

  def config(self) -> dict:
    return {"name": "seed_only"}

  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]:
    return [
      Recipe(derive(seed, "expert", i), Shuffle(len(data.expert)), {})
      for i in range(size)
    ]


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
    y = data.expert.y
    counts = torch.bincount(y, minlength=data.num_classes).float()
    recipes = []
    for i in range(size):
      expert_seed = derive(seed, "expert", i)
      mix = np.random.default_rng(derive(expert_seed, "class_mix")).dirichlet(
        [self.alpha] * data.num_classes
      )
      weights = (
        torch.as_tensor(mix, dtype=torch.float32, device=y.device)[y] / counts[y]
      )
      recipes.append(
        Recipe(
          expert_seed, Weighted(weights, len(y)), {"class_mix": mix.round(4).tolist()}
        )
      )
    return recipes


class DisjointParts(IndependentScheme):
  """Schemes that split the expert split's examples into disjoint parts, one
  per expert of the pool, and draw each expert's batches from its own part.

  An epoch is still len(expert split) draws, so every expert gets the same
  number of steps however large its part: a part holding 3/10 of the examples
  is drawn about 3.3 times an epoch. The parts are split over the pool, from
  the pool seed, so they cover the whole split only when the group is the
  whole pool, the default."""

  # The share of an expert's draws from outside its own part. DisjointSubset
  # makes it a setting; for any other scheme here it is 0.
  epsilon: float = 0.0

  @abstractmethod
  def parts(
    self, labels: np.ndarray, num_classes: int, size: int, seed: int
  ) -> tuple[np.ndarray, list[dict]]:
    """Which expert's part each example is in, and each expert's profile."""

  def recipes(self, data: Data, size: int, seed: int) -> list[Recipe]:
    y = data.expert.y
    owner, profiles = self.parts(y.cpu().numpy(), data.num_classes, size, seed)
    owner = torch.as_tensor(owner, device=y.device)
    return [
      Recipe(derive(seed, "expert", i), own_part_sampler(owner == i, self.epsilon), p)
      for i, p in enumerate(profiles)
    ]


def own_part_sampler(own: torch.Tensor, epsilon: float) -> Weighted:
  """len(own) draws an epoch: a share 1 - epsilon uniformly from the examples
  where `own` is true, and a share epsilon uniformly from the rest."""
  inside = int(own.sum())
  outside = len(own) - inside
  if inside == 0:
    raise ValueError("an expert's own part has no examples")
  weights = torch.full(
    own.shape, epsilon / outside if outside else 0.0, device=own.device
  )
  weights[own] = (1 - epsilon) / inside
  return Weighted(weights, len(own))


@dataclass(frozen=True)
class DisjointSubset(DisjointParts):
  """Each expert of the pool gets its own classes. The pool shuffles the
  classes and splits them as evenly as they divide, the first experts taking
  the extra ones: 4, 3 and 3 of 10 classes for 3 experts. An expert's part is
  every example of its classes.

  Each expert draws from its own part, except a share `epsilon` of its draws
  from every other example, uniformly within each. At epsilon = 0 an expert
  never sees another class; at epsilon = 1 minus its part's share of the
  examples, its draws are uniform over the whole split, like SeedOnly's but
  with replacement."""

  epsilon: float = 0.0

  def __post_init__(self):
    if not 0 <= self.epsilon < 1:
      raise ValueError(
        "epsilon is the share of an expert's draws from outside its own classes, "
        f"in [0, 1); got {self.epsilon}"
      )

  def config(self) -> dict:
    return {"name": "disjoint_subset", "epsilon": float(self.epsilon)}

  def parts(
    self, labels: np.ndarray, num_classes: int, size: int, seed: int
  ) -> tuple[np.ndarray, list[dict]]:
    if size > num_classes:
      raise ValueError(
        f"{num_classes} classes can't give {size} experts classes of their own"
      )
    order = np.random.default_rng(derive(seed, "class_partition")).permutation(
      num_classes
    )
    subsets = [sorted(s.tolist()) for s in np.array_split(order, size)]
    owner_of_class = np.empty(num_classes, dtype=np.int64)
    for i, classes in enumerate(subsets):
      owner_of_class[classes] = i
    return owner_of_class[labels], [{"classes": classes} for classes in subsets]


@dataclass(frozen=True)
class DataShard(DisjointParts):
  """The control for DisjointSubset at epsilon = 0. The pool cuts every
  class's examples into the same shares, one per expert, so each shard holds
  every class. Expert i's shard is the share of the examples that expert i's
  class subset holds of the classes under DisjointSubset, 4, 3 and 3 tenths
  for 3 experts, and each expert draws only from its own shard. The two
  schemes then differ only in where the cut falls, between classes or within
  each, so DisjointSubset minus DataShard is what specialising on classes
  does, not seeing fewer examples or seeing them more often."""

  def config(self) -> dict:
    return {"name": "data_shard"}

  def parts(
    self, labels: np.ndarray, num_classes: int, size: int, seed: int
  ) -> tuple[np.ndarray, list[dict]]:
    if size > num_classes:
      raise ValueError(
        f"shards take the sizes of disjoint_subset's class subsets, and "
        f"{num_classes} classes make at most {num_classes} of them, not {size}"
      )
    counts = [len(s) for s in np.array_split(np.arange(num_classes), size)]
    cuts = np.cumsum(counts)[:-1] / num_classes
    rng = np.random.default_rng(derive(seed, "shards"))
    owner = np.empty(len(labels), dtype=np.int64)
    for c in range(num_classes):
      examples = rng.permutation(np.flatnonzero(labels == c))
      bounds = np.round(cuts * len(examples)).astype(int)
      for i, shard in enumerate(np.split(examples, bounds)):
        owner[shard] = i
    return owner, [{"shard": i} for i in range(size)]


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
      Recipe(
        derive(r.seed, "copy", c), r.sampler, {**r.profile, "recipe": i, "copy": c}
      )
      for i, r in enumerate(base)
      for c in range(self.copies)
    ][:size]


@dataclass(frozen=True)
class Untrained(Scheme):
  """Experts with random weights: nothing is learned and no data is seen, so
  the budget changes nothing."""

  def config(self) -> dict:
    return {"name": "untrained"}

  def train_pool(
    self, arch: str, data: Data, size: int, seed: int, budget: Budget
  ) -> list[TrainedExpert]:
    return [untrained_expert(arch, data, seed, i) for i in range(size)]


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
    gen = generator(derive(seed, "batches"), data.device)
    for m in models:
      m.train()
    for _ in range(budget.epochs):
      for idx in Shuffle(len(data.expert)).epoch(gen).split(budget.batch_size):
        x, y = data.expert.x[idx], data.expert.y[idx]
        logits = torch.stack([m(x) for m in models])  # (M, B, C)
        probs = logits.softmax(-1)
        spread = (probs - probs.mean(0)).pow(2).sum(-1).mean()
        loss = F.cross_entropy(logits.flatten(0, 1), y.repeat(size)) - self.lam * spread
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return [TrainedExpert(m.eval(), {"lam": self.lam}) for m in models]
