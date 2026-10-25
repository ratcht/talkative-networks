"""
Expert architectures, written as named stages.

The communication channel reads and writes at the boundaries between stages,
so an expert is an ordered list of stages rather than one opaque forward. A
point is named after the stage whose output it is: reading at "stage2" means
reading what stage2 produced. The last stage is the head; its output is the
expert's answer, not a point.

Stages hand over their outputs raw. The head owns its flatten, so the point
after the last conv stage is a full (B, C, H, W) map, and whether to pool is
up to the encoder that reads it.

Any model joins by being cut into stages: wrap the pieces of an nn.Sequential
(or the blocks of a pretrained network) in an Expert.
"""

import torch
import torch.nn as nn


class Expert(nn.Module):
  def __init__(self, stages: dict[str, nn.Module]):
    super().__init__()
    self.stages = nn.ModuleDict(stages)
    self._order = {name: i for i, name in enumerate(self.stages)}

  @property
  def points(self) -> list[str]:
    """Where a channel can read or write: after every stage but the head."""
    return list(self.stages)[:-1]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.run(x)

  def run(
    self, x: torch.Tensor, after: str | None = None, until: str | None = None
  ) -> torch.Tensor:
    """Run the stages after point `after` up to and including stage `until`.

    after=None starts from the input; until=None runs through the head.
    """
    names = list(self.stages)
    start = 0 if after is None else self._order[after] + 1
    stop = len(names) if until is None else self._order[until] + 1
    for name in names[start:stop]:
      x = self.stages[name](x)
    return x

  @torch.no_grad()
  def point_shapes(self, input_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    """Feature shape (without batch) at every point, from one dry forward."""
    x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)
    shapes = {}
    for name, stage in self.stages.items():
      x = stage(x)
      shapes[name] = tuple(x.shape[1:])
    return {p: shapes[p] for p in self.points}


def cnn3(input_shape: tuple[int, ...], num_classes: int) -> Expert:
  """Three conv stages and an MLP head; about 98k parameters on 28x28 inputs."""
  channels, size = input_shape[0], input_shape[1]

  def stage(c_in: int, c_out: int) -> nn.Module:
    return nn.Sequential(
      nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
    )

  return Expert(
    {
      "stage1": stage(channels, 16),
      "stage2": stage(16, 32),
      "stage3": stage(32, 64),
      "head": nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * (size // 8) ** 2, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
      ),
    }
  )


def lenet5(input_shape: tuple[int, ...], num_classes: int) -> Expert:
  """Classic LeNet-5 (2 conv + 3 FC); about 62k parameters on 28x28 inputs."""
  channels, size = input_shape[0], input_shape[1]
  return Expert(
    {
      "stage1": nn.Sequential(
        nn.Conv2d(channels, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)
      ),
      "stage2": nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2)),
      "stage3": nn.Sequential(
        nn.Flatten(), nn.Linear(16 * ((size // 2 - 4) // 2) ** 2, 120), nn.ReLU()
      ),
      "head": nn.Sequential(nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes)),
    }
  )


ARCHITECTURES = {"cnn3": cnn3, "lenet5": lenet5}


def build(arch: str, input_shape: tuple[int, ...], num_classes: int) -> Expert:
  return ARCHITECTURES[arch](input_shape, num_classes)
