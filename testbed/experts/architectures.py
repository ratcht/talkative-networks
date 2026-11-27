"""
Expert architectures, written as named stages around one residual stream.

The communication channel reads and writes at the boundaries between stages,
so an expert is an ordered list of stages rather than one opaque forward. A
point is named after the stage whose output it is: reading at "block3" means
reading what block3 produced. The last stage is the head; its output is the
expert's answer, not a point.

Every v4 expert has a uniform residual stream: an embedding stage starts the
stream, and every block adds its output to it, so the stream has one shape at
every point, from "embed" to the last block. The channel reads and writes
only there, in the stream (docs/testbed/experts.md). The stream is handed
over channels first, (B, C, *positions), in both families:
  resnet  (B, width, grid, grid): one row per grid cell
  vit     (B, width, 1 + patches): the class token, then one row per patch
so a part that reads positions treats a token like a grid cell.

Each family comes with the recipe it is usually trained with (RECIPES).
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.experts.recipes import Budget


class Expert(nn.Module):
  # Parameters an AdamW recipe leaves out of weight decay, by their last name,
  # besides every parameter with fewer than two dimensions (norms, biases).
  no_weight_decay: frozenset[str] = frozenset()

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
    was_training = self.training
    self.eval()  # a batch of one can't train batch norm
    x = torch.zeros(1, *input_shape, device=next(self.parameters()).device)
    shapes = {}
    for name, stage in self.stages.items():
      x = stage(x)
      shapes[name] = tuple(x.shape[1:])
    self.train(was_training)
    return {p: shapes[p] for p in self.points}

  def stream_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """The residual stream's shape (without batch). Raises unless every point
    has it, i.e. unless every point is the residual stream."""
    shapes = self.point_shapes(input_shape)
    if len(set(shapes.values())) != 1:
      raise ValueError(
        f"the experts' points don't share one stream shape: {shapes}. A channel "
        "reads and writes only in a uniform residual stream."
      )
    return next(iter(shapes.values()))


# ---------------------------------------------------------------- ResNet


class PreActBlock(nn.Module):
  """x + conv(relu(bn(conv(relu(bn(x)))))): the pre-activation residual block
  (He et al. 2016). Nothing but the block's output is added to the stream."""

  def __init__(self, width: int):
    super().__init__()
    self.bn1 = nn.BatchNorm2d(width)
    self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(width)
    self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.conv2(F.relu(self.bn2(self.conv1(F.relu(self.bn1(x))))))


class ResNetHead(nn.Module):
  """BN, ReLU, the mean over the grid, a linear classifier."""

  def __init__(self, width: int, num_classes: int):
    super().__init__()
    self.bn = nn.BatchNorm2d(width)
    self.fc = nn.Linear(width, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(F.relu(self.bn(x)).mean((2, 3)))


def resnet(
  input_shape: tuple[int, ...],
  num_classes: int,
  width: int = 64,
  depth: int = 12,
  grid: int = 16,
) -> Expert:
  """Isotropic pre-activation ResNet: a 3x3 convolution with stride side/grid
  starts a width x grid x grid stream, then `depth` blocks of that width, with
  none of a standard ResNet's downsampling stages, which would change the
  stream's shape. 12x64 on 32x32 CIFAR-100 has 896,164 parameters."""
  channels, side = input_shape[0], input_shape[1]
  if side % grid:
    raise ValueError(f"a {side}x{side} image can't be cut into a {grid}x{grid} grid")
  stages: dict[str, nn.Module] = {
    "embed": nn.Conv2d(channels, width, 3, stride=side // grid, padding=1, bias=False)
  }
  stages |= {f"block{i}": PreActBlock(width) for i in range(1, depth + 1)}
  stages["head"] = ResNetHead(width, num_classes)
  model = Expert(stages)
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
  return model


# ---------------------------------------------------------------- ViT


class PatchEmbedding(nn.Module):
  """Patches embedded linearly, a class token in front, learned positions
  added: the start of the stream, (B, width, 1 + patches)."""

  def __init__(self, channels: int, width: int, side: int, patch: int):
    super().__init__()
    self.proj = nn.Conv2d(channels, width, patch, stride=patch)
    self.cls = nn.Parameter(torch.zeros(1, width, 1))
    self.pos = nn.Parameter(
      nn.init.trunc_normal_(torch.empty(1, width, 1 + (side // patch) ** 2), std=0.02)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    patches = self.proj(x).flatten(2)  # (B, width, patches)
    return torch.cat([self.cls.expand(len(x), -1, -1), patches], dim=2) + self.pos


class Attention(nn.Module):
  def __init__(self, width: int, heads: int):
    super().__init__()
    self.heads = heads
    self.qkv = nn.Linear(width, 3 * width)
    self.proj = nn.Linear(width, width)

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    b, n, d = t.shape
    q, k, v = (
      self.qkv(t).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
    )
    out = F.scaled_dot_product_attention(q, k, v)
    return self.proj(out.transpose(1, 2).reshape(b, n, d))


class TransformerBlock(nn.Module):
  """x + attention(LN(x)), then x + MLP(LN(x)), the pre-norm block of the
  standard ViT (Dosovitskiy et al. 2021); MLP 4x wide with a GELU. It takes
  and returns the stream channels first; inside, a row per token."""

  def __init__(self, width: int, heads: int):
    super().__init__()
    self.norm1 = nn.LayerNorm(width)
    self.attn = Attention(width, heads)
    self.norm2 = nn.LayerNorm(width)
    self.mlp = nn.Sequential(
      nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    t = x.transpose(1, 2)  # (B, tokens, width)
    t = t + self.attn(self.norm1(t))
    t = t + self.mlp(self.norm2(t))
    return t.transpose(1, 2)


class ViTHead(nn.Module):
  """Layer norm on the class token, then a linear classifier."""

  def __init__(self, width: int, num_classes: int):
    super().__init__()
    self.norm = nn.LayerNorm(width)
    self.fc = nn.Linear(width, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(self.norm(x[:, :, 0]))


def vit(
  input_shape: tuple[int, ...],
  num_classes: int,
  width: int = 96,
  depth: int = 12,
  head_width: int = 32,
  patch: int | None = None,
) -> Expert:
  """Standard ViT: patches of side/8 (64 of them), a class token, learned
  positions, `depth` pre-norm blocks with heads of `head_width`, and a head on
  the class token. 12x96 on 32x32 CIFAR-100 has 1,363,012 parameters."""
  channels, side = input_shape[0], input_shape[1]
  patch = patch or side // 8
  if side % patch or width % head_width:
    raise ValueError(
      f"patches of {patch} must tile {side}, and heads of {head_width} split {width}"
    )
  stages: dict[str, nn.Module] = {"embed": PatchEmbedding(channels, width, side, patch)}
  stages |= {
    f"block{i}": TransformerBlock(width, width // head_width)
    for i in range(1, depth + 1)
  }
  stages["head"] = ViTHead(width, num_classes)
  model = Expert(stages)
  model.no_weight_decay = frozenset({"cls", "pos"})
  for m in model.modules():
    if isinstance(m, nn.Linear):
      nn.init.trunc_normal_(m.weight, std=0.02)
      nn.init.zeros_(m.bias)
  return model


# ---------------------------------------------------------------- registry

ARCHITECTURES = {"resnet": resnet, "vit": vit}

# The recipe each family is usually trained with (docs/testbed/experts.md):
# the ResNet's SGD and the ViT's AdamW, both on a cosine schedule, with random
# crop and flip, as the v4 selection trained them.
RECIPES: dict[str, Budget] = {
  "resnet": Budget(
    epochs=100,
    batch_size=128,
    optimizer="sgd",
    lr=0.1,
    weight_decay=5e-4,
    schedule="cosine",
    augment=True,
    bf16=True,
  ),
  "vit": Budget(
    epochs=200,
    batch_size=256,
    optimizer="adamw",
    lr=1e-3,
    weight_decay=0.05,
    schedule="cosine",
    warmup=0.05,
    label_smoothing=0.1,
    grad_clip=1.0,
    augment=True,
    bf16=True,
  ),
}


def build(arch: str, input_shape: tuple[int, ...], num_classes: int) -> Expert:
  return ARCHITECTURES[arch](input_shape, num_classes)


def recipe(arch: str) -> Budget:
  """The recipe `arch` is trained with by default. An architecture registered
  without one (a test's, say) gets Budget's defaults."""
  return RECIPES.get(arch, Budget())


def register(name: str, builder, budget: Budget | None = None, **sizes) -> None:
  """Add an architecture, e.g. a smaller size of a family:
  register("resnet_8x32", resnet, width=32, depth=8)."""
  ARCHITECTURES[name] = partial(builder, **sizes) if sizes else builder
  if budget is not None:
    RECIPES[name] = budget
