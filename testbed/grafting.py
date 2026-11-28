"""
Model grafting: one expert of the group, the anchor, sees the image and gives
the answer. The others, the obligates, never see the image and their answers
are never read; they take part only through two exchanges in one forward pass
(docs/testbed/grafting.md):

  outbound  the anchor's stream at the read point is stitched into every
            obligate's stream at the same point (Bansal et al. 2021): the
            stitch's output is the whole stream the obligate starts from, so
            its embedding and earlier blocks never run.
  return    each obligate runs its own blocks from the read point to the
            write point. Every position of the anchor's stream at the write
            point then attends over every position of every obligate's stream
            there (CALM, Bansal et al. 2024), and the result is added to the
            anchor's stream, which runs on to the anchor's head.

This is the side branch: both stitches join the same depth in the two
networks, and the anchor runs all of its own blocks. The last layer of the
return starts at zero, so before training the output is exactly the anchor
alone.

  read     the anchor's read point, where the obligates are written
  write    the anchor's write point, where the obligates are read
  mixer    how the anchor combines the replies (MIXERS):
             attention  CALM's cross-attention over every position of every
                        obligate, one softmax over all of them
             mean       each anchor position takes the plain average of the
                        same position in every obligate; nothing is chosen
  stitch   the outbound map (STITCHES); None picks the experts' family's own
  heads    attention heads; None gives one per 32 channels of the stream
  bypass   the control: the same channel with the obligates' blocks from the
           read point to the write point left out, so the stitched stream goes
           straight to the return
  dropout  share of each obligate's projected stream zeroed in training, before
           the return mixes them; 0 by default

In evaluation the graft also runs each obligate from its stitched stream
through its own remaining blocks and head, and the meter scores that answer
(stitched_accuracy). It is a measurement of what the stitch hands the
obligate, with no gradient, and never part of the graft's answer.

Only the channel trains: the stitches, the per-obligate projections (CALM's
f_proj) and the attention's query, key, value and output maps.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.communication.ensemble import Output
from testbed.experts.architectures import Expert
from testbed.metrics.grafting import GraftMeter
from testbed.seeds import seeded


@dataclass(frozen=True)
class GraftConfig:
  read: str = "block4"
  write: str = "block8"
  mixer: str = "attention"
  stitch: str | None = None
  heads: int | None = None
  bypass: bool = False
  dropout: float = 0.0


class PointwiseLinear(nn.Module):
  """The same affine map from C channels to C channels at every position of
  the stream: a 1x1 convolution on a ResNet's grid, a token-wise linear map
  on a ViT's tokens, the class token included."""

  def __init__(self, channels: int):
    super().__init__()
    self.map = nn.Conv1d(channels, channels, 1)

  def forward(self, stream: torch.Tensor) -> torch.Tensor:
    return self.map(stream.flatten(2)).view_as(stream)


class BNPointwiseLinearBN(PointwiseLinear):
  """pointwise_linear between two batch norms: Bansal et al.'s stitch for
  convolutional networks. The norms help optimisation only; at evaluation the
  whole stitch is one affine map at every position."""

  def __init__(self, channels: int):
    super().__init__(channels)
    self.before = nn.BatchNorm1d(channels)
    self.after = nn.BatchNorm1d(channels)

  def forward(self, stream: torch.Tensor) -> torch.Tensor:
    return self.after(self.map(self.before(stream.flatten(2)))).view_as(stream)


STITCHES: dict[str, type[nn.Module]] = {
  "pointwise_linear": PointwiseLinear,
  "bn_pointwise_linear_bn": BNPointwiseLinearBN,
}
MIXERS = ("attention", "mean")


def default_stitch(stream_shape: tuple[int, ...]) -> str:
  """A grid (C, H, W) gets the batch-normed stitch, a token stream (C, P) the
  plain one, as in Bansal et al."""
  return "bn_pointwise_linear_bn" if len(stream_shape) == 3 else "pointwise_linear"


def rows(stream: torch.Tensor) -> torch.Tensor:
  """(B, C, *positions) -> (B, P, C): one row per position, every position
  included (a ViT's class token too)."""
  return stream.flatten(2).transpose(1, 2)


@dataclass
class Mixed:
  injection: torch.Tensor  # shaped like the anchor's stream at the write point
  # (B, P, J): how much each anchor position drew on each obligate, averaged
  # over heads; None for the mean, which chooses nothing
  shares: torch.Tensor | None = None
  query: torch.Tensor | None = None  # (B, P, C), before the split into heads
  key: torch.Tensor | None = None  # (B, J * P, C)


class GraftChannel(nn.Module):
  """Everything grafting trains: one stitch and one projection per obligate,
  and the return's maps. The output map starts at zero."""

  def __init__(
    self, config: GraftConfig, stream_shape: tuple[int, ...], obligates: int
  ):
    super().__init__()
    channels = stream_shape[0]
    stitch = STITCHES[config.stitch or default_stitch(stream_shape)]
    self.mixer = config.mixer
    self.dropout = config.dropout
    self.heads = config.heads or max(1, channels // 32)
    if channels % self.heads:
      raise ValueError(f"{channels} channels don't split into {self.heads} heads")
    self.stitch = nn.ModuleList(stitch(channels) for _ in range(obligates))
    self.project = nn.ModuleList(
      nn.Linear(channels, channels) for _ in range(obligates)
    )
    if config.mixer == "attention":
      self.query = nn.Linear(channels, channels)
      self.key = nn.Linear(channels, channels)
    self.value = nn.Linear(channels, channels)
    self.out = nn.Linear(channels, channels)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)

  def mix(self, anchor: torch.Tensor, replies: list[torch.Tensor]) -> Mixed:
    """The anchor's stream at the write point and every obligate's reply
    there -> what is added to the anchor's stream."""
    projected = [
      F.dropout(project(rows(r)), self.dropout, self.training)
      for project, r in zip(self.project, replies)
    ]
    if self.mixer == "mean":
      received = self.value(torch.stack(projected).mean(0))  # (B, P, C)
      return Mixed(self._back(self.out(received), anchor))
    c = anchor.shape[1]
    h, j = self.heads, len(replies)
    query = self.query(rows(anchor))  # (B, P, C)
    sent = torch.cat(projected, dim=1)  # (B, J * P, C)
    key = self.key(sent)

    def split(x: torch.Tensor) -> torch.Tensor:  # (B, N, C) -> (B, h, N, C / h)
      return x.unflatten(-1, (h, c // h)).transpose(1, 2)

    weights = (split(query) @ split(key).transpose(-1, -2) / math.sqrt(c // h)).softmax(
      -1
    )  # (B, h, P, J * P)
    received = (weights @ split(self.value(sent))).transpose(1, 2).flatten(2)
    shares = weights.unflatten(-1, (j, -1)).sum(-1).mean(1)  # (B, P, J)
    return Mixed(self._back(self.out(received), anchor), shares, query, key)

  @staticmethod
  def _back(received_rows: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """(B, P, C) rows -> shaped like the anchor's stream."""
    return received_rows.transpose(1, 2).reshape(anchor.shape)


MIXED_PARTS = ("query", "key")


@dataclass
class GraftTrace:
  """One forward pass of the graft as it ran, for the metrics."""

  stitched: list[torch.Tensor]  # per obligate: its stream as the stitch made it
  replies: list[torch.Tensor]  # per obligate: its stream at the write point
  mixed: Mixed
  tokens: int = 0  # leading positions of the stream that aren't grid positions
  site: str = "ex0"
  new_meter: type = field(default=GraftMeter, repr=False)
  # (B, J): each obligate's own answer, run from its stitched stream through its
  # remaining blocks and head; evaluation only, a measurement, never the answer
  obligate_answers: torch.Tensor | None = None


class GraftedEnsemble(nn.Module):
  """experts[0] is the anchor; every other expert is an obligate."""

  direction = "graft"
  # only the anchor answers, so the metrics that compare experts' answers don't apply
  ensemble_answers = False

  def __init__(self, experts: list[Expert], channel: GraftChannel, config: GraftConfig):
    super().__init__()
    self.experts = nn.ModuleList(experts)
    self.experts.requires_grad_(False)
    self.channel = channel
    self.config = config
    self.tokens = experts[0].tokens

  def train(self, mode: bool = True):
    super().train(mode)
    self.experts.eval()  # frozen experts never leave eval mode
    return self

  @property
  def trainable(self) -> nn.Module:
    """What joint training trains: the channel, as exchange 0."""
    return nn.ModuleList([self.channel])

  def forward(self, x: torch.Tensor, answer_each_round: bool = False) -> Output:
    """answer_each_round belongs to backward exchanges; a graft has none."""
    anchor, obligates = self.experts[0], self.experts[1:]
    read, write = self.config.read, self.config.write
    with torch.no_grad():  # nothing trained comes before these
      at_read = anchor.run(x, until=read)
      at_write = anchor.run(at_read, after=read, until=write)
    stitched = [stitch(at_read) for stitch in self.channel.stitch]
    if self.config.bypass:
      replies = stitched
    else:
      replies = [
        obligate.run(s, after=read, until=write)
        for obligate, s in zip(obligates, stitched)
      ]
    mixed = self.channel.mix(at_write, replies)
    logits = anchor.run(at_write + mixed.injection, after=write)[:, None]  # (B, 1, C)
    answers = None
    if not self.training:  # stitched_accuracy: measured, never part of the answer
      with torch.no_grad():
        if self.config.bypass:
          ends = [o.run(s, after=read) for o, s in zip(obligates, stitched)]
        else:
          ends = [o.run(r, after=write) for o, r in zip(obligates, replies)]
      answers = torch.stack([end.argmax(-1) for end in ends], dim=1)
    trace = GraftTrace(stitched, replies, mixed, self.tokens, obligate_answers=answers)
    return Output(F.log_softmax(logits[:, 0], -1), logits, [trace])

  def baseline(self, x: torch.Tensor) -> Output:
    """The anchor alone: the zero point."""
    logits = self.experts[0](x)[:, None]
    return Output(F.log_softmax(logits[:, 0], -1), logits, [])


def build_graft(
  experts: list[Expert],
  config: GraftConfig,
  input_shape: tuple[int, ...],
  seed: int,
) -> GraftedEnsemble:
  """experts[0] is the anchor. Sizes the channel from the experts' stream and
  initialises it on its own seed."""
  if len(experts) < 2:
    raise ValueError("a graft needs an anchor and at least one obligate")
  if config.mixer not in MIXERS:
    raise ValueError(f"unknown mixer {config.mixer!r}; choose from {MIXERS}")
  if not 0 <= config.dropout < 1:
    raise ValueError(f"dropout must be in [0, 1), not {config.dropout}")
  points = experts[0].points
  if any(e.points != points for e in experts):
    raise ValueError("experts must share their points")
  for name in (config.read, config.write):
    if name not in points:
      raise ValueError(f"{name!r} isn't a point; the experts have {points}")
  if points.index(config.write) <= points.index(config.read):
    raise ValueError(
      f"the graft must write ({config.write}) after it reads ({config.read})"
    )
  shape = experts[0].stream_shape(input_shape)
  for expert in experts[1:]:
    if expert.stream_shape(input_shape) != shape:
      raise ValueError("experts must share one stream shape")
  with seeded(seed):
    channel = GraftChannel(config, shape, len(experts) - 1)
  resolved = GraftConfig(
    config.read,
    config.write,
    config.mixer,
    config.stitch or default_stitch(shape),
    channel.heads,
    config.bypass,
    config.dropout,
  )
  device = next(experts[0].parameters()).device
  return GraftedEnsemble(experts, channel, resolved).to(device)
