"""
The communication channel: encoders, a mixer, decoders.

One exchange reads every expert's features at a read point, mixes them across
experts and writes back into every expert at a write point. Its channel has
three parts, each a small contract:

  Encoder  one per expert. Gets the raw features at the read point, a full
           (B, C, H, W) map or a (B, D) vector; nothing is pooled before it
           sees them. Returns that expert's message: a query, key and value,
           each (B, T, D) with T the number of messages per expert.
  Mixer    one per exchange. Mixes the messages of all M experts and returns
           what each expert received, (B, M, T, D_value), with the weights it
           mixed them by when it has any.
  Decoder  one per expert. Gets what its expert received and that expert's
           current features at the write point, and returns an injection of
           exactly the features' shape, which is added to them. Its last layer
           starts at zero, so an untrained channel injects nothing: the
           ensemble starts as bagging.

The `linear_taper` encoders and decoders are two linear layers through a
narrow width that read and write the whole feature map, so they never pool or
broadcast; `linear_taper_gelu` adds a GELU between the two. The `self` and
`mean` mixers are controls: the same channel without communication, or without
choosing whom to listen to. `pooled`, `spatial` and `broadcast` make spatial
assumptions. docs/testbed/channel.md defines every part.

For now every encoder sends exactly one message per expert
(MESSAGES_PER_EXPERT). The mixer and decoders already handle any T, so
allowing more takes an encoder that sends them and a change to that constant.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MESSAGES_PER_EXPERT = 1


@dataclass(frozen=True)
class ChannelConfig:
  encoder: str = "linear_taper"
  mixer: str = "attention"
  decoder: str = "linear_taper"
  key_dim: int = 16
  value_dim: int = 64
  width: int = 8  # r: the narrow middle of the linear taper parts


@dataclass
class Message:
  query: torch.Tensor  # (B, T, key_dim) what the sender wants to hear about
  key: torch.Tensor  # (B, T, key_dim) what the message is about
  value: torch.Tensor  # (B, T, value_dim) what it says


class Encoder(nn.Module, ABC):
  @abstractmethod
  def forward(self, features: torch.Tensor) -> Message: ...


class Mixer(nn.Module, ABC):
  @abstractmethod
  def forward(
    self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(B, M, T, D) each -> what each expert received, (B, M, T, value_dim),
    and the mixing weights, (B, M, T, M, T): how much each received message
    drew on each sent one. None for a mixer whose weights are fixed by
    construction, which leaves no routing to read."""


class Decoder(nn.Module, ABC):
  @abstractmethod
  def forward(self, received: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """(B, T, value_dim) and the features at the write point -> an injection shaped like the features."""


@dataclass
class Reading:
  """One exchange's read: what every expert sent, how it was mixed, and what
  each expert received."""

  sent: Message  # each part stacked over experts, (B, M, T, D)
  weights: torch.Tensor | None  # (B, M, T, M, T), as the mixer returns them
  received: torch.Tensor  # (B, M, T, value_dim)


def positions(features: torch.Tensor) -> torch.Tensor:
  """(B, C, H, W) -> (B, H*W, C), one row per position; (B, D) -> (B, 1, D)."""
  return (
    features.flatten(2).transpose(1, 2) if features.dim() > 2 else features.unsqueeze(1)
  )


class LinearTaperEncoder(Encoder):
  """Two linear layers through a narrow width. The first reads the whole
  feature map, every channel at every position, through `width` learned
  templates; the second builds the query, key and value from those readings.
  So the message is one linear map of the features, and it carries at most
  `width` independent numbers."""

  gelu = False

  def __init__(self, config: ChannelConfig, in_shape: tuple[int, ...]):
    super().__init__()
    self.read = nn.Linear(math.prod(in_shape), config.width)
    self.query = nn.Linear(config.width, config.key_dim)
    self.key = nn.Linear(config.width, config.key_dim)
    self.value = nn.Linear(config.width, config.value_dim)

  def forward(self, features: torch.Tensor) -> Message:
    readings = self.read(features.flatten(1)).unsqueeze(1)  # (B, 1, width)
    if self.gelu:
      readings = F.gelu(readings)
    return Message(self.query(readings), self.key(readings), self.value(readings))


class LinearTaperGELUEncoder(LinearTaperEncoder):
  """The linear taper encoder with a GELU on its readings, so the message can
  be a nonlinear function of the features. Its weights are the linear taper
  encoder's."""

  gelu = True


class PooledEncoder(Encoder):
  """Averages over positions inside the encoder, then one linear head each for
  query, key and value. The v2 default, and no spatial read port beat it in
  v2's Study 02."""

  def __init__(self, config: ChannelConfig, in_shape: tuple[int, ...]):
    super().__init__()
    channels = in_shape[0]
    self.query = nn.Linear(channels, config.key_dim)
    self.key = nn.Linear(channels, config.key_dim)
    self.value = nn.Linear(channels, config.value_dim)

  def forward(self, features: torch.Tensor) -> Message:
    summary = positions(features).mean(1, keepdim=True)  # (B, 1, C)
    return Message(self.query(summary), self.key(summary), self.value(summary))


class SpatialEncoder(Encoder):
  """Weighs every position, tagged with a learned position embedding, before
  summing them into the message, so where an expert saw something can shape
  what it says."""

  def __init__(self, config: ChannelConfig, in_shape: tuple[int, ...]):
    super().__init__()
    channels, count = in_shape[0], math.prod(in_shape[1:])
    self.position = nn.Parameter(torch.randn(count, channels) * 0.02)
    self.score = nn.Linear(channels, 1)
    self.query = nn.Linear(channels, config.key_dim)
    self.key = nn.Linear(channels, config.key_dim)
    self.value = nn.Linear(channels, config.value_dim)

  def forward(self, features: torch.Tensor) -> Message:
    rows = positions(features) + self.position  # (B, P, C)
    weights = self.score(rows).softmax(1)  # (B, P, 1)
    summary = (weights * rows).sum(1, keepdim=True)  # (B, 1, C)
    return Message(self.query(summary), self.key(summary), self.value(summary))


class AttentionMixer(Mixer):
  """Single-head scaled dot-product attention over every expert's messages,
  each expert's own included. With M = 2, leaving self out would give each
  receiver one sender and a softmax stuck at 1 (TarMAC; v2's ensemble/communicative/bus.py, on main)."""

  def __init__(self, config: ChannelConfig):
    super().__init__()

  def forward(
    self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
  ) -> torch.Tensor:
    m, t = query.shape[1:3]
    q, k, v = (x.flatten(1, 2) for x in (query, key, value))  # (B, M*T, D)
    weights = (q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])).softmax(-1)
    received = (weights @ v).unflatten(1, (m, t))
    return received, weights.unflatten(1, (m, t)).unflatten(3, (m, t))


class SelfMixer(Mixer):
  """A control: each expert receives only its own message. The channel keeps
  all its parts but loses communication, so what it still gains over bagging
  is each expert's own add-on, trained on the joint split. The query and key go
  unused, so they get no gradient."""

  def __init__(self, config: ChannelConfig):
    super().__init__()

  def forward(
    self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
  ) -> tuple[torch.Tensor, None]:
    return value, None


class MeanMixer(Mixer):
  """A second control: every expert receives the plain average of every
  expert's message, its own included. That is communication without choosing
  whom to listen to. The query and key go unused, so they get no gradient."""

  def __init__(self, config: ChannelConfig):
    super().__init__()

  def forward(
    self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
  ) -> tuple[torch.Tensor, None]:
    return value.mean((1, 2), keepdim=True).expand_as(value), None


class LinearTaperDecoder(Decoder):
  """Two linear layers through a narrow width. The first turns what was
  received into `width` numbers, each the strength of one learned pattern the
  size of the whole write point; the second injects that mix of patterns. The
  patterns have no bias, so the injection is always a mix of them, and they
  start at zero."""

  gelu = False

  def __init__(self, config: ChannelConfig, out_shape: tuple[int, ...]):
    super().__init__()
    self.narrow = nn.Linear(config.value_dim, config.width)
    self.write = nn.Linear(config.width, math.prod(out_shape), bias=False)
    nn.init.zeros_(self.write.weight)

  def forward(self, received: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    strengths = self.narrow(received.mean(1))  # (B, width)
    if self.gelu:
      strengths = F.gelu(strengths)
    return self.write(strengths).view(features.shape)


class LinearTaperGELUDecoder(LinearTaperDecoder):
  """The linear taper decoder with a GELU on the patterns' strengths, so the
  injection can be a nonlinear function of the message. Its weights are the
  linear taper decoder's."""

  gelu = True


class BroadcastDecoder(Decoder):
  """Turns what was received into one value per channel and injects it at
  every position."""

  def __init__(self, config: ChannelConfig, out_shape: tuple[int, ...]):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(config.value_dim, config.value_dim),
      nn.GELU(),
      nn.Linear(config.value_dim, out_shape[0]),
    )
    nn.init.zeros_(self.mlp[-1].weight)
    nn.init.zeros_(self.mlp[-1].bias)

  def forward(self, received: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    injection = self.mlp(received.mean(1))  # (B, C)
    return injection.view(*injection.shape, *[1] * (features.dim() - 2)).expand_as(
      features
    )


class SpatialDecoder(Decoder):
  """Every position of the receiving expert's features combines its own
  features with what was received, so a message can land differently in
  different places."""

  def __init__(self, config: ChannelConfig, out_shape: tuple[int, ...]):
    super().__init__()
    channels = out_shape[0]
    self.mlp = nn.Sequential(
      nn.Linear(channels + config.value_dim, config.value_dim),
      nn.GELU(),
      nn.Linear(config.value_dim, channels),
    )
    nn.init.zeros_(self.mlp[-1].weight)
    nn.init.zeros_(self.mlp[-1].bias)

  def forward(self, received: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    rows = positions(features)  # (B, P, C)
    message = received.mean(1, keepdim=True).expand(-1, rows.shape[1], -1)
    injection = self.mlp(torch.cat([rows, message], dim=-1))  # (B, P, C)
    return injection.transpose(1, 2).reshape(features.shape)


ENCODERS: dict[str, type[Encoder]] = {
  "linear_taper": LinearTaperEncoder,
  "linear_taper_gelu": LinearTaperGELUEncoder,
  "pooled": PooledEncoder,
  "spatial": SpatialEncoder,
}
MIXERS: dict[str, type[Mixer]] = {
  "attention": AttentionMixer,
  "self": SelfMixer,
  "mean": MeanMixer,
}
DECODERS: dict[str, type[Decoder]] = {
  "linear_taper": LinearTaperDecoder,
  "linear_taper_gelu": LinearTaperGELUDecoder,
  "broadcast": BroadcastDecoder,
  "spatial": SpatialDecoder,
}


class Channel(nn.Module):
  """Everything one exchange trains: an encoder and a decoder per expert, and
  the mixer between them."""

  def __init__(self, encoders: list[Encoder], mixer: Mixer, decoders: list[Decoder]):
    super().__init__()
    self.encoders = nn.ModuleList(encoders)
    self.mixer = mixer
    self.decoders = nn.ModuleList(decoders)

  def read(self, features: list[torch.Tensor]) -> Reading:
    """Every expert's features at the read point -> what each sent and received."""
    messages = [encode(f) for encode, f in zip(self.encoders, features)]
    for message in messages:
      if message.value.shape[1] != MESSAGES_PER_EXPERT:
        raise ValueError(
          f"an encoder sent {message.value.shape[1]} messages; "
          f"each expert sends {MESSAGES_PER_EXPERT} for now"
        )
    sent = Message(
      torch.stack([m.query for m in messages], dim=1),
      torch.stack([m.key for m in messages], dim=1),
      torch.stack([m.value for m in messages], dim=1),
    )
    received, weights = self.mixer(sent.query, sent.key, sent.value)
    return Reading(sent, weights, received)

  def write(
    self, received: torch.Tensor, features: list[torch.Tensor]
  ) -> list[torch.Tensor]:
    """What each expert's decoder writes at the write point: an injection
    shaped like its features. The caller adds it, so the injection itself
    stays available, exact, for the metrics."""
    injections = []
    for i, (decode, f) in enumerate(zip(self.decoders, features)):
      injection = decode(received[:, i], f)
      assert injection.shape == f.shape, (
        f"decoder returned {tuple(injection.shape)} for features {tuple(f.shape)}"
      )
      injections.append(injection)
    return injections


def build_channel(
  config: ChannelConfig,
  read_shape: tuple[int, ...],
  write_shape: tuple[int, ...],
  num_experts: int,
) -> Channel:
  return Channel(
    [ENCODERS[config.encoder](config, read_shape) for _ in range(num_experts)],
    MIXERS[config.mixer](config),
    [DECODERS[config.decoder](config, write_shape) for _ in range(num_experts)],
  )
