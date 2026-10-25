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

For now every encoder sends exactly one message per expert
(MESSAGES_PER_EXPERT). The mixer and decoders already handle any T, so
allowing more takes an encoder that sends them and a change to that constant.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

MESSAGES_PER_EXPERT = 1


@dataclass(frozen=True)
class ChannelConfig:
  encoder: str = "pooled"
  mixer: str = "attention"
  decoder: str = "broadcast"
  key_dim: int = 16
  value_dim: int = 64


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
    drew on each sent one. None for a mixer that doesn't weight senders."""


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
  "pooled": PooledEncoder,
  "spatial": SpatialEncoder,
}
MIXERS: dict[str, type[Mixer]] = {"attention": AttentionMixer}
DECODERS: dict[str, type[Decoder]] = {
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
