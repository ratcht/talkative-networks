"""
A communicating ensemble: frozen experts and the exchanges between them.

An exchange reads at one point and writes at another (see channel.py).
  forward   writes at a later point than it reads, in a single pass. Several
            forward exchanges may stack down the network, but each must read
            after the previous one has written *and* the experts have
            processed that write, so no two exchanges share a point.
  backward  writes at an earlier point than it reads. The stages in between
            re-run after each write, so the exchange repeats for K rounds.
Reading and writing at the same point is not supported (docs/docket.md lists
it as an extension).

With no exchanges, or an untrained channel, the output is exactly bagging:
the log of the mean of the experts' probabilities.

Every forward pass also returns a trace of each exchange, which the metrics
read (testbed/metrics, docs/testbed/metrics.md).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.communication.channel import Channel, ChannelConfig, Reading, build_channel
from testbed.experts.architectures import Expert
from testbed.seeds import seeded


@dataclass(frozen=True)
class Exchange:
  read: str
  write: str


@dataclass
class Trace:
  """One exchange as it ran: what was read, mixed and written."""

  exchange: int  # index into the ensemble's exchanges, and so its channels
  round: int | None  # which round of a backward exchange; None for forward
  reading: Reading
  injection: list[
    torch.Tensor
  ]  # per expert, shaped like its features at the write point

  @property
  def site(self) -> str:
    """'ex0' for a forward exchange, 'ex0/round1' for a backward exchange's second round."""
    return f"ex{self.exchange}" + ("" if self.round is None else f"/round{self.round}")


@dataclass
class Output:
  log_probs: torch.Tensor  # (B, C) log of the mean expert probability: pair with NLLLoss, not CrossEntropyLoss
  logits: torch.Tensor  # (B, M, C) each expert's own answer
  traces: list[Trace]  # in the order the exchanges ran; empty without communication


def direction(exchanges: list[Exchange], points: list[str], rounds: int) -> str:
  """'none', 'forward' or 'backward'; raises if the exchanges break the rules above."""
  order = {p: i for i, p in enumerate(points)}
  for ex in exchanges:
    if ex.read not in order or ex.write not in order:
      raise ValueError(f"{ex} names a point the experts don't have; they have {points}")
    if ex.read == ex.write:
      raise ValueError(
        f"{ex} reads and writes at the same point, which isn't supported"
      )
  if not exchanges:
    return "none"
  if any(order[ex.write] < order[ex.read] for ex in exchanges):
    if len(exchanges) > 1:
      raise ValueError("a backward exchange can't be combined with other exchanges")
    if rounds < 1:
      raise ValueError("a backward exchange needs at least one round")
    return "backward"
  if rounds != 1:
    raise ValueError(
      "rounds apply only to a backward exchange; forward exchanges stack instead"
    )
  ordered = sorted(exchanges, key=lambda ex: order[ex.read])
  for before, after in zip(ordered, ordered[1:]):
    if order[after.read] <= order[before.write]:
      raise ValueError(
        f"{after} reads before the experts have processed {before}'s write"
      )
  return "forward"


def average(logits: torch.Tensor) -> torch.Tensor:
  """(B, M, C) logits -> (B, C) log of the mean probability, via logsumexp."""
  return torch.logsumexp(F.log_softmax(logits, dim=-1), dim=1) - math.log(
    logits.shape[1]
  )


class CommunicatingEnsemble(nn.Module):
  def __init__(
    self,
    experts: list[Expert],
    exchanges: list[Exchange],
    channels: list[Channel],
    rounds: int = 1,
  ):
    super().__init__()
    points = experts[0].points
    if any(e.points != points for e in experts):
      raise ValueError("experts must share their points")
    if len(channels) != len(exchanges):
      raise ValueError("one channel per exchange")
    self.direction = direction(exchanges, points, rounds)
    self.experts = nn.ModuleList(experts)
    self.experts.requires_grad_(False)
    self.exchanges = list(exchanges)
    self.channels = nn.ModuleList(channels)
    self.rounds = rounds

  def train(self, mode: bool = True):
    super().train(mode)
    self.experts.eval()  # frozen experts never leave eval mode
    return self

  def forward(self, x: torch.Tensor) -> Output:
    run = self._backward if self.direction == "backward" else self._forward
    logits, traces = run(x)
    return Output(average(logits), logits, traces)

  def bagging(self, x: torch.Tensor) -> Output:
    """The same experts with no communication: the zero point."""
    logits = torch.stack([expert(x) for expert in self.experts], dim=1)
    return Output(average(logits), logits, [])

  def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[Trace]]:
    """One pass down the stages; each exchange reads on the way and writes further down."""
    features = [x] * len(self.experts)
    readings: dict[int, Reading] = {}
    traces = []
    for stage in self.experts[0].stages:
      features = [expert.stages[stage](f) for expert, f in zip(self.experts, features)]
      for i, ex in enumerate(self.exchanges):
        if ex.write == stage:
          reading = readings.pop(i)
          injection = self.channels[i].write(reading.received, features)
          features = [f + d for f, d in zip(features, injection)]
          traces.append(Trace(i, None, reading, injection))
        if ex.read == stage:
          readings[i] = self.channels[i].read(features)
    return torch.stack(features, dim=1), traces

  def _backward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[Trace]]:
    """A perception pass, then K rounds of: read late, write early, re-run the stages between."""
    (ex,) = self.exchanges
    (channel,) = self.channels
    early = [expert.run(x, until=ex.write) for expert in self.experts]
    late = [
      expert.run(f, after=ex.write, until=ex.read)
      for expert, f in zip(self.experts, early)
    ]
    traces = []
    for r in range(self.rounds):
      # each round writes into the original early features, so messages don't accumulate
      reading = channel.read(late)
      injection = channel.write(reading.received, early)
      late = [
        expert.run(f + d, after=ex.write, until=ex.read)
        for expert, f, d in zip(self.experts, early, injection)
      ]
      traces.append(Trace(0, r, reading, injection))
    logits = torch.stack(
      [expert.run(f, after=ex.read) for expert, f in zip(self.experts, late)], dim=1
    )
    return logits, traces


def build_ensemble(
  experts: list[Expert],
  exchanges: list[Exchange],
  config: ChannelConfig,
  input_shape: tuple[int, ...],
  rounds: int,
  seed: int,
) -> CommunicatingEnsemble:
  """Size each channel from the experts' point shapes and initialise it on its own seed."""
  direction(exchanges, experts[0].points, rounds)  # fail before building anything
  shapes = experts[0].point_shapes(input_shape)
  with seeded(seed):
    channels = [
      build_channel(config, shapes[ex.read], shapes[ex.write], len(experts))
      for ex in exchanges
    ]
  device = next(experts[0].parameters()).device
  return CommunicatingEnsemble(experts, exchanges, channels, rounds).to(device)
