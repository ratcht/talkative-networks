"""
Phase 5 of the channel ladder: why queries and keys grow
(docs/studies/channel-ladder.md). Every arm is level 1, forward, and differs in
one thing that training may or may not touch. These are measurements of level 1,
not channel parts, so they live here and go in through testbed.run's `adjust`.

  uv run python -m studies.channel_ladder.phase5 level1-routing-cut-stage2-stage3
  uv run python -m studies.channel_ladder.phase5 level1-self-centred-stage2-stage3 \\
    --device cuda:1 --trial-seeds 0 1 2

Every other flag is testbed.run's, as for studies.channel_ladder.run.
"""

import argparse
import sys
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from studies.channel_ladder import COSINE, MIXERS
from studies.channel_ladder.run import flags_for_testbed
from testbed import run
from testbed.communication import CommunicatingEnsemble, Encoder, Message
from testbed.communication.channel import LinearTaperEncoder
from testbed.data import Data
from testbed.experts.architectures import Expert

PHASE = "5"
LEVEL = "1"
EXCHANGE = ("stage2", "stage3")  # the forward pair phase 1 picked
ATTENTION, SELF = MIXERS[0], MIXERS[1]
NONE = "none"
EVERY_EXPERT = (0, 1, 2)
FIRST_EXPERT = (0,)  # first in group order, so a different one on each trial seed


@dataclass(frozen=True)
class Changes:
  """What an arm does to level 1's encoders. By default, nothing."""

  centred_input: bool = False  # subtract the expert's average features
  frozen_readings: bool = False  # the read layer never trains
  routing_cut: tuple[int, ...] = ()  # these experts' query and key send no
  value_cut: tuple[int, ...] = ()  # or value, gradient back into the readings
  no_head_biases: bool = False  # for every expert's query and key heads


@dataclass(frozen=True)
class Arm:
  mixer: str
  intervention: str = NONE
  changes: Changes = Changes()

  @property
  def label(self) -> str:
    """Named like the arms of phases 1 and 4, so the references match theirs."""
    mixer = [] if self.mixer == ATTENTION else [self.mixer]
    what = [] if self.intervention == NONE else [self.intervention]
    return "-".join([f"level{LEVEL}", *mixer, *what, *EXCHANGE])


ARMS: dict[str, Arm] = {
  arm.label: arm
  for arm in (
    Arm(SELF, "centred", Changes(centred_input=True)),
    Arm(ATTENTION, "centred", Changes(centred_input=True)),
    Arm(ATTENTION, "frozen-readings", Changes(frozen_readings=True)),
    Arm(ATTENTION, "routing-cut", Changes(routing_cut=EVERY_EXPERT)),
    Arm(COSINE, "value-cut", Changes(value_cut=FIRST_EXPERT)),
    Arm(
      COSINE,
      "value-cut-no-bias",
      Changes(value_cut=FIRST_EXPERT, no_head_biases=True),
    ),
    # the references
    Arm(SELF),
    Arm(ATTENTION),
    Arm(COSINE),
  )
}


class EncoderUnderTest(Encoder):
  """A trained level 1 encoder's own layers, unchanged, with the phase's
  changes on the way through them. Keeping the layers keeps their names, so the
  metrics read the same parts, and their initial weights, so an arm pairs with
  its reference trial seed by trial seed."""

  def __init__(
    self,
    original: LinearTaperEncoder,
    centre: torch.Tensor | None,
    cut_routing: bool,
    cut_value: bool,
  ):
    super().__init__()
    self.read, self.query = original.read, original.query
    self.key, self.value = original.key, original.value
    self.gelu = original.gelu
    self.register_buffer("centre", centre)
    self.cut_routing, self.cut_value = cut_routing, cut_value

  def forward(self, features: torch.Tensor) -> Message:
    flat = features.flatten(1)
    readings = self.read(flat if self.centre is None else flat - self.centre)
    readings = readings.unsqueeze(1)  # (B, 1, width)
    if self.gelu:
      readings = F.gelu(readings)
    routing = readings.detach() if self.cut_routing else readings
    value = readings.detach() if self.cut_value else readings
    return Message(self.query(routing), self.key(routing), self.value(value), readings)


@torch.no_grad()
def average_features(
  expert: Expert, x: torch.Tensor, point: str, batch_size: int = 2000
) -> torch.Tensor:
  """The expert's average features at a point, flattened: the mean image it
  sees there, over x."""
  total = sum(expert.run(b, until=point).flatten(1).sum(0) for b in x.split(batch_size))
  return total / len(x)


def apply(changes: Changes, model: CommunicatingEnsemble, data: Data) -> None:
  """Make the arm's changes to a built ensemble. The average is taken on the
  joint split, which the channel trains on; joint_val and test stay unseen."""
  (exchange,), (channel,) = model.exchanges, model.channels
  for i, (expert, encoder) in enumerate(zip(model.experts, channel.encoders)):
    centre = None
    if changes.centred_input:
      centre = average_features(expert, data.joint.x, exchange.read)
    channel.encoders[i] = EncoderUnderTest(
      encoder, centre, i in changes.routing_cut, i in changes.value_cut
    )
    if changes.frozen_readings:
      encoder.read.requires_grad_(False)
    if changes.no_head_biases:
      for head in (encoder.query, encoder.key):
        nn.init.zeros_(head.bias)
        head.bias.requires_grad_(False)


def flags(arm: Arm) -> list[str]:
  """An arm as studies.channel_ladder.run's flags, tagged with its intervention,
  since channel parameters can't show one."""
  return [
    *("--phase", PHASE, "--level", LEVEL, "--mixer", arm.mixer),
    *("--arm", arm.label, "--exchange", *EXCHANGE),
    *("--tag", f"intervention={arm.intervention}"),
  ]


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False,
  )
  p.add_argument("arm", choices=ARMS)
  args, rest = p.parse_known_args(sys.argv[1:] if argv is None else argv)
  arm = ARMS[args.arm]
  run.main(
    flags_for_testbed([*flags(arm), *rest]),
    partial(apply, arm.changes) if arm.changes != Changes() else None,
  )


if __name__ == "__main__":
  main()
