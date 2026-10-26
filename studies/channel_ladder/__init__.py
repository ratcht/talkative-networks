"""
The channel ladder: how much each step up in a channel's complexity adds,
with no spatial assumptions (docs/studies/channel-ladder.md). This module
holds the study's architectural choices, built from the parts in testbed/.

Each level is a pair of channel parts from testbed/communication:

  level      encoder              decoder              adds
  1          linear_taper         linear_taper         a linear message and a linear injection
  2-encoder  linear_taper_gelu    linear_taper         a GELU in the encoder
  2-decoder  linear_taper         linear_taper_gelu    a GELU in the decoder
  3          linear_taper_gelu    linear_taper_gelu    a GELU in both

Level 0 is bagging, which is every run's step 0, so it has no entry. Any
level runs with any of the study's mixers (MIXERS): attention, the channel's
own, or one of its two controls, self and mean. The baseline that weighs the
experts' answers instead of communicating is the soft-gated MoE with GATE
(testbed/moe.py). Every part reads or writes through WIDTH, the r of the
linear taper parts and of the gate.
"""

from dataclasses import dataclass

from testbed.communication import ChannelConfig

STUDY = "channel-ladder"


@dataclass(frozen=True)
class Level:
  encoder: str
  decoder: str


LEVELS: dict[str, Level] = {
  "1": Level("linear_taper", "linear_taper"),
  "2-encoder": Level("linear_taper_gelu", "linear_taper"),
  "2-decoder": Level("linear_taper", "linear_taper_gelu"),
  "3": Level("linear_taper_gelu", "linear_taper_gelu"),
}
MIXERS = ("attention", "self", "mean")  # the channel's, then its controls
GATE = "linear_taper_gelu"
WIDTH = 8


def at_level(level: str, **settings) -> ChannelConfig:
  """The channel at a level, at the study's width. `settings` are the
  config's other fields, such as the mixer, and may override the width."""
  parts = LEVELS[level]
  return ChannelConfig(
    encoder=parts.encoder, decoder=parts.decoder, **{"width": WIDTH, **settings}
  )
