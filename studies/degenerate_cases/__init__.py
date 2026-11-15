"""
Degenerate cases: how the communicating ensemble behaves when its expert group
isn't the standard one (docs/studies/degenerate-cases.md). This module holds
the study's choices, built from the parts in testbed/.

Every arm is the channel ladder's level 1 at its width, at the pairs and with
the mixers the ladder chose, so that it compares with the ladder's arms trial
seed by trial seed. Only the group changes:

  case                       the group                                    epochs
  clones                     three copies of the ladder's first expert    30
  two-trained-one-untrained  the ladder's group, third expert untrained   60
  negative-correlation       three experts trained to disagree, at a λ    30
  untrained                  three experts with random weights            150

A run logs every epoch and the joint learning rate is constant, so a long run
holds the shorter one's epochs exactly: untrained is read at 30 epochs as well
as at 150, and two-trained-one-untrained at 30 as well as at 60.
"""

from dataclasses import dataclass

from studies.channel_ladder import GATE, LEVELS, MIXERS, WIDTH

__all__ = [
  "AGREEMENT_BELOW",
  "BAGGING_WITHIN",
  "CASES",
  "DIAL",
  "EXCHANGES",
  "GATE",
  "LAMS",
  "LEVEL",
  "MIXERS",
  "MOE_READ",
  "NEGATIVE_CORRELATION",
  "STUDY",
  "WIDTH",
  "Case",
  "parts",
]

STUDY = "degenerate-cases"

LEVEL = "1"
EXCHANGES = {"forward": ("stage2", "stage3"), "backward": ("stage3", "stage1")}
MOE_READ = "stage3"
NEGATIVE_CORRELATION = "negative-correlation"


@dataclass(frozen=True)
class Case:
  flags: tuple[str, ...]  # the testbed.run flags that make the group
  epochs: int


CASES: dict[str, Case] = {
  "clones": Case(("--selector", "clones_of_first"), 30),
  "two-trained-one-untrained": Case(("--untrained-members", "1"), 60),
  NEGATIVE_CORRELATION: Case(("--scheme", "negative_correlation"), 30),
  "untrained": Case(("--scheme", "untrained"), 150),
}

# Which λ run through the channel: what dial.py picks by the rule below, on
# joint_val, before any arm runs. 0 is the control.
LAMS = (0.0, 2.5, 4.0)
DIAL = (0.0, 2.0, 2.5, 3.0, 4.0, 5.0)
# the middle λ: the smallest with agreement below this on every trial seed
AGREEMENT_BELOW = 0.5
# the far λ: the largest whose mean bagging is this close to λ = 0's
BAGGING_WITHIN = 0.015


def parts() -> tuple[str, str]:
  """The study's encoder and decoder."""
  level = LEVELS[LEVEL]
  return level.encoder, level.decoder
