"""
Print the command for every arm of phase 4, one per line: the four forward
levels with attention, and levels 1 and 2-encoder with COSINE, its queries and
keys normalised (docs/studies/channel-ladder.md). Any flags given are added to
every line.

  uv run python -m studies.channel_ladder.phase4 > arms.txt
  bash arms.txt
"""

import shlex
import sys

from studies.channel_ladder import COSINE, LEVELS, MIXERS

PHASE = "4"
EXCHANGE = ("stage2", "stage3")  # the forward pair phase 1 picked
COSINE_LEVELS = ("1", "2-encoder")
COMMAND = "uv run python -m studies.channel_ladder.run"


def arm(level: str, mixer: str) -> list[str]:
  """One arm as studies.channel_ladder.run's flags. Attention's arms keep the
  names they have in phases 1 and 2a."""
  label = "-".join(
    [f"level{level}", *([] if mixer == MIXERS[0] else [mixer]), *EXCHANGE]
  )
  return [
    *("--phase", PHASE, "--level", level, "--mixer", mixer),
    *("--arm", label, "--exchange", *EXCHANGE),
  ]


def arms() -> list[list[str]]:
  """Every arm of the phase, the new ones first."""
  return [
    *(arm(level, COSINE) for level in COSINE_LEVELS),
    *(arm(level, MIXERS[0]) for level in LEVELS),
  ]


def main(argv: list[str] | None = None) -> None:
  extra = sys.argv[1:] if argv is None else argv
  for flags in arms():
    print(COMMAND, shlex.join([*flags, *extra]))


if __name__ == "__main__":
  main()
