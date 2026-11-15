"""
Print the command for every arm of the study, one per line, in the order to
run them: the cheapest and most decisive case first. Any flags given are added
to every line.

  uv run python -m studies.degenerate_cases.plan > arms.txt
  bash arms.txt
  uv run python -m studies.degenerate_cases.plan --device cuda:1 | awk 'NR % 2 == 0' | bash
"""

import shlex
import sys

from studies.degenerate_cases import (
  CASES,
  EXCHANGES,
  LAMS,
  MIXERS,
  NEGATIVE_CORRELATION,
)

COMMAND = "uv run python -m studies.degenerate_cases.run"


def arms() -> list[list[str]]:
  """Every arm of the study, as studies.degenerate_cases.run's flags."""
  plan = []
  for case in CASES:
    settings = (
      [["--case", case, "--lam", f"{lam:g}"] for lam in LAMS]
      if case == NEGATIVE_CORRELATION
      else [["--case", case]]
    )
    for setting in settings:
      plan += [
        [*setting, "--direction", direction, "--mixer", mixer]
        for direction in EXCHANGES
        for mixer in MIXERS
      ]
      if case == NEGATIVE_CORRELATION:
        plan.append([*setting, "--moe"])
  return plan


def main(argv: list[str] | None = None) -> None:
  extra = sys.argv[1:] if argv is None else argv
  for arm in arms():
    print(COMMAND, shlex.join([*arm, *extra]))


if __name__ == "__main__":
  main()
