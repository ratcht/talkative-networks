"""
Run one arm of the model-grafting study over a list of trial seeds, through
testbed.run. Every other flag goes to testbed.run as it is.

  uv run python -m studies.model_grafting.run --graft-arm graft_attention --trial-seeds 0 1 2
  uv run python -m studies.model_grafting.run --graft-arm bypass --arch vit \\
      --graft-read block2 --graft-write block10 --trial-seeds 0 1 2

Runs go to the MLflow experiment "model-grafting" unless --experiment says
otherwise, tagged study=model-grafting and graft_arm=<arm>, and are labelled
<arm>-<read>-<write>.
"""

import argparse

from studies.model_grafting import ARM_FLAGS, ARMS, EXPERIMENT, STUDY
from testbed import run as testbed_run
from testbed.grafting import GraftConfig


def flags_for_testbed(argv: list[str] | None = None) -> list[str]:
  p = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
  p.add_argument("--graft-arm", choices=ARMS, required=True)
  p.add_argument("--graft-read", default=GraftConfig.read)
  p.add_argument("--graft-write", default=GraftConfig.write)
  args, rest = p.parse_known_args(argv)
  given = [flag for flag in rest if flag.split("=")[0] in ARM_FLAGS]
  if given:
    raise SystemExit(f"{', '.join(given)}: set by --graft-arm, not given")
  label = f"{args.graft_arm}-{args.graft_read}-{args.graft_write}"
  experiment = [] if "--experiment" in rest else ["--experiment", EXPERIMENT]
  return [
    "--arm",
    label,
    *ARMS[args.graft_arm],
    "--graft-read",
    args.graft_read,
    "--graft-write",
    args.graft_write,
    *experiment,
    "--tag",
    f"study={STUDY}",
    "--tag",
    f"graft_arm={args.graft_arm}",
    *rest,
  ]


def main(argv: list[str] | None = None) -> None:
  testbed_run.main(flags_for_testbed(argv))


if __name__ == "__main__":
  main()
