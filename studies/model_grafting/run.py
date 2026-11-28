"""
Run one arm of the model-grafting study over a list of trial seeds, through
testbed.run. Every other flag goes to testbed.run as it is.

  uv run python -m studies.model_grafting.run --graft-arm graft_attention --trial-seeds 0 1 2
  uv run python -m studies.model_grafting.run --graft-arm bypass --arch vit \\
      --graft-read block2 --graft-write block10 --trial-seeds 0 1 2

Runs go to the MLflow experiment "model-grafting" unless --experiment says
otherwise, tagged study=model-grafting, graft_arm=<arm> and stitch_fit=<fit>,
and are labelled <arm>-<read>-<write>, with -<fit> for a fitted stitch.

--stitch-fit fitted or fitted_frozen fits the stitches by least squares before
training (studies/model_grafting/stitch_fit.py); it needs, and sets, the plain
stitch.
"""

import argparse

from studies.model_grafting import ARM_FLAGS, ARMS, EXPERIMENT, STUDY
from studies.model_grafting.stitch_fit import FITS, fit_stitches
from testbed import run as testbed_run
from testbed.grafting import GraftConfig

PLAIN = "pointwise_linear"


def _parse(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
  p = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
  p.add_argument("--graft-arm", choices=ARMS, required=True)
  p.add_argument("--graft-read", default=GraftConfig.read)
  p.add_argument("--graft-write", default=GraftConfig.write)
  p.add_argument("--graft-stitch")
  p.add_argument("--stitch-fit", choices=FITS, default="random")
  return p.parse_known_args(argv)


def flags_for_testbed(argv: list[str] | None = None) -> list[str]:
  args, rest = _parse(argv)
  given = [flag for flag in rest if flag.split("=")[0] in ARM_FLAGS]
  if given:
    raise SystemExit(f"{', '.join(given)}: set by --graft-arm, not given")
  stitch = args.graft_stitch
  if args.stitch_fit != "random":
    if stitch not in (None, PLAIN):
      raise SystemExit(f"--stitch-fit {args.stitch_fit} fits the plain stitch, not {stitch}")
    stitch = PLAIN
  label = f"{args.graft_arm}-{args.graft_read}-{args.graft_write}"
  if args.stitch_fit != "random":
    label += f"-{args.stitch_fit}"
  experiment = [] if "--experiment" in rest else ["--experiment", EXPERIMENT]
  return [
    "--arm",
    label,
    *ARMS[args.graft_arm],
    "--graft-read",
    args.graft_read,
    "--graft-write",
    args.graft_write,
    *(["--graft-stitch", stitch] if stitch else []),
    *experiment,
    "--tag",
    f"study={STUDY}",
    "--tag",
    f"graft_arm={args.graft_arm}",
    "--tag",
    f"stitch_fit={args.stitch_fit}",
    *rest,
  ]


def main(argv: list[str] | None = None) -> None:
  flags = flags_for_testbed(argv)
  fit = _parse(argv)[0].stitch_fit
  adjust = None
  if fit != "random":

    def adjust(model, data) -> None:
      explained = fit_stitches(model, data, freeze=fit == "fitted_frozen")
      print(
        f"[stitch fit] {fit}: share of each obligate's stream explained "
        f"{[round(e, 4) for e in explained]}",
        flush=True,
      )

  testbed_run.main(flags, adjust=adjust)


if __name__ == "__main__":
  main()
