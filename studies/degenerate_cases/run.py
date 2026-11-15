"""
Run one arm of the degenerate cases over a list of trial seeds. An arm is a
case (--case) and one of two kinds:

  --direction D --mixer M   level 1 at the direction's chosen pair
  --moe                     the soft-gated MoE baseline, negative correlation only

  uv run python -m studies.degenerate_cases.run --case clones \\
    --direction forward --mixer self
  uv run python -m studies.degenerate_cases.run --case negative-correlation --lam 2.5 \\
    --direction backward --mixer mean
  uv run python -m studies.degenerate_cases.run --case negative-correlation --lam 2.5 --moe

studies/degenerate_cases/plan.py prints this command for every arm of the
study. Each case is its own MLflow experiment, degenerate-cases-<CASE>.

The runner hands testbed.run the study's choices (studies/degenerate_cases):
what makes the case's group, level 1's parts, the pair, and the epochs. Runs
are tagged with the study, the case, and the level or baseline, and named for
the arm. Every other flag is testbed.run's, and a flag given here overrides a
default, as --device, --trial-seeds and --epochs do.
"""

import argparse
import sys

from studies.channel_ladder.run import TRIAL_SEEDS
from studies.degenerate_cases import (
  CASES,
  EXCHANGES,
  GATE,
  LEVEL,
  MIXERS,
  MOE_READ,
  NEGATIVE_CORRELATION,
  STUDY,
  WIDTH,
  parts,
)
from testbed import run

# set by the case or the arm, so a caller can't give them as well
ARM_FLAGS = {
  "--arm",
  "--decoder",
  "--encoder",
  "--exchange",
  "--experiment",
  "--gate",
  "--gate-read",
  "--scheme",
  "--selector",
  "--untrained-members",
}


def experiment(case: str) -> str:
  """The MLflow experiment a case's runs go to."""
  return f"{STUDY}-{case}"


def flags_for_testbed(argv: list[str]) -> list[str]:
  """This runner's flags -> testbed.run's."""
  p = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False,
  )
  p.add_argument("--case", choices=CASES, required=True)
  p.add_argument("--lam", type=float, help=f"{NEGATIVE_CORRELATION} only, and needed")
  kind = p.add_mutually_exclusive_group(required=True)
  kind.add_argument("--direction", choices=EXCHANGES)
  kind.add_argument("--moe", action="store_true", help=f"gate reading {MOE_READ}")
  p.add_argument("--mixer", choices=MIXERS, help=f"{MIXERS[0]} by default")
  args, rest = p.parse_known_args(argv)
  given = {flag.partition("=")[0] for flag in rest}
  if given & ARM_FLAGS:
    raise SystemExit(
      f"the arm sets {', '.join(sorted(given & ARM_FLAGS))}; don't give it as well"
    )
  # testbed's default λ does almost nothing, so a run never falls back on it
  if (args.case == NEGATIVE_CORRELATION) != (args.lam is not None):
    raise SystemExit(f"--lam is for --case {NEGATIVE_CORRELATION}, and it needs one")
  if args.moe and args.case != NEGATIVE_CORRELATION:
    raise SystemExit(f"--moe is for --case {NEGATIVE_CORRELATION}")
  if args.moe and args.mixer:
    raise SystemExit("--mixer belongs to a --direction, not the MoE")

  case = CASES[args.case]
  name = args.case if args.lam is None else f"{args.case}-lam{args.lam:g}"
  study = [
    *("--epochs", str(case.epochs), "--width", str(WIDTH)),
    *("--trial-seeds", *map(str, TRIAL_SEEDS)),
    *rest,
    *case.flags,
    *("--experiment", experiment(args.case)),
    *("--tag", f"study={STUDY}", "--tag", f"case={args.case}"),
  ]
  if args.lam is not None:
    study += ["--lam", str(args.lam), "--tag", f"lam={args.lam:g}"]

  if args.moe:
    return [
      *study,
      *("--arm", f"{name}-moe-{MOE_READ}"),
      *("--gate", GATE, "--gate-read", MOE_READ, "--tag", "baseline=moe"),
    ]
  mixer = args.mixer or MIXERS[0]
  encoder, decoder = parts()
  return [
    *study,
    *("--arm", f"{name}-{args.direction}-{mixer}"),
    *("--exchange", *EXCHANGES[args.direction], "--mixer", mixer),
    *("--encoder", encoder, "--decoder", decoder),
    *("--tag", f"level={LEVEL}"),
  ]


def main(argv: list[str] | None = None) -> None:
  run.main(flags_for_testbed(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
  main()
