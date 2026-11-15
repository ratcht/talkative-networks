"""
Run one arm of the channel ladder over a list of trial seeds. An arm is one
of three kinds:

  --level L --exchange READ WRITE   a level's channel, with --mixer for a control
  --moe READ                        the soft-gated MoE baseline, its gate at READ
  --bagging                         the experts with equal weights

  uv run python -m studies.channel_ladder.run --phase 2 --level 2-encoder --arm forward-2e \\
    --exchange stage2 stage3 --trial-seeds 0 1 2
  uv run python -m studies.channel_ladder.run --phase 2 --level 1 --mixer self \\
    --arm forward-1-self --exchange stage2 stage3 --trial-seeds 0 1 2
  uv run python -m studies.channel_ladder.run --phase 2 --moe stage3 --arm moe-stage3 \\
    --trial-seeds 0 1 2

studies/channel_ladder/phase4.py prints this command for every arm of phase 4.
Every run belongs to a phase of the study (--phase), and each phase is its own
MLflow experiment, channel-ladder-phase<PHASE>, so no experiment piles up
every arm the study ever ran.

The runner hands testbed.run the study's choices (studies/channel_ladder):
the level's parts or the study's gate, its mixer, and its width. Every run
gets the study's defaults (DEFAULTS) and is tagged with the study, its phase,
and its level or which baseline it is. Every other flag is testbed.run's,
and a flag given here overrides a default.
"""

import argparse
import sys

from studies.channel_ladder import COSINE, GATE, LEVELS, MIXERS, STUDY, WIDTH
from testbed import run

TRIAL_SEEDS = range(15)  # every arm runs on the same list, so arms pair up
DEFAULTS = [
  *("--epochs", "30", "--width", str(WIDTH)),
  *("--trial-seeds", *map(str, TRIAL_SEEDS)),
]
# set by the arm, so a caller can't give them as well
ARM_FLAGS = {"--exchange", "--encoder", "--decoder", "--gate", "--gate-read"}


def experiment(phase: str) -> str:
  """The MLflow experiment a phase's runs go to."""
  return f"{STUDY}-phase{phase}"


def flags_for_testbed(argv: list[str]) -> list[str]:
  """This runner's flags -> testbed.run's."""
  p = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False,
  )
  kind = p.add_mutually_exclusive_group(required=True)
  kind.add_argument("--level", choices=LEVELS)
  kind.add_argument("--moe", metavar="READ", help="the point the gate reads")
  kind.add_argument("--bagging", action="store_true")
  p.add_argument(
    "--exchange",
    nargs=2,
    action="append",
    metavar=("READ", "WRITE"),
    help="a level's one exchange",
  )
  p.add_argument(
    "--mixer", choices=(*MIXERS, COSINE), help=f"a level's; {MIXERS[0]} by default"
  )
  p.add_argument(
    "--phase",
    required=True,
    help="the phase the runs belong to, e.g. 1 or 2a; it names their MLflow experiment",
  )
  args, rest = p.parse_known_args(argv)
  given = {flag.partition("=")[0] for flag in rest}
  if given & ARM_FLAGS:
    raise SystemExit(
      f"the arm sets {', '.join(sorted(given & ARM_FLAGS))}; don't give it as well"
    )
  if "--experiment" in given:
    raise SystemExit("--phase names the experiment; don't give --experiment as well")
  study = [
    *DEFAULTS,
    *rest,
    *("--experiment", experiment(args.phase)),
    *("--tag", f"study={STUDY}", "--tag", f"phase={args.phase}"),
  ]

  if args.level is None:
    if args.exchange or args.mixer:
      raise SystemExit("--exchange and --mixer belong to a --level, not a baseline")
    if args.moe:
      return [*study, "--gate", GATE, "--gate-read", args.moe, "--tag", "baseline=moe"]
    return [*study, "--tag", "baseline=bagging"]

  if not args.exchange:
    raise SystemExit(f"--level {args.level} needs an --exchange to put the channel in")
  if len(args.exchange) > 1:
    raise SystemExit("an arm of the ladder has one exchange")
  level = LEVELS[args.level]
  return [
    *study,
    *("--exchange", *args.exchange[0], "--mixer", args.mixer or MIXERS[0]),
    *("--encoder", level.encoder, "--decoder", level.decoder),
    *("--tag", f"level={args.level}"),
  ]


def main(argv: list[str] | None = None) -> None:
  run.main(flags_for_testbed(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
  main()
