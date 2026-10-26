"""
Run one arm of a study over a list of trial seeds.

  uv run python -m testbed.run --arm bagging  --trial-seeds 0 1 2
  uv run python -m testbed.run --arm forward  --exchange stage1 stage2 --trial-seeds 0 1 2
  uv run python -m testbed.run --arm forward-self --exchange stage1 stage2 --mixer self --trial-seeds 0 1 2
  uv run python -m testbed.run --arm backward --exchange stage2 stage1 --rounds 2 --trial-seeds 0 1 2
  uv run python -m testbed.run --arm moe --gate linear_taper_gelu --gate-read stage3 --trial-seeds 0 1 2

An arm is bagging (no exchange), a communicating ensemble (--exchange), or the
soft-gated MoE baseline (--gate, testbed/moe.py), which weighs the experts'
answers instead.

This runner knows no study. A study's own runner (studies/) passes it the
parts to use and labels the runs with --tag. Flags must be spelled out in
full, so that no abbreviation is taken for a flag a study's runner sets.

A trial seed is the parent of every seed in one trial (seeds.py): the experts,
the group, the channel's initialisation and its joint training. For each one,
the pool comes from the expert cache, or is trained into it. The group is
selected, then the channel or gate is trained, and the run is boiled down to
one number (results.py). Arms run on the same trial seeds get the same
experts, so they compare trial by trial.
"""

import argparse
from dataclasses import asdict
from statistics import mean

from testbed.communication import (
  DECODERS,
  ENCODERS,
  MIXERS,
  ChannelConfig,
  Exchange,
  build_ensemble,
)
from testbed.data import DATASETS, load
from testbed.metrics import description
from testbed.moe import GATES, GateConfig, build_gated
from testbed.experts import (
  ARCHITECTURES,
  Budget,
  ClassSkew,
  IndependentScheme,
  NegativeCorrelation,
  Redundant,
  Scheme,
  SeedOnly,
  get_pool,
)
from testbed.results import epochs_around_best, median_around_best
from testbed.seeds import derive
from testbed.selection import SELECTORS
from testbed.tracking import MlflowTracker, NullTracker
from testbed.train import JointConfig, train

SCHEMES = {
  "seed_only": lambda args: SeedOnly(),
  "class_skew": lambda args: ClassSkew(args.alpha),
  "negative_correlation": lambda args: NegativeCorrelation(args.lam),
}


def make_scheme(args: argparse.Namespace) -> Scheme:
  scheme = SCHEMES[args.scheme](args)
  if args.copies == 1:
    return scheme
  if not isinstance(scheme, IndependentScheme):
    raise SystemExit(
      f"--copies needs an independently trained scheme, not {args.scheme}"
    )
  return Redundant(scheme, args.copies)


def check_flags(args: argparse.Namespace) -> None:
  """Combinations of flags that would build something other than asked."""
  if args.gate and args.exchange:
    raise SystemExit(
      "--gate weighs the experts instead of exchanges; give one or the other"
    )
  if args.gate and args.loss == "per_expert":
    raise SystemExit(
      "--loss per_expert trains no gate: no expert's answer depends on it"
    )
  if args.gate_read and not args.gate:
    raise SystemExit("--gate-read needs a --gate")


def extra_tags(args: argparse.Namespace) -> dict[str, str]:
  """The --tag KEY=VALUE flags as a dict."""
  tags = {}
  for tag in args.tag:
    key, sep, value = tag.partition("=")
    if not (key and sep):
      raise SystemExit(f"--tag takes KEY=VALUE, not {tag!r}")
    tags[key] = value
  return tags


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False,
  )
  p.add_argument("--arm", required=True, help="label for this arm, e.g. forward")
  p.add_argument("--trial-seeds", type=int, nargs="+", required=True)

  data = p.add_argument_group("data")
  data.add_argument("--dataset", choices=DATASETS, default="fashion")
  data.add_argument("--split-seed", type=int, default=0, help="pinned while iterating")
  data.add_argument("--device", default="cuda")

  experts = p.add_argument_group("experts")
  experts.add_argument("--arch", choices=ARCHITECTURES, default="cnn3")
  experts.add_argument("--scheme", choices=SCHEMES, default="class_skew")
  experts.add_argument(
    "--alpha", type=float, default=1.0, help="class_skew: Dirichlet concentration"
  )
  experts.add_argument(
    "--lam", type=float, default=0.5, help="negative_correlation: penalty weight"
  )
  experts.add_argument(
    "--copies",
    type=int,
    default=1,
    help="> 1 makes experts redundant in groups of this size",
  )
  experts.add_argument("--expert-epochs", type=int, default=10)
  experts.add_argument("--experts", type=int, default=3, help="M, the group size")
  experts.add_argument(
    "--pool-size",
    type=int,
    default=None,
    help="defaults to M: a fresh pool per trial seed",
  )
  experts.add_argument(
    "--selector",
    choices=SELECTORS,
    default=None,
    help="defaults to first if the pool is the group",
  )

  comm = p.add_argument_group("communication")
  comm.add_argument(
    "--exchange", nargs=2, action="append", default=[], metavar=("READ", "WRITE")
  )
  comm.add_argument("--rounds", type=int, default=1, help="backward only")
  comm.add_argument("--encoder", choices=ENCODERS, default=ChannelConfig.encoder)
  comm.add_argument(
    "--mixer",
    choices=MIXERS,
    default=ChannelConfig.mixer,
    help="self and mean are the controls: each expert hears only itself, or "
    "the plain average of every expert",
  )
  comm.add_argument("--decoder", choices=DECODERS, default=ChannelConfig.decoder)
  comm.add_argument("--key-dim", type=int, default=ChannelConfig.key_dim)
  comm.add_argument("--value-dim", type=int, default=ChannelConfig.value_dim)
  comm.add_argument(
    "--width",
    type=int,
    default=ChannelConfig.width,
    help="r, the narrow middle of the linear taper parts, and the readings a "
    "gate takes of each expert",
  )
  comm.add_argument(
    "--gate",
    choices=GATES,
    help="instead of exchanges, the soft-gated MoE baseline: a gate of this "
    "architecture weighs the experts' answers (docs/testbed/moe.md)",
  )
  comm.add_argument(
    "--gate-read", help="the point the gate reads; the experts' last point by default"
  )

  joint = p.add_argument_group("joint training")
  joint.add_argument("--epochs", type=int, default=10)
  joint.add_argument("--batch-size", type=int, default=128)
  joint.add_argument("--lr", type=float, default=1e-3)
  joint.add_argument("--loss", choices=("ensemble", "per_expert"), default="ensemble")

  tracking = p.add_argument_group("tracking")
  tracking.add_argument("--experiment", default="v3")
  tracking.add_argument("--no-track", action="store_true")
  tracking.add_argument(
    "--tag",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="an extra MLflow tag on every run, e.g. a study's label for it",
  )
  return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  check_flags(args)  # before loading anything, so bad flags fail fast
  tags = extra_tags(args)
  exchanges = [Exchange(read, write) for read, write in args.exchange]
  gate = (
    GateConfig(architecture=args.gate, read=args.gate_read, width=args.width)
    if args.gate
    else None
  )
  channel = ChannelConfig(
    encoder=args.encoder,
    mixer=args.mixer,
    decoder=args.decoder,
    key_dim=args.key_dim,
    value_dim=args.value_dim,
    width=args.width,
  )
  data = load(args.dataset, split_seed=args.split_seed, device=args.device)
  scheme = make_scheme(args)
  budget = Budget(epochs=args.expert_epochs)
  pool_size = args.pool_size or args.experts
  selector = SELECTORS[
    args.selector or ("first" if pool_size == args.experts else "random")
  ]()
  joint = JointConfig(args.epochs, args.batch_size, args.lr, args.loss)

  accuracies, gains = [], []
  for trial_seed in args.trial_seeds:
    pool = get_pool(
      scheme, args.arch, data, pool_size, derive(trial_seed, "pool"), budget
    )
    group = selector.select(pool, args.experts, derive(trial_seed, "group"))
    experts = [pool.experts[i] for i in group]
    if gate:
      model = build_gated(
        [e.model for e in experts], gate, data.input_shape, derive(trial_seed, "gate")
      )
    else:
      model = build_ensemble(
        [e.model for e in experts],
        exchanges,
        channel,
        data.input_shape,
        args.rounds,
        derive(trial_seed, "channel"),
      )

    tracker = (
      NullTracker()
      if args.no_track
      else MlflowTracker(
        args.experiment, f"{args.arm}-trial{trial_seed}", description()
      )
    )
    with tracker:
      tracker.set_tags(  # the runner's own tags win a clash with --tag
        tags
        | {"arm": args.arm, "trial_seed": trial_seed, "direction": model.direction}
        | ({"gate": gate.architecture} if gate else {})
      )
      tracker.log_params(
        {
          "data": {
            "name": data.name,
            "split_seed": args.split_seed,
            "digest": data.digest,
          },
          "experts": {
            "arch": args.arch,
            "scheme": scheme.config(),
            "budget": asdict(budget),
            "M": args.experts,
          },
          "pool": {
            "key": pool.key,
            "digest": pool.digest,
            "size": pool_size,
            "group": group,
          },
          "exchanges": [(ex.read, ex.write) for ex in exchanges],
          "rounds": args.rounds,
          **({"channel": asdict(channel)} if exchanges else {}),
          **({"gate": asdict(gate) | {"read": model.read}} if gate else {}),
          "joint": asdict(joint),
          **{f"expert{i}": e.profile for i, e in enumerate(experts) if e.profile},
        }
      )
      history = train(model, data, joint, derive(trial_seed, "joint"), tracker)

    # the run as one number: the median over the 5 epochs around its best
    # joint_val epoch (results.py). Step 0 is bagging.
    epochs = epochs_around_best(history)
    accuracy = median_around_best(history)
    gain = median_around_best(history, "test/joint_answer_shift/net")
    accuracies.append(accuracy)
    gains.append(gain)
    print(
      f"[{args.arm}] trial seed {trial_seed}: pool {pool.key}  bagging {history[0]['test/accuracy']:.4f}"
      f"  ensemble {accuracy:.4f}  gain {gain:+.4f}  epochs {epochs.start}-{epochs.stop - 1}"
    )
  print(
    f"[{args.arm}] mean over {len(gains)} trial seeds: ensemble {mean(accuracies):.4f}"
    f"  gain over bagging {mean(gains):+.4f}"
  )


if __name__ == "__main__":
  main()
