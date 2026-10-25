"""
Run one arm of a study over a list of trial seeds.

  uv run python -m testbed.run --arm bagging  --trial-seeds 0 1 2
  uv run python -m testbed.run --arm forward  --exchange stage1 stage2 --trial-seeds 0 1 2
  uv run python -m testbed.run --arm backward --exchange stage2 stage1 --rounds 2 --trial-seeds 0 1 2

A trial seed is the parent of every seed in one trial (seeds.py): the experts,
the group, the channel's initialisation and its joint training. For each one,
the pool comes from the expert cache, or is trained into it. The group is
selected, then the channel is trained and scored. Arms run on the same trial
seeds get the same experts, so they compare trial by trial.
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


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
  comm.add_argument("--encoder", choices=ENCODERS, default="pooled")
  comm.add_argument("--mixer", choices=MIXERS, default="attention")
  comm.add_argument("--decoder", choices=DECODERS, default="broadcast")
  comm.add_argument("--key-dim", type=int, default=16)
  comm.add_argument("--value-dim", type=int, default=64)

  joint = p.add_argument_group("joint training")
  joint.add_argument("--epochs", type=int, default=10)
  joint.add_argument("--batch-size", type=int, default=128)
  joint.add_argument("--lr", type=float, default=1e-3)
  joint.add_argument("--loss", choices=("ensemble", "per_expert"), default="ensemble")

  tracking = p.add_argument_group("tracking")
  tracking.add_argument("--experiment", default="v3")
  tracking.add_argument("--no-track", action="store_true")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  data = load(args.dataset, split_seed=args.split_seed, device=args.device)
  scheme = make_scheme(args)
  budget = Budget(epochs=args.expert_epochs)
  pool_size = args.pool_size or args.experts
  selector = SELECTORS[
    args.selector or ("first" if pool_size == args.experts else "random")
  ]()
  exchanges = [Exchange(read, write) for read, write in args.exchange]
  channel = ChannelConfig(
    args.encoder, args.mixer, args.decoder, args.key_dim, args.value_dim
  )
  joint = JointConfig(args.epochs, args.batch_size, args.lr, args.loss)

  gains = []
  for trial_seed in args.trial_seeds:
    pool = get_pool(
      scheme, args.arch, data, pool_size, derive(trial_seed, "pool"), budget
    )
    group = selector.select(pool, args.experts, derive(trial_seed, "group"))
    experts = [pool.experts[i] for i in group]
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
      tracker.set_tags(
        {"arm": args.arm, "trial_seed": trial_seed, "direction": model.direction}
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
          "channel": asdict(channel),
          "joint": asdict(joint),
          **{f"expert{i}": e.profile for i, e in enumerate(experts) if e.profile},
        }
      )
      history = train(model, data, joint, derive(trial_seed, "joint"), tracker)

    # a quick look only, at the step with the best joint_val accuracy; how runs
    # are scored is still open (docs/docket.md). Step 0 is bagging.
    step = max(range(len(history)), key=lambda s: history[s]["joint_val/accuracy"])
    gain = history[step]["test/joint_answer_shift/net"]
    gains.append(gain)
    print(
      f"[{args.arm}] trial seed {trial_seed}: pool {pool.key}  bagging {history[0]['test/accuracy']:.4f}"
      f"  ensemble {history[step]['test/accuracy']:.4f}  gain {gain:+.4f}  at step {step}"
    )
  print(
    f"[{args.arm}] mean gain over bagging across {len(gains)} trial seeds: {mean(gains):+.4f}"
  )


if __name__ == "__main__":
  main()
