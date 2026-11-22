"""
A classifier on the experts' final streams: on one expert's, and on all of
them together.

  uv run python -m studies.pacs.readout --arm readout --arch stream64 [testbed.run's data flags]

Takes testbed.run's flags. The experts come from the expert cache. Joint
training trains only the classifier, so each run is logged and read like a
channel's. Every trial makes one run per expert and one for all of them, with
a linear classifier and with an MLP.
"""

from dataclasses import asdict

import torch
import torch.nn as nn

from testbed.communication.ensemble import Output, average
from testbed.data import load
from testbed.experts import Budget, Expert, PerDataset, get_pool
from testbed.metrics import description
from testbed.results import median_around_best
from testbed.run import extra_tags, parse_args
from testbed.seeds import derive, seeded
from testbed.tracking import MlflowTracker, NullTracker
from testbed.train import JointConfig, train

HIDDEN = {"linear": None, "mlp": 256}


class Readout(nn.Module):
  """Frozen experts and a classifier on their streams, in the form joint
  training takes."""

  def __init__(self, experts: list[Expert], head: nn.Module):
    super().__init__()
    self.experts = nn.ModuleList(experts)
    self.head = head

  @property
  def trainable(self) -> nn.Module:
    return self.head

  def forward(self, x: torch.Tensor, answer_each_round: bool = False) -> Output:
    with torch.no_grad():
      streams = [e.run(x, until=e.points[-1]).mean((2, 3)) for e in self.experts]
    logits = self.head(torch.cat(streams, dim=1))[:, None]  # (B, 1, C)
    return Output(average(logits), logits, [])

  def bagging(self, x: torch.Tensor) -> Output:
    return self(x)


def build_head(inputs: int, classes: int, hidden: int | None) -> nn.Module:
  if hidden is None:
    # a Sequential, as the optimisation meter needs dotted weight names
    return nn.Sequential(nn.Linear(inputs, classes))
  return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, classes))


def main() -> None:
  args = parse_args()
  torch.backends.cudnn.deterministic = True
  data = load(
    args.expert_dataset,
    args.joint_dataset,
    args.test_dataset,
    args.expert_samples,
    args.joint_samples,
    split_seed=args.split_seed,
    device=args.device,
  )
  budget = Budget(epochs=args.expert_epochs)
  joint = JointConfig(args.epochs, args.batch_size, args.lr, args.loss)
  reads = {str(i): [i] for i in range(args.experts)} | {
    "all": list(range(args.experts))
  }

  for trial_seed in args.trial_seeds:
    pool = get_pool(
      PerDataset(), args.arch, data, args.experts, derive(trial_seed, "pool"), budget
    )
    experts = [e.model for e in pool.experts]
    point = experts[0].points[-1]
    width = experts[0].point_shapes(data.input_shape)[point][0]
    for kind, hidden in HIDDEN.items():
      for name, group in reads.items():
        with seeded(derive(trial_seed, "readout")):
          head = build_head(width * len(group), data.num_classes, hidden)
        model = Readout([experts[i] for i in group], head.to(data.device))
        tracker = (
          NullTracker()
          if args.no_track
          else MlflowTracker(
            args.experiment,
            f"{args.arm}-{kind}-{name}-trial{trial_seed}",
            description(),
          )
        )
        with tracker:
          tracker.set_tags(
            extra_tags(args)
            | {"arm": args.arm, "trial_seed": trial_seed, "head": kind, "reads": name}
          )
          tracker.log_params(
            {
              "data": {
                "expert": args.expert_dataset,
                "joint": args.joint_dataset,
                "test": args.test_dataset,
                "digest": data.digest,
              },
              "experts": {"arch": args.arch, "budget": asdict(budget), "M": len(group)},
              "pool": {"key": pool.key, "digest": pool.digest, "group": group},
              "weights": {"trained": sum(p.numel() for p in head.parameters())},
              "joint": asdict(joint),
            }
          )
          history = train(model, data, joint, derive(trial_seed, "joint"), tracker)
        print(
          f"[{args.arm}] trial seed {trial_seed}: {kind} on {name}"
          f"  {median_around_best(history):.4f}",
          flush=True,
        )


if __name__ == "__main__":
  main()
