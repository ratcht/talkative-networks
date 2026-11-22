"""
One pooled model, trained further on the joint images: a model given the
images the experts were trained on, then the images a channel is trained on.

  uv run python -m studies.multiple_datasets.pooled --arm pooled --arch cnn3_large [testbed.run's data flags]

Takes testbed.run's flags. The model comes from the expert cache as a
seed_only pool of one. Joint training then trains the model's own weights, so
the run is logged and read like a channel's, and step 0 is the model before
the joint images.
"""

from dataclasses import asdict
from statistics import mean

import torch
import torch.nn as nn

from testbed.communication.ensemble import Output, average
from testbed.data import load
from testbed.experts import Budget, Expert, SeedOnly, get_pool
from testbed.metrics import description
from testbed.results import epochs_around_best, median_around_best
from testbed.run import extra_tags, parse_args
from testbed.seeds import derive
from testbed.tracking import MlflowTracker, NullTracker
from testbed.train import JointConfig, train


class EnsembleOfOne(nn.Module):
  """One model in the form joint training takes: its answer as an ensemble's,
  and its own weights as the part that trains."""

  def __init__(self, expert: Expert):
    super().__init__()
    self.expert = expert

  @property
  def trainable(self) -> nn.Module:
    return self.expert

  def forward(self, x: torch.Tensor, answer_each_round: bool = False) -> Output:
    logits = self.expert(x)[:, None]  # (B, 1, C)
    return Output(average(logits), logits, [])

  def bagging(self, x: torch.Tensor) -> Output:
    return self(x)


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
  scheme = SeedOnly()
  budget = Budget(epochs=args.expert_epochs)
  joint = JointConfig(args.epochs, args.batch_size, args.lr, args.loss)

  accuracies = []
  for trial_seed in args.trial_seeds:
    pool = get_pool(scheme, args.arch, data, 1, derive(trial_seed, "pool"), budget)
    model = EnsembleOfOne(pool.experts[0].model)
    weights = sum(p.numel() for p in model.parameters())
    tracker = (
      NullTracker()
      if args.no_track
      else MlflowTracker(
        args.experiment, f"{args.arm}-trial{trial_seed}", description()
      )
    )
    with tracker:
      tracker.set_tags(extra_tags(args) | {"arm": args.arm, "trial_seed": trial_seed})
      tracker.log_params(
        {
          "data": {
            "expert": args.expert_dataset,
            "joint": args.joint_dataset,
            "test": args.test_dataset,
            "expert_samples": args.expert_samples,
            "joint_samples": args.joint_samples,
            "split_seed": args.split_seed,
            "digest": data.digest,
          },
          "experts": {
            "arch": args.arch,
            "scheme": scheme.config(),
            "budget": asdict(budget),
            "M": 1,
          },
          "pool": {"key": pool.key, "digest": pool.digest},
          "weights": {"experts": weights, "trained": weights},
          "joint": asdict(joint),
        }
      )
      history = train(model, data, joint, derive(trial_seed, "joint"), tracker)

    epochs = epochs_around_best(history)
    accuracy = median_around_best(history)
    accuracies.append(accuracy)
    print(
      f"[{args.arm}] trial seed {trial_seed}: pool {pool.key}"
      f"  before {history[0]['test/accuracy']:.4f}  after {accuracy:.4f}"
      f"  epochs {epochs.start}-{epochs.stop - 1}"
    )
  print(
    f"[{args.arm}] mean over {len(accuracies)} trial seeds: after {mean(accuracies):.4f}"
  )


if __name__ == "__main__":
  main()
