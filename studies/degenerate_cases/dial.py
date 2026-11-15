"""
Pick negative correlation's λ, before any channel runs. Trains each λ's pool
on every trial seed, into the expert cache, and reads what the experts do on
joint_val alone. No channel is involved and test is never read, so nothing
selects on an arm's result.

  uv run python -m studies.degenerate_cases.dial

The rule (docs/studies/degenerate-cases.md) takes λ = 0 as the control, then:
  - the middle λ is the smallest with agreement below AGREEMENT_BELOW on every
    trial seed, so the experts really disagree and it isn't a λ where a small
    change swings them;
  - the far λ is the largest whose bagging, averaged over trial seeds, stays
    within BAGGING_WITHIN of λ = 0's, so the group still knows as much.

Put what it prints in LAMS (studies/degenerate_cases/__init__.py).
"""

import argparse
from statistics import mean

import torch

from studies.channel_ladder.run import TRIAL_SEEDS
from studies.degenerate_cases import AGREEMENT_BELOW, BAGGING_WITHIN, DIAL
from testbed.data import Data, load
from testbed.experts import Budget, Expert, NegativeCorrelation, get_pool
from testbed.metrics import accuracy, expert_accuracy
from testbed.metrics.answers import agreement
from testbed.seeds import derive

Row = dict[str, float]


@torch.no_grad()
def profile(models: list[Expert], data: Data) -> Row:
  """What the experts do on joint_val with no channel. All three are at step 0
  of any run on them."""
  x, y = data.joint_val.x, data.joint_val.y
  logits = torch.stack(
    [torch.cat([m(batch) for batch in x.split(2000)]) for m in models], dim=1
  )
  return {
    "expert": expert_accuracy(logits, y),
    "bagging": accuracy(logits, y),
    "agreement": agreement(logits),
  }


def pick(rows: dict[float, list[Row]]) -> tuple[float, float, float]:
  """(λ = 0, the middle λ, the far λ), from each λ's row per trial seed."""
  bagging = mean(row["bagging"] for row in rows[0.0])
  far = max(
    lam
    for lam, seeds in rows.items()
    if bagging - mean(row["bagging"] for row in seeds) <= BAGGING_WITHIN
  )
  disagree = [
    lam
    for lam, seeds in rows.items()
    if all(row["agreement"] < AGREEMENT_BELOW for row in seeds)
  ]
  if not disagree:
    raise SystemExit("no λ has the experts disagree on every trial seed; widen DIAL")
  return 0.0, min(disagree), far


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
  p.add_argument("--trial-seeds", type=int, nargs="+", default=list(TRIAL_SEEDS))
  p.add_argument("--device", default="cuda")
  args = p.parse_args(argv)

  data = load("fashion", split_seed=0, device=args.device)
  rows: dict[float, list[Row]] = {}
  for lam in DIAL:
    rows[lam] = []
    for trial_seed in args.trial_seeds:
      pool = get_pool(
        NegativeCorrelation(lam), "cnn3", data, 3, derive(trial_seed, "pool"), Budget()
      )
      rows[lam].append(profile([e.model for e in pool.experts], data))
    seeds = rows[lam]
    agreements = [row["agreement"] for row in seeds]
    print(
      f"λ {lam:<4g} expert {mean(r['expert'] for r in seeds):.4f}"
      f"  bagging {mean(r['bagging'] for r in seeds):.4f}"
      f"  agreement {mean(agreements):.4f} [{min(agreements):.4f}, {max(agreements):.4f}]"
    )
  print("LAMS =", pick(rows))


if __name__ == "__main__":
  main()
