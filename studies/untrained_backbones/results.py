"""
The untrained-backbones study's tables, from its logged runs
(docs/studies/untrained-backbones.md):

  uv run python -m studies.untrained_backbones.results

Reads mlflow.db read-only. Every run is boiled down by testbed/results.py:
a metric's median over the 5 trained epochs around the run's best joint_val
epoch. Values are in points (x100) where they are accuracies, as the mean
± SE over trial seeds; a paired difference is taken seed by seed first.
"""

import sqlite3
from collections import defaultdict
from statistics import mean, stdev

from studies.untrained_backbones.run import (
  BACKBONES,
  EXPERIMENT,
  STUDY_LEVELS,
  TRIAL_SEEDS,
)
from studies.channel_ladder import MIXERS
from testbed.data import REPO_ROOT
from testbed.results import PICKED_BY, median_around_best

DIRECTIONS = ("forward", "backward")
CONTROL = "self"
SITES = {"forward": "ex0", "backward": "ex0/round0"}  # where exchange metrics log
CHECKPOINTS = (1, 10, 20, 30)  # epochs at which the joint_val curve is shown


def load(experiment: str = EXPERIMENT) -> dict[tuple, list[dict[str, float]]]:
  """(backbone, direction, level, mixer, trial seed) -> the run's history,
  one {metric: value} per step, step 0 first."""
  db = sqlite3.connect(f"file:{REPO_ROOT / 'mlflow.db'}?mode=ro", uri=True)
  runs = [
    run
    for (run,) in db.execute(
      """select run_uuid from runs join experiments using (experiment_id)
         where experiments.name = ? and runs.status = 'FINISHED'""",
      (experiment,),
    )
  ]
  tags, steps = defaultdict(dict), defaultdict(lambda: defaultdict(dict))
  for run in runs:
    for key, value in db.execute(
      "select key, value from tags where run_uuid = ?", (run,)
    ):
      tags[run][key] = value
    for key, value, step in db.execute(
      "select key, value, step from metrics where run_uuid = ?", (run,)
    ):
      steps[run][step][key] = value
  out = {}
  for run in runs:
    t = tags[run]
    key = (t["backbone"], t["direction"], t["level"], t["mixer"], int(t["trial_seed"]))
    if key in out:
      raise ValueError(f"two finished runs for {key}")
    out[key] = [steps[run][s] for s in sorted(steps[run])]
  return out


def fmt(xs: list[float], scale: float = 100, signed: bool = False) -> str:
  sign = "+" if signed else ""
  return f"{scale * mean(xs):{sign}.2f} ± {scale * stdev(xs) / len(xs) ** 0.5:.2f}"


def settings():
  for direction in DIRECTIONS:
    for level in STUDY_LEVELS:
      yield direction, level


def table(header: list[str], rows: list[list[str]]) -> str:
  lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
  return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def main() -> None:
  runs = load()
  expected = len(BACKBONES) * len(DIRECTIONS) * len(STUDY_LEVELS) * len(MIXERS)
  print(f"{len(runs)} finished runs; the grid has {expected * len(TRIAL_SEEDS)}\n")

  def score(key, metric="test/accuracy"):
    return median_around_best(runs[key], metric)

  def per_seed(backbone, direction, level, mixer, metric="test/accuracy"):
    return [score((backbone, direction, level, mixer, s), metric) for s in TRIAL_SEEDS]

  print("## Test accuracy\n")
  rows = []
  for backbone in BACKBONES:
    for direction, level in settings():
      bagging = [runs[(backbone, direction, level, CONTROL, s)][0]["test/accuracy"] for s in TRIAL_SEEDS]
      rows.append(
        [backbone, direction, level, fmt(bagging)]
        + [fmt(per_seed(backbone, direction, level, m)) for m in MIXERS]
      )
  print(table(["backbone", "direction", "level", "bagging (step 0)", *MIXERS], rows), "\n")

  print("## Experts' own test accuracy (expert_accuracy)\n")
  rows = [
    [backbone, direction, level]
    + [fmt(per_seed(backbone, direction, level, m, "test/expert_accuracy")) for m in MIXERS]
    for backbone in BACKBONES
    for direction, level in settings()
  ]
  print(table(["backbone", "direction", "level", *MIXERS], rows), "\n")

  others = [m for m in MIXERS if m != CONTROL]
  print(f"## Lead over {CONTROL}, test, paired by trial seed\n")
  rows = []
  for backbone in BACKBONES:
    for direction, level in settings():
      control = per_seed(backbone, direction, level, CONTROL)
      rows.append(
        [backbone, direction, level]
        + [
          fmt([a - c for a, c in zip(per_seed(backbone, direction, level, m), control)], signed=True)
          for m in others
        ]
      )
    pooled = []
    for m in others:
      seeds = [
        mean(
          score((backbone, d, lv, m, s)) - score((backbone, d, lv, CONTROL, s))
          for d, lv in settings()
        )
        for s in TRIAL_SEEDS
      ]
      pooled.append(fmt(seeds, signed=True))
    rows.append([backbone, "pooled", "", *pooled])
  print(table(["backbone", "direction", "level", *others], rows), "\n")

  print("## Injection norm (msg_norm/injection, joint_val)\n")
  rows = [
    [backbone, direction, level]
    + [
      fmt(per_seed(backbone, direction, level, m, f"msg_norm/injection/{SITES[direction]}"), scale=1)
      for m in MIXERS
    ]
    for backbone in BACKBONES
    for direction, level in settings()
  ]
  print(table(["backbone", "direction", "level", *MIXERS], rows), "\n")

  print("## Overfitting: joint accuracy minus joint_val accuracy\n")
  rows = []
  for backbone in BACKBONES:
    for direction, level in settings():
      cells = []
      for m in MIXERS:
        joint = per_seed(backbone, direction, level, m, "joint/accuracy")
        val = per_seed(backbone, direction, level, m, "joint_val/accuracy")
        cells.append(fmt([j - v for j, v in zip(joint, val)], signed=True))
      rows.append([backbone, direction, level, *cells])
  print(table(["backbone", "direction", "level", *MIXERS], rows), "\n")

  print("## Convergence\n")
  rows = []
  for backbone in BACKBONES:
    for direction, level in settings():
      for m in MIXERS:
        histories = [runs[(backbone, direction, level, m, s)] for s in TRIAL_SEEDS]
        best = [
          max(range(1, len(h)), key=lambda step: h[step][PICKED_BY]) for h in histories
        ]
        curve = [fmt([h[e]["joint_val/accuracy"] for h in histories]) for e in CHECKPOINTS]
        rows.append([backbone, direction, level, m, *curve, ", ".join(map(str, best))])
  print(
    table(
      ["backbone", "direction", "level", "mixer"]
      + [f"joint_val, epoch {e}" for e in CHECKPOINTS]
      + ["best joint_val epoch, per seed"],
      rows,
    ),
    "\n",
  )

  print("## Untrained backbones: test accuracy per trial seed\n")
  rows = [
    [direction, level, m] + [f"{100 * x:.2f}" for x in per_seed("untrained", direction, level, m)]
    for direction, level in settings()
    for m in MIXERS
  ]
  print(table(["direction", "level", "mixer", *map(str, TRIAL_SEEDS)], rows))


if __name__ == "__main__":
  main()
