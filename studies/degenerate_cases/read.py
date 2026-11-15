"""
Read the degenerate cases by the rules fixed in docs/studies/degenerate-cases.md.
Read-only on MLflow. Each run is read by testbed/results.py's rule, and arms
are compared trial seed by trial seed. A difference is printed as its mean over
the trial seeds, in accuracy points, ± the 95% half-width (t over trial seeds),
then on how many trial seeds it is positive.

  uv run python -m studies.degenerate_cases.read

Clones and two-trained-one-untrained are compared with the channel ladder's own
level 1 runs at r = 8 (phase 1 for attention and self, phase 2c for mean),
which used the same pools on every trial seed.
"""

import math
import sqlite3
from statistics import mean, median, stdev

from scipy import stats

from studies.channel_ladder.run import TRIAL_SEEDS
from studies.degenerate_cases import (
  CASES,
  EXCHANGES,
  LAMS,
  MIXERS,
  NEGATIVE_CORRELATION,
  STUDY,
)
from studies.degenerate_cases.run import experiment
from testbed.data import REPO_ROOT
from testbed.results import epochs_around_best, median_around_best

LADDER = {"self": "channel-ladder-phase1", "attention": "channel-ladder-phase1"}
LADDER["mean"] = "channel-ladder-phase2c"
UNTRAINED = 2  # the place two-trained-one-untrained gives its untrained expert
OWN = f"expert_accuracy/expert{UNTRAINED}"
SEEDS = list(TRIAL_SEEDS)
Run = dict  # {"tags", "params", "steps": [{metric: value} per step]}


def runs(db: sqlite3.Connection, name: str) -> list[Run]:
  """An experiment's finished runs, with every metric at every step."""
  out = []
  for (run,) in db.execute(
    """select r.run_uuid from runs r join experiments e using (experiment_id)
       where e.name = ? and r.status = 'FINISHED' and r.lifecycle_stage = 'active'""",
    (name,),
  ):
    tags = dict(db.execute('select "key", value from tags where run_uuid = ?', (run,)))
    params = dict(
      db.execute('select "key", value from params where run_uuid = ?', (run,))
    )
    steps: dict[int, dict[str, float]] = {}
    for key, step, value in db.execute(
      'select "key", step, value from metrics where run_uuid = ?', (run,)
    ):
      steps.setdefault(step, {})[key] = value
    out.append(
      {"tags": tags, "params": params, "steps": [steps[s] for s in sorted(steps)]}
    )
  return out


def by_seed(rows: list[Run]) -> dict[int, Run]:
  out = {int(r["tags"]["trial_seed"]): r for r in rows}
  if sorted(out) != SEEDS or len(rows) != len(SEEDS):
    raise SystemExit(
      f"{rows[0]['tags']['arm']}: trial seeds {sorted(out)}, {len(rows)} runs"
    )
  return out


def number(
  run: Run,
  metric: str = "test/accuracy",
  picked_by: str = "joint_val/accuracy",
  upto=None,
) -> float:
  """A run's number by the rule, in points, reading only its first `upto` epochs."""
  steps = run["steps"] if upto is None else run["steps"][: upto + 1]
  return 100 * median_around_best(steps, metric, picked_by=picked_by)


def ci(xs: list[float]) -> tuple[float, float]:
  """Mean and 95% half-width, t over trial seeds."""
  half = float(stats.t.ppf(0.975, len(xs) - 1)) * stdev(xs) / math.sqrt(len(xs))
  return mean(xs), half


def paired(a: list[float], b: list[float]) -> str:
  """a − b trial seed by trial seed."""
  diffs = [x - y for x, y in zip(a, b, strict=True)]
  m, h = ci(diffs)
  return f"{m:+6.2f} ± {h:4.2f} ({sum(d > 0 for d in diffs):2d}/{len(diffs)})"


def level(xs: list[float]) -> str:
  m, h = ci(xs)
  return f"{m:6.2f} ± {h:4.2f}"


def values(arm: dict[int, Run], read=number, **how) -> list[float]:
  return [read(arm[s], **how) for s in SEEDS]


def main() -> None:
  db = sqlite3.connect(f"file:{REPO_ROOT / 'mlflow.db'}?mode=ro", uri=True, timeout=30)
  rows: dict[tuple, list[Run]] = {}
  for case in CASES:
    for run in runs(db, experiment(case)):
      t, p = run["tags"], run["params"]
      kind = (
        "moe" if t.get("baseline") == "moe" else (t["direction"], p["channel.mixer"])
      )
      rows.setdefault((case, t.get("lam"), kind), []).append(run)
  study = {key: by_seed(r) for key, r in rows.items()}
  ladder = {}
  for direction, (read, write) in EXCHANGES.items():
    for mixer in MIXERS:
      name = f"level1-{'' if mixer == 'attention' else mixer + '-'}{read}-{write}"
      matching = [r for r in runs(db, LADDER[mixer]) if r["tags"]["arm"] == name]
      ladder[direction, mixer] = by_seed(matching)

  print(f"=== integrity: {STUDY}")
  epochs = {CASES[case].epochs for case in CASES}
  print(
    "arms",
    len(study),
    "runs",
    sum(len(a) for a in study.values()),
    "expected epochs",
    epochs,
  )
  wrong = [
    (key, s)
    for key, arm in study.items()
    for s, run in arm.items()
    if len(run["steps"]) - 1 != CASES[key[0]].epochs
  ]
  print("runs with the wrong number of epochs:", wrong)
  nan = sum(
    v != v
    for arm in study.values()
    for r in arm.values()
    for st in r["steps"]
    for v in st.values()
  )
  print("NaN values logged:", nan)
  print(
    "commits",
    {r["tags"]["git_commit"][:7] for a in study.values() for r in a.values()},
    "dirty",
    {r["tags"]["git_dirty"] for a in study.values() for r in a.values()},
  )
  for case in ("clones", "two-trained-one-untrained"):
    differ = [
      (key, s)
      for key, arm in study.items()
      if key[0] == case
      for s in SEEDS
      if arm[s]["params"]["pool.digest"]
      != ladder["forward", "self"][s]["params"]["pool.digest"]
    ]
    print(
      f"{case} runs whose pool differs from the ladder's on the trial seed:", differ
    )
  groups = {
    r["params"]["pool.group"]
    for k, a in study.items()
    if k[0] == "clones"
    for r in a.values()
  }
  print("clones' groups", groups)
  for lam in sorted({k[1] for k in study if k[0] == NEGATIVE_CORRELATION}, key=float):
    digests = {
      (s, r["params"]["pool.digest"])
      for k, a in study.items()
      if k[:2] == (NEGATIVE_CORRELATION, lam)
      for s, r in a.items()
    }
    print(
      f"negative correlation λ {lam}: pools per trial seed", len(digests) / len(SEEDS)
    )

  def arm(case, direction, mixer, lam=None):
    return study[case, lam, (direction, mixer)]

  print("\n=== clones: three copies of the ladder's first expert, 30 epochs")
  for direction in EXCHANGES:
    c = {m: values(arm("clones", direction, m)) for m in MIXERS}
    d = {m: values(ladder[direction, m]) for m in MIXERS}
    bag_c = [
      100 * r["steps"][0]["test/accuracy"]
      for r in arm("clones", direction, "self").values()
    ]
    bag_d = [
      100 * r["steps"][0]["test/accuracy"] for r in ladder[direction, "self"].values()
    ]
    print(f"\n{direction}: bagging clones {level(bag_c)}, ladder {level(bag_d)}")
    print(f"  {'mixer':10s}{'clones':>16s}{'ladder':>16s}   clones − ladder")
    for m in MIXERS:
      print(f"  {m:10s}{level(c[m]):>16s}{level(d[m]):>16s}   {paired(c[m], d[m])}")
    for a, b in (("mean", "self"), ("attention", "self"), ("attention", "mean")):
      gap_c = [x - y for x, y in zip(c[a], c[b])]
      gap_d = [x - y for x, y in zip(d[a], d[b])]
      print(
        f"  {a} − {b:5s}  clones {paired(c[a], c[b])}   ladder {paired(d[a], d[b])}"
        f"   ladder − clones {paired(gap_d, gap_c)}"
      )

  print("\n=== two trained + one untrained: the untrained expert's own accuracy")
  print("read around its own best joint_val epoch; and the ensemble's number")
  case = "two-trained-one-untrained"
  for direction in EXCHANGES:
    print(f"\n{direction}")
    for upto in (60, 30):
      own = {
        m: values(
          arm(case, direction, m),
          metric=f"test/{OWN}",
          picked_by=f"joint_val/{OWN}",
          upto=upto,
        )
        for m in MIXERS
      }
      print(
        f"  first {upto} epochs, untrained expert: "
        + "  ".join(f"{m} {level(own[m])}" for m in MIXERS)
      )
      print(
        f"    mean − self {paired(own['mean'], own['self'])}   attention − self {paired(own['attention'], own['self'])}"
        f"   attention − mean {paired(own['attention'], own['mean'])}"
      )
    start = [
      100 * r["steps"][0][f"test/{OWN}"] for r in arm(case, direction, "self").values()
    ]
    print(f"  untrained expert at step 0: {level(start)}")
    ens = {m: values(arm(case, direction, m)) for m in MIXERS}
    d = {m: values(ladder[direction, m]) for m in MIXERS}
    print("  ensemble: " + "  ".join(f"{m} {level(ens[m])}" for m in MIXERS))
    print(
      f"    mean − self {paired(ens['mean'], ens['self'])}   attention − self {paired(ens['attention'], ens['self'])}"
    )
    for m in MIXERS:
      print(
        f"    {m:10s} minus the ladder's (three trained experts) {paired(ens[m], d[m])}"
      )
    for m in MIXERS:
      trained = [
        mean(
          100 * median_around_best(r["steps"], f"test/expert_accuracy/expert{i}")
          for i in (0, 1)
        )
        for r in arm(case, direction, m).values()
      ]
      print(f"    {m:10s} the two trained experts' own accuracy {level(trained)}")

  print("\n=== negative correlation: experts trained to disagree, 30 epochs")
  gaps = {}
  for lam in (f"{x:g}" for x in LAMS):
    moe = values(study[NEGATIVE_CORRELATION, lam, "moe"])
    for direction in EXCHANGES:
      n = {m: values(arm(NEGATIVE_CORRELATION, direction, m, lam)) for m in MIXERS}
      runs0 = arm(NEGATIVE_CORRELATION, direction, "self", lam)
      bag = [100 * r["steps"][0]["test/accuracy"] for r in runs0.values()]
      pre = [
        100 * r["steps"][0]["test/joint_answer_shift/agreement_pre"]
        for r in runs0.values()
      ]
      print(
        f"\nλ {lam} {direction}: bagging {level(bag)}  agreement before {level(pre)}  MoE {level(moe)}"
      )
      for m in MIXERS:
        agree = values(
          arm(NEGATIVE_CORRELATION, direction, m, lam),
          metric="test/joint_answer_shift/agreement_post",
        )
        own = values(
          arm(NEGATIVE_CORRELATION, direction, m, lam), metric="test/expert_accuracy"
        )
        print(
          f"  {m:10s}{level(n[m]):>16s}   agreement after {level(agree)}   expert accuracy {level(own)}"
          f"   MoE − {m} {paired(moe, n[m])}"
        )
      for a, b in (("mean", "self"), ("attention", "self"), ("attention", "mean")):
        gaps[lam, direction, a, b] = [x - y for x, y in zip(n[a], n[b])]
        print(f"  {a} − {b:5s} {paired(n[a], n[b])}")
  print("\nthe gap at each λ minus the same gap at λ = 0")
  for lam in (f"{x:g}" for x in LAMS[1:]):
    for direction in EXCHANGES:
      print(
        f"  λ {lam} {direction:9s}"
        + "   ".join(
          f"{a} − {b}: {paired(gaps[lam, direction, a, b], gaps['0', direction, a, b])}"
          for a, b in (("mean", "self"), ("attention", "self"))
        )
      )

  print("\n=== untrained: three experts with random weights, 150 epochs")
  for direction in EXCHANGES:
    print(f"\n{direction}")
    for upto in (30, 60, 150):
      u = {m: values(arm("untrained", direction, m), upto=upto) for m in MIXERS}
      print(
        f"  first {upto:3d} epochs: " + "  ".join(f"{m} {level(u[m])}" for m in MIXERS)
      )
      print(
        f"    mean − self {paired(u['mean'], u['self'])}   attention − self {paired(u['attention'], u['self'])}"
        f"   attention − mean {paired(u['attention'], u['mean'])}"
      )
    for m in MIXERS:
      ends = [
        epochs_around_best(r["steps"]).stop - 1
        for r in arm("untrained", direction, m).values()
      ]
      print(
        f"  {m:10s} windows ending in the last 10 epochs: {sum(e > 140 for e in ends)}/{len(ends)}"
      )

  print("\n=== how attention spread its weight, over each run's window")
  print("entropy 0 is one sender, log 3 = 1.099 all three equally; top1 1/3 to 1;")
  print("mode share 1 is the same routing for every example, whatever it shows")
  sites = {"forward": "ex0", "backward": "ex0/round0"}
  names = ("attn_entropy", "attn_top1", "routing_mode_share/joint")
  for key in sorted(k for k in study if k[2] != "moe" and k[2][1] == "attention"):
    case, lam, (direction, _) = key
    read = [
      [
        median_around_best(r["steps"], f"{name}/{sites[direction]}")
        for r in study[key].values()
      ]
      for name in names
    ]
    label = case if lam is None else f"{case} λ {lam}"
    print(
      f"  {label:28s}{direction:9s}"
      + "".join(
        f"  {name.split('/')[0]} {median(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}]"
        for name, xs in zip(names, read)
      )
    )


if __name__ == "__main__":
  main()
