"""
The degenerate cases study (studies/degenerate_cases): its choices are parts
the testbed has, its runner hands each case to testbed.run, and its plan and λ
rule cover the study.

  uv run pytest tests/test_degenerate_cases.py
"""

import json

import pytest

from studies.channel_ladder import GATE, LEVELS, WIDTH
from studies.degenerate_cases import (
  CASES,
  DIAL,
  EXCHANGES,
  LAMS,
  MIXERS,
  MOE_READ,
  NEGATIVE_CORRELATION,
  STUDY,
)
from studies.degenerate_cases.dial import pick
from studies.degenerate_cases.plan import arms
from studies.degenerate_cases.read import number, paired
from studies.degenerate_cases.run import experiment, flags_for_testbed
from testbed.experts import NegativeCorrelation, Untrained
from testbed.run import check_flags, extra_tags, make_scheme, parse_args
from testbed.selection import SELECTORS, ClonesOfFirst, TakeFirst

SELF = ["--direction", "forward", "--mixer", "self"]


def arm_args(argv: list[str]):
  args = parse_args(flags_for_testbed(argv))
  check_flags(args)
  return args


def test_a_channel_arm_is_the_ladders_level_1_at_the_chosen_pair():
  args = arm_args(["--case", "clones", "--direction", "backward", "--mixer", "mean"])
  level = LEVELS["1"]
  assert (args.encoder, args.decoder) == (level.encoder, level.decoder)
  assert (args.exchange, args.mixer) == ([["stage3", "stage1"]], "mean")
  assert (args.width, args.trial_seeds) == (WIDTH, list(range(15)))
  assert args.arm == "clones-backward-mean"
  assert extra_tags(args) == {"study": STUDY, "case": "clones", "level": "1"}


@pytest.mark.parametrize(
  "case, epochs",
  [("clones", 30), ("two-trained-one-untrained", 60), ("untrained", 150)],
)
def test_each_case_has_its_epochs_and_experiment(case, epochs):
  args = arm_args(["--case", case, *SELF])
  assert args.epochs == epochs
  assert args.experiment == experiment(case) == f"{STUDY}-{case}"


def test_clones_and_the_mixed_group_use_the_ladders_pools():
  """The same scheme, λ and budget as the ladder, so the same cached experts."""
  ladder = parse_args(["--arm", "a", "--trial-seeds", "0"])
  for case in ("clones", "two-trained-one-untrained"):
    args = arm_args(["--case", case, *SELF])
    assert make_scheme(args).config() == make_scheme(ladder).config()
    assert (args.arch, args.expert_epochs) == (ladder.arch, ladder.expert_epochs)
    assert (args.experts, args.pool_size) == (ladder.experts, ladder.pool_size)


def test_each_case_builds_its_group():
  clones = arm_args(["--case", "clones", *SELF])
  assert SELECTORS[clones.selector] is ClonesOfFirst
  assert clones.untrained_members == 0
  mixed = arm_args(["--case", "two-trained-one-untrained", *SELF])
  assert mixed.selector is None and mixed.untrained_members == 1  # defaults to first
  assert SELECTORS["first"] is TakeFirst
  untrained = arm_args(["--case", "untrained", *SELF])
  assert isinstance(make_scheme(untrained), Untrained)


def test_negative_correlation_sets_its_lam_and_the_moe_runs_there():
  args = arm_args(["--case", NEGATIVE_CORRELATION, "--lam", "2.5", *SELF])
  assert make_scheme(args) == NegativeCorrelation(2.5)
  assert args.arm == f"{NEGATIVE_CORRELATION}-lam2.5-forward-self"
  assert extra_tags(args)["lam"] == "2.5"
  args = arm_args(["--case", NEGATIVE_CORRELATION, "--lam", "0", "--moe"])
  assert make_scheme(args) == NegativeCorrelation(0.0)
  assert (args.gate, args.gate_read, args.exchange) == (GATE, MOE_READ, [])
  assert args.arm == f"{NEGATIVE_CORRELATION}-lam0-moe-{MOE_READ}"
  assert extra_tags(args)["baseline"] == "moe"


def test_the_plan_and_the_dial_ask_for_the_same_pools():
  """A pool's cache key hashes the scheme's config as JSON, where 0 and 0.0
  differ, so a λ must reach the scheme as a float however it was written."""
  planned = [arm for arm in arms() if NEGATIVE_CORRELATION in arm]
  built = {json.dumps(make_scheme(arm_args(arm)).config()) for arm in planned}
  dial = {json.dumps(NegativeCorrelation(lam).config()) for lam in LAMS}
  assert built == dial and len(built) == len(LAMS)
  assert set(LAMS) <= set(DIAL) and all(isinstance(lam, float) for lam in DIAL)


def test_a_given_flag_wins():
  args = arm_args(
    ["--case", "untrained", *SELF, "--epochs", "1", "--trial-seeds", "3", "--no-track"]
  )
  assert (args.epochs, args.trial_seeds, args.no_track) == (1, [3], True)


@pytest.mark.parametrize(
  "argv",
  [
    ["--case", NEGATIVE_CORRELATION, *SELF],  # testbed's default λ does almost nothing
    ["--case", "clones", "--lam", "2.5", *SELF],
    ["--case", "clones", "--moe"],
    ["--case", NEGATIVE_CORRELATION, "--lam", "2.5", "--moe", "--mixer", "self"],
    ["--case", NEGATIVE_CORRELATION, "--lam", "2.5", "--moe", *SELF],
    ["--case", "clones"],  # no kind of arm
    ["--direction", "forward"],  # no case
    ["--case", "clones", *SELF, "--arm", "mine"],
    ["--case", "clones", *SELF, "--experiment", "v3"],
    ["--case", "clones", *SELF, "--scheme", "seed_only"],
    ["--case", "clones", *SELF, "--selector=first"],
    ["--case", "clones", *SELF, "--untrained-members", "2"],
    ["--case", "clones", *SELF, "--exchange", "stage1", "stage2"],
    ["--case", "clones", *SELF, "--encoder", "pooled"],
  ],
)
def test_the_runner_refuses_an_arm_it_cant_build(argv):
  with pytest.raises(SystemExit):
    flags_for_testbed(argv)


def test_the_plan_is_every_arm_once_and_each_one_builds():
  plan = arms()
  mixers, pairs = len(MIXERS), len(EXCHANGES)
  channel = mixers * pairs * (len(CASES) - 1 + len(LAMS))
  assert len(plan) == channel + len(LAMS)  # one MoE per λ
  assert len({tuple(arm) for arm in plan}) == len(plan)
  assert len({arm_args(arm).arm for arm in plan}) == len(plan)
  assert plan[0][:2] == ["--case", "clones"]  # the most decisive case first


def test_the_lams_are_on_the_dial_and_start_at_the_control():
  assert LAMS[0] == 0 and set(LAMS) <= set(DIAL) and list(LAMS) == sorted(LAMS)


def test_reading_the_first_epochs_of_a_run_is_reading_a_shorter_run():
  """A run's first 30 epochs are the 30-epoch run's, so the rule applied to
  them gives that run's number. Step 0 always stays in."""
  val = [0.5] + [0.1] * 29 + [0.2] + [0.9] * 5  # best at 31: after the cut
  steps = [
    {"joint_val/accuracy": v, "test/accuracy": i / 100} for i, v in enumerate(val)
  ]
  run = {"steps": steps}
  assert number(run, upto=30) == number({"steps": steps[:31]}) == pytest.approx(28)
  assert number(run) == pytest.approx(31)  # epochs 29-33


def test_a_difference_is_its_mean_half_width_and_positive_count():
  assert paired([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]).split() == [
    "+1.00",
    "±",
    "2.48",
    "(",
    "2/3)",
  ]


def rows(*settings: tuple[float, list[float], list[float]]) -> dict:
  """{λ: a row per trial seed} from (λ, bagging per seed, agreement per seed)."""
  return {
    lam: [{"bagging": b, "agreement": a} for b, a in zip(bagging, agreement)]
    for lam, bagging, agreement in settings
  }


def test_the_lam_rule_on_a_made_up_dial():
  dial = rows(
    (0.0, [0.900, 0.900], [0.94, 0.94]),
    (2.0, [0.900, 0.900], [0.45, 0.55]),  # below 0.5 on only one trial seed
    (2.5, [0.900, 0.900], [0.33, 0.33]),  # the middle λ
    (4.0, [0.900, 0.891], [0.05, 0.05]),  # 0.45 below on average: the far λ
    (5.0, [0.880, 0.880], [0.00, 0.00]),  # bagging has broken
  )
  assert pick(dial) == (0.0, 2.5, 4.0)


def test_the_lam_rule_says_so_if_no_lam_makes_the_experts_disagree():
  with pytest.raises(SystemExit):
    pick(rows((0.0, [0.9], [0.94]), (2.0, [0.9], [0.8])))
