"""The model-grafting study's runner (studies/model_grafting).

uv run pytest tests/test_model_grafting.py
"""

import pytest

from studies.model_grafting import ARMS, EXPERIMENT
from studies.model_grafting.run import flags_for_testbed
from testbed.run import check_flags, parse_args


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_builds_valid_testbed_flags(arm):
  flags = flags_for_testbed(["--graft-arm", arm, "--trial-seeds", "0"])
  args = parse_args(flags)
  check_flags(args)
  assert args.graft and args.experiment == EXPERIMENT
  assert args.arm == f"{arm}-block4-block8"
  assert args.graft_bypass == (arm == "bypass")
  assert f"graft_arm={arm}" in args.tag


def test_points_reach_the_testbed_and_the_label():
  flags = flags_for_testbed(
    [
      "--graft-arm",
      "bypass",
      "--graft-read",
      "block2",
      "--graft-write",
      "block10",
      "--trial-seeds",
      "0",
    ]
  )
  args = parse_args(flags)
  assert (args.graft_read, args.graft_write) == ("block2", "block10")
  assert args.arm == "bypass-block2-block10"


def test_flags_the_arm_sets_are_refused():
  with pytest.raises(SystemExit, match="set by --graft-arm"):
    flags_for_testbed(
      ["--graft-arm", "graft_mean", "--graft-bypass", "--trial-seeds", "0"]
    )


def test_an_explicit_experiment_wins():
  flags = flags_for_testbed(
    ["--graft-arm", "graft_mean", "--trial-seeds", "0", "--experiment", "scratch"]
  )
  assert parse_args(flags).experiment == "scratch"
