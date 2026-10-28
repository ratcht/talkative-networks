"""
The channel ladder study (studies/channel_ladder): its architectural choices
are parts the testbed has, its levels differ only where their GELUs are, and
its runner hands each kind of arm to testbed.run.

  uv run pytest tests/test_channel_ladder.py
"""

import pytest

from studies.channel_ladder import GATE, LEVELS, MIXERS, STUDY, WIDTH, at_level
from studies.channel_ladder.run import experiment, flags_for_testbed
from testbed.communication import DECODERS, ENCODERS, build_channel
from testbed.communication import MIXERS as TESTBED_MIXERS
from testbed.moe import GATES
from testbed.run import extra_tags, parse_args


def test_every_choice_is_a_part_the_testbed_has():
  for level in LEVELS.values():
    assert level.encoder in ENCODERS and level.decoder in DECODERS
  assert set(MIXERS) <= set(TESTBED_MIXERS) and MIXERS[0] == "attention"
  assert GATE in GATES


def test_every_level_has_the_same_weights_at_the_studys_width():
  """Levels differ only in where their GELUs are."""
  shapes = set()
  for name in LEVELS:
    config = at_level(name)
    assert config.width == WIDTH
    channel = build_channel(config, (32, 7, 7), (16, 14, 14), num_experts=2)
    shapes.add(tuple((k, v.shape) for k, v in channel.state_dict().items()))
  assert len(shapes) == 1
  assert at_level("1", width=32).width == 32  # the width can still be ablated


ARGS = ["--arm", "a", "--phase", "2a"]
TAGS = {"study": STUDY, "phase": "2a"}


def test_a_level_gets_its_parts_mixer_and_tags():
  argv = ["--level", "2-decoder", "--exchange", "stage2", "stage1", *ARGS]
  args = parse_args(flags_for_testbed(argv))
  assert (args.encoder, args.decoder) == ("linear_taper", "linear_taper_gelu")
  assert (args.exchange, args.mixer) == ([["stage2", "stage1"]], "attention")
  assert extra_tags(args) == TAGS | {"level": "2-decoder"}
  assert (args.epochs, args.width) == (30, WIDTH)  # the study's defaults
  assert args.trial_seeds == list(range(15))
  args = parse_args(
    flags_for_testbed([*argv, "--mixer", "self", "--epochs", "5", "--trial-seeds", "3"])
  )
  assert (args.mixer, args.epochs, args.trial_seeds) == (
    "self",
    5,
    [3],
  )  # a given flag wins


def test_the_moe_gets_the_studys_gate_and_tags():
  args = parse_args(flags_for_testbed(["--moe", "stage2", *ARGS]))
  assert (args.gate, args.gate_read, args.exchange) == (GATE, "stage2", [])
  assert extra_tags(args) == TAGS | {"baseline": "moe"}
  assert (args.epochs, args.width) == (30, WIDTH)


def test_bagging_trains_nothing_and_is_tagged():
  args = parse_args(flags_for_testbed(["--bagging", *ARGS]))
  assert (args.gate, args.exchange) == (None, [])
  assert extra_tags(args) == TAGS | {"baseline": "bagging"}


def test_each_phase_is_its_own_experiment():
  args = parse_args(flags_for_testbed(["--bagging", *ARGS]))
  assert args.experiment == experiment("2a") == f"{STUDY}-phase2a"
  with pytest.raises(SystemExit):
    flags_for_testbed(["--bagging", "--arm", "a"])  # a run needs a phase
  with pytest.raises(SystemExit):
    flags_for_testbed(["--bagging", *ARGS, "--experiment", "v3"])


@pytest.mark.parametrize(
  "argv",
  [
    ["--level", "3"],  # no exchange to put the channel in
    [
      "--level",
      "3",
      "--exchange",
      "stage2",
      "stage1",
      "--exchange",
      "stage3",
      "stage1",
    ],
    ["--level", "3", "--exchange", "stage2", "stage1", "--encoder", "pooled"],
    ["--level", "3", "--exchange", "stage2", "stage1", "--decoder=broadcast"],
    ["--level", "3", "--exchange", "stage2", "stage1", "--gate-read", "stage3"],
    ["--moe", "stage3", "--exchange", "stage2", "stage3"],
    ["--moe", "stage3", "--mixer", "self"],
    ["--moe", "stage3", "--gate", "linear_taper_gelu"],
    ["--bagging", "--exchange", "stage2", "stage3"],
    ["--level", "1", "--moe", "stage3"],  # one kind of arm at a time
    [],  # and one is needed
  ],
)
def test_the_runner_refuses_an_arm_it_cant_build(argv):
  with pytest.raises(SystemExit):
    flags_for_testbed([*argv, *ARGS])
