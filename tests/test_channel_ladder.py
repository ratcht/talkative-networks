"""
The channel ladder study (studies/channel_ladder): its architectural choices
are parts the testbed has, its levels differ only where their GELUs are, and
its runner hands each kind of arm to testbed.run.

  uv run pytest tests/test_channel_ladder.py
"""

import pytest
import torch
from fakes import fake_data, random_experts

from studies.channel_ladder import (
  COSINE,
  GATE,
  LEVELS,
  MIXERS,
  STUDY,
  WIDTH,
  at_level,
)
from studies.channel_ladder.phase4 import arms as phase4_arms
from studies.channel_ladder.phase5 import ARMS as PHASE5
from studies.channel_ladder.phase5 import Changes, apply
from studies.channel_ladder.phase5 import flags as phase5_flags
from studies.channel_ladder.run import experiment, flags_for_testbed
from testbed.communication import (
  DECODERS,
  ENCODERS,
  Exchange,
  build_channel,
  build_ensemble,
)
from testbed.communication import MIXERS as TESTBED_MIXERS
from testbed.moe import GATES
from testbed.run import extra_tags, parse_args


def test_every_choice_is_a_part_the_testbed_has():
  for level in LEVELS.values():
    assert level.encoder in ENCODERS and level.decoder in DECODERS
  assert set(MIXERS) <= set(TESTBED_MIXERS) and MIXERS[0] == "attention"
  assert COSINE in TESTBED_MIXERS
  assert GATE in GATES


def test_cosine_attention_stays_out_of_the_mixers_other_studies_borrow():
  """The degenerate cases make an arm of every one of MIXERS, so adding it
  there would add arms to a study that doesn't want them."""
  assert COSINE not in MIXERS


def test_phase_4_runs_the_forward_levels_and_two_with_cosine_attention():
  args = [parse_args(flags_for_testbed(flags)) for flags in phase4_arms()]
  assert {(a.arm, a.mixer) for a in args} == {
    ("level1-cosine_attention-stage2-stage3", COSINE),
    ("level2-encoder-cosine_attention-stage2-stage3", COSINE),
    # as named in phases 1 and 2a, so arms can be matched across phases
    ("level1-stage2-stage3", "attention"),
    ("level2-encoder-stage2-stage3", "attention"),
    ("level2-decoder-stage2-stage3", "attention"),
    ("level3-stage2-stage3", "attention"),
  }
  for a in args:
    assert a.exchange == [["stage2", "stage3"]]
    assert a.experiment == f"{STUDY}-phase4"
    assert (a.epochs, a.width, a.trial_seeds) == (30, WIDTH, list(range(15)))
    assert extra_tags(a)["phase"] == "4"


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


def test_phase_5_runs_level_1_forward_with_its_interventions_tagged():
  args = {
    label: parse_args(flags_for_testbed(phase5_flags(arm)))
    for label, arm in PHASE5.items()
  }
  assert set(args) == {
    "level1-self-centred-stage2-stage3",
    "level1-centred-stage2-stage3",
    "level1-frozen-readings-stage2-stage3",
    "level1-routing-cut-stage2-stage3",
    "level1-cosine_attention-value-cut-stage2-stage3",
    "level1-cosine_attention-value-cut-no-bias-stage2-stage3",
    # as named in phases 1 and 4, so they match across phases
    "level1-self-stage2-stage3",
    "level1-stage2-stage3",
    "level1-cosine_attention-stage2-stage3",
  }
  for label, a in args.items():
    assert a.arm == label and a.mixer == PHASE5[label].mixer
    assert (a.encoder, a.decoder) == ("linear_taper", "linear_taper")
    assert a.exchange == [["stage2", "stage3"]]
    assert a.experiment == f"{STUDY}-phase5"
    assert (a.epochs, a.width, a.trial_seeds) == (30, WIDTH, list(range(15)))
    tags = extra_tags(a)
    assert tags["phase"] == "5" and tags["level"] == "1"
    assert tags["intervention"] == PHASE5[label].intervention
  references = [arm for arm in PHASE5.values() if arm.intervention == "none"]
  assert len(references) == 3 and all(a.changes == Changes() for a in references)


def phase5_model(mixer: str, changes: Changes, seed: int = 0):
  data = fake_data()
  model = build_ensemble(
    random_experts(),
    [Exchange("stage2", "stage3")],
    at_level("1", mixer=mixer),
    data.input_shape,
    1,
    seed,
  )
  apply(changes, model, data)
  return model, data


def features_of(model, x):
  return [expert.run(x, until="stage2") for expert in model.experts]


@pytest.mark.parametrize("label", PHASE5)
def test_phase_5_changes_nothing_before_training(label):
  """Every arm starts as bagging, from level 1's own initial weights under the
  same parameter names, so it pairs with its reference trial seed by trial seed
  and the metrics read the same parts. Only biases an arm removes differ."""
  arm = PHASE5[label]
  model, data = phase5_model(arm.mixer, arm.changes)
  plain, _ = phase5_model(arm.mixer, Changes())
  x = data.joint_val.x
  with torch.no_grad():
    assert torch.allclose(model(x).log_probs, model.bagging(x).log_probs, atol=1e-6)
  ours, theirs = model.channels.state_dict(), plain.channels.state_dict()
  assert set(k for k in ours if "centre" not in k) == set(theirs)
  for name, weight in theirs.items():
    if arm.changes.no_head_biases and name.endswith(("query.bias", "key.bias")):
      assert not ours[name].any()
    else:
      assert torch.equal(ours[name], weight), name


def test_cutting_routing_stops_its_gradient_at_the_readings_and_nowhere_else():
  model, data = phase5_model(COSINE, Changes(routing_cut=(0, 1, 2)))
  (channel,) = model.channels
  reading = channel.read(features_of(model, data.joint.x[:32]))
  reading.weights[..., 1, 0].sum().backward()
  for encoder in channel.encoders:
    assert encoder.read.weight.grad is None
    assert encoder.query.weight.grad.abs().sum() > 0
    assert encoder.key.weight.grad.abs().sum() > 0


def test_cutting_the_value_stops_only_that_experts_gradient():
  model, data = phase5_model(COSINE, Changes(value_cut=(0,)))
  (channel,) = model.channels
  reading = channel.read(features_of(model, data.joint.x[:32]))
  reading.sent.value.pow(2).sum().backward()
  assert channel.encoders[0].read.weight.grad is None
  assert channel.encoders[0].value.weight.grad.abs().sum() > 0
  assert channel.encoders[1].read.weight.grad.abs().sum() > 0


def test_a_cut_value_gradient_leaves_routing_reaching_the_readings():
  model, data = phase5_model(COSINE, Changes(value_cut=(0,)))
  (channel,) = model.channels
  reading = channel.read(features_of(model, data.joint.x[:32]))
  reading.weights[..., 0, 0].sum().backward()
  assert channel.encoders[0].read.weight.grad.abs().sum() > 0


def test_frozen_readings_never_move_while_the_heads_can():
  model, data = phase5_model("attention", Changes(frozen_readings=True))
  (channel,) = model.channels
  before = channel.encoders[0].read.weight.clone()
  value = channel.encoders[0].value.weight.clone()
  for decoder in channel.decoders:
    torch.nn.init.normal_(decoder.write.weight)
  optimizer = torch.optim.Adam(channel.parameters(), lr=1e-3)
  model(data.joint.x[:32]).log_probs.sum().backward()
  optimizer.step()
  assert torch.equal(channel.encoders[0].read.weight, before)
  assert not torch.equal(channel.encoders[0].value.weight, value)


def test_centring_subtracts_the_average_of_the_joint_split_exactly():
  model, data = phase5_model(COSINE, Changes(centred_input=True))
  (channel,) = model.channels
  with torch.no_grad():
    readings = channel.read(features_of(model, data.joint.x)).sent.readings
    for i, encoder in enumerate(channel.encoders):
      # with the average gone, what's left of the mean reading is the read bias
      assert torch.allclose(
        readings[:, i, 0].mean(0), encoder.read.bias, atol=1e-3
      )


def test_without_head_biases_the_size_of_the_readings_cant_reach_routing():
  """A query is then a plain linear map of the readings, so scaling the
  readings leaves every cosine where it was."""
  model, data = phase5_model(COSINE, Changes(no_head_biases=True))
  (channel,) = model.channels
  with torch.no_grad():
    z = channel.read(features_of(model, data.joint.x[:32])).sent.readings
    heads = [(e.query, e.key) for e in channel.encoders]

    def weights(scale):
      q = torch.stack([h[0](z[:, i] * scale) for i, h in enumerate(heads)], 1)
      k = torch.stack([h[1](z[:, i] * scale) for i, h in enumerate(heads)], 1)
      return channel.mixer(q, k, torch.zeros(*q.shape[:-1], 64))[1]

    assert torch.allclose(weights(1), weights(5), atol=1e-5)
  for encoder in channel.encoders:
    assert not encoder.query.bias.requires_grad


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
