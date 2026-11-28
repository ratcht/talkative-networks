"""The model-grafting study's runner (studies/model_grafting).

uv run pytest tests/test_model_grafting.py
"""

import copy

import pytest
import torch
from fakes import fake_data, random_experts

from studies.model_grafting import ARMS, EXPERIMENT
from studies.model_grafting.run import flags_for_testbed
from studies.model_grafting.stitch_fit import fit_stitches
from testbed.grafting import GraftConfig, build_graft
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


@pytest.mark.parametrize("fit", ["fitted", "fitted_frozen"])
def test_a_fitted_stitch_sets_the_plain_stitch_and_says_so(fit):
  flags = flags_for_testbed(["--graft-arm", "bypass", "--trial-seeds", "0", "--stitch-fit", fit])
  args = parse_args(flags)
  check_flags(args)
  assert args.graft_stitch == "pointwise_linear"
  assert f"stitch_fit={fit}" in args.tag and args.arm.endswith(f"-{fit}")
  with pytest.raises(SystemExit, match="fits the plain stitch"):
    flags_for_testbed(
      ["--graft-arm", "bypass", "--trial-seeds", "0", "--stitch-fit", fit,
       "--graft-stitch", "bn_pointwise_linear_bn"]
    )


def test_a_random_stitch_keeps_the_testbed_default():
  args = parse_args(flags_for_testbed(["--graft-arm", "bypass", "--trial-seeds", "0"]))
  assert args.graft_stitch is None and "stitch_fit=random" in args.tag


@pytest.mark.parametrize("arch", ["tiny_resnet", "tiny_vit"])
@pytest.mark.parametrize("read", ["embed", "block1"])
def test_the_fit_recovers_a_copy_of_the_anchor(arch, read):
  data = fake_data(n=200)
  experts = random_experts(3, arch)
  experts[1] = copy.deepcopy(experts[0])  # obligate 1 is the anchor itself
  config = GraftConfig(read, "block2", stitch="pointwise_linear")
  model = build_graft(experts, config, data.input_shape, seed=0)
  explained = fit_stitches(model, data, freeze=True)
  first = model.channel.stitch[0].map
  c = first.in_channels
  assert explained[0] == pytest.approx(1, abs=1e-6)
  assert torch.allclose(first.weight[:, :, 0], torch.eye(c), atol=1e-4)
  assert torch.allclose(first.bias, torch.zeros(c), atol=1e-4)
  assert 0 <= explained[1] <= 1
  assert not any(p.requires_grad for p in model.channel.stitch.parameters())
  x = data.test.x[:8]
  with torch.no_grad():  # the return still starts at zero: the anchor alone
    assert torch.equal(model.eval()(x).logits[:, 0], model.experts[0](x))


def test_the_fit_brings_each_stitch_nearer_the_obligates_own_stream():
  data = fake_data(n=200)
  model = build_graft(
    random_experts(3, "tiny_resnet"), GraftConfig("block1", "block2", stitch="pointwise_linear"),
    data.input_shape, seed=0,
  )
  x = data.joint.x[:64]

  def error():
    with torch.no_grad():
      u = model.experts[0].run(x, until="block1")
      return [
        (stitch(u) - obligate.run(x, until="block1")).pow(2).mean().item()
        for stitch, obligate in zip(model.channel.stitch, model.experts[1:])
      ]

  before = error()
  fit_stitches(model, data)
  assert all(a < b for a, b in zip(error(), before))
  assert all(p.requires_grad for p in model.channel.stitch.parameters())
