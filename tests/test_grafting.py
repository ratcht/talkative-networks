"""
Construction checks for model grafting (testbed/grafting.py,
docs/testbed/grafting.md): the start is the anchor alone, the obligates run
only their blocks between the two points, the class token goes through both
stitches, the stitches learn from the second update, the sizes, and a short
training run whose every logged key is in the registry.

  uv run pytest tests/test_grafting.py
"""

import math

import pytest
import torch
from fakes import fake_data, random_experts

from testbed.grafting import GraftChannel, GraftConfig, build_graft
from testbed.metrics.registry import find
from testbed.run import check_flags, parse_args
from testbed.train import JointConfig, train

ARCHS = ("tiny_resnet", "tiny_vit")
CONFIGS = {
  "attention": GraftConfig("block1", "block2"),
  "mean": GraftConfig("block1", "block2", mixer="mean"),
  "bypass": GraftConfig("block1", "block2", bypass=True),
}


def graft(arch="tiny_resnet", config=CONFIGS["attention"], m=3):
  experts = random_experts(m, arch)
  return build_graft(experts, config, fake_data().input_shape, seed=0)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("name", CONFIGS)
def test_before_training_the_graft_is_the_anchor_alone(arch, name):
  model = graft(arch, CONFIGS[name]).eval()
  x = fake_data().test.x[:16]
  with torch.no_grad():
    out = model(x)
    alone = model.experts[0](x)
  assert torch.equal(out.logits[:, 0], alone)
  assert torch.equal(model.baseline(x).logits[:, 0], alone)
  assert out.logits.shape == (16, 1, alone.shape[-1])


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("bypass", [False, True])
def test_obligates_run_only_their_blocks_between_the_points(arch, bypass):
  # in training, where only the answer's path runs; evaluation also runs the rest
  # of each obligate, for stitched_accuracy alone
  model = graft(arch, GraftConfig("block1", "block2", bypass=bypass)).train()
  calls: dict[tuple[int, str], int] = {}
  for i, expert in enumerate(model.experts):
    for name, stage in expert.stages.items():
      stage.register_forward_hook(
        lambda *_, key=(i, name): calls.__setitem__(key, calls.get(key, 0) + 1)
      )
  with torch.no_grad():
    model(fake_data().test.x[:4])
  names = list(model.experts[0].stages)
  assert all(calls.get((0, n)) == 1 for n in names)  # the anchor runs every stage once
  for i in range(1, len(model.experts)):
    ran = [n for n in names if calls.get((i, n))]
    assert ran == ([] if bypass else ["block2"])


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("bypass", [False, True])
def test_the_obligates_answers_are_measured_never_used(arch, bypass):
  model = graft(arch, GraftConfig("block1", "block2", bypass=bypass))
  x = fake_data().test.x[:8]
  with torch.no_grad():
    for p in model.channel.out.parameters():  # open the return
      p.normal_()
    model.eval()
    before = model(x)
    head = list(model.experts[1].stages.values())[-1]
    for p in head.parameters():  # an obligate's head: measured, never in the answer
      p.add_(torch.randn_like(p))
    after = model(x)
    assert torch.equal(before.logits, after.logits)
    answers = after.traces[0].obligate_answers
    assert answers is not None and answers.shape == (8, len(model.experts) - 1)
    model.train()
    assert model(x).traces[0].obligate_answers is None


def test_reply_dropout_acts_in_training_only():
  channel = GraftChannel(GraftConfig(dropout=0.5), (8, 4, 4), 2)
  with torch.no_grad():
    for p in channel.out.parameters():
      p.normal_()
  anchor, replies = torch.randn(4, 8, 4, 4), [torch.randn(4, 8, 4, 4) for _ in range(2)]
  channel.train()
  a, b = (channel.mix(anchor, replies).injection for _ in range(2))
  assert not torch.equal(a, b)
  channel.eval()
  a, b = (channel.mix(anchor, replies).injection for _ in range(2))
  assert torch.equal(a, b)
  with pytest.raises(ValueError, match="dropout"):
    graft(config=GraftConfig("block1", "block2", dropout=1.0))


def test_joint_augment_and_weight_decay_train_and_log_the_same_keys():
  data = fake_data(n=200)
  plain = train(graft(), data, JointConfig(epochs=1, batch_size=50), 0, Recorder())
  extra = train(
    graft(), data, JointConfig(epochs=1, batch_size=50, augment=True, weight_decay=0.05), 0, Recorder()
  )
  assert [set(s) for s in plain] == [set(s) for s in extra]
  assert extra[0]["test/accuracy"] == plain[0]["test/accuracy"]  # the same start


def test_the_class_token_goes_through_both_stitches():
  model = graft("tiny_vit").eval()
  with torch.no_grad():
    for p in model.channel.out.parameters():  # open the return
      p.normal_()
    x = fake_data().test.x[:4]
    at_read = model.experts[0].run(x, until="block1")
    moved = at_read.clone()
    moved[:, :, 0] += 1  # only the class token
    stitch = model.channel.stitch[0]
    assert not torch.allclose(stitch(at_read)[:, :, 0], stitch(moved)[:, :, 0])
    assert torch.equal(stitch(at_read)[:, :, 1:], stitch(moved)[:, :, 1:])
    out = model(x)
    assert out.traces[0].mixed.injection[:, :, 0].abs().sum() > 0


def test_the_stitches_learn_from_the_second_update():
  model = graft().train()
  data = fake_data()
  opt = torch.optim.Adam(model.trainable.parameters(), lr=1e-2)
  stitch = model.channel.stitch[0].map.weight
  norms = []
  for step in range(2):
    opt.zero_grad()
    out = model(data.joint.x[:32])
    torch.nn.functional.nll_loss(out.log_probs, data.joint.y[:32]).backward()
    norms.append(0.0 if stitch.grad is None else stitch.grad.norm().item())
    opt.step()
  assert (
    norms[0] == 0
  )  # the return's output map is still zero: nothing reaches the stitch
  assert norms[1] > 0


def test_mean_has_no_query_or_key():
  channel = GraftChannel(GraftConfig(mixer="mean"), (8, 4, 4), 2)
  assert not hasattr(channel, "query") and not hasattr(channel, "key")


@pytest.mark.parametrize(
  "shape, attention, mean",
  [((64, 16, 16), 33_792, 25_472), ((96, 65), 74_496, 55_872)],
)
def test_sizes_of_the_v4_channels(shape, attention, mean):
  # docs/testbed/grafting.md §5: a group of three, so two obligates
  def size(mixer):
    channel = GraftChannel(GraftConfig(mixer=mixer), shape, 2)
    return sum(p.numel() for p in channel.parameters())

  assert size("attention") == attention
  assert size("mean") == mean


def test_heads_follow_the_stream_width():
  assert GraftChannel(GraftConfig(), (64, 16, 16), 2).heads == 2
  assert GraftChannel(GraftConfig(), (96, 65), 2).heads == 3


def test_points_must_be_in_order():
  with pytest.raises(ValueError, match="after it reads"):
    graft(config=GraftConfig("block2", "block1"))


class Recorder:
  def __init__(self):
    self.steps: dict[int, dict[str, float]] = {}

  def log_metrics(self, metrics, step=None):
    self.steps.setdefault(step, {}).update(metrics)


@pytest.fixture(scope="module", params=[(a, n) for a in ARCHS for n in CONFIGS])
def run(request):
  arch, name = request.param
  data = fake_data(n=400)
  model = build_graft(random_experts(3, arch), CONFIGS[name], data.input_shape, 0)
  history = train(model, data, JointConfig(epochs=2, batch_size=50), 0, Recorder())
  return name, model, data, history


def test_every_logged_key_is_in_the_registry(run):
  _, _, _, history = run
  assert not {key for step in history for key in step if find(key) is None}


def test_no_metric_that_compares_experts_answers_is_logged(run):
  _, _, _, history = run
  for step in history:
    for key in step:
      assert not key.split("/")[1:2] == ["answer_grid"], key
      assert "expert_accuracy" not in key and "marginal_answer_shift" not in key
      assert "agreement" not in key and "overlap" not in key


def test_step_zero_is_the_anchor_alone(run):
  _, model, data, history = run
  with torch.no_grad():
    alone = (model.experts[0](data.test.x).argmax(-1) == data.test.y).float().mean()
  first = history[0]
  assert first["test/accuracy"] == pytest.approx(alone.item())
  assert first["test/joint_answer_shift/net"] == 0
  assert first["msg_norm/injection/ex0"] == 0
  assert len(history) == 3


def test_stitched_accuracy_is_logged_for_every_obligate(run):
  _, model, _, history = run
  obligates = len(model.experts) - 1
  for step in history:
    each = [v for k, v in step.items() if k.startswith("stitched_accuracy/expert")]
    assert len(each) == obligates and all(0 <= v <= 1 for v in each)
    assert step["stitched_accuracy/ex0"] == pytest.approx(sum(each) / obligates)


def test_attention_shares_are_a_split_of_one(run):
  name, _, _, history = run
  for step in history:
    heard = [v for k, v in step.items() if k.startswith("attn_heard/")]
    if name == "mean":
      assert not heard
      continue
    assert len(heard) == 2 and sum(heard) == pytest.approx(1)
    assert 0.5 - 1e-6 <= step["attn_top1/ex0"] <= 1 + 1e-6
    assert -1e-9 <= step["attn_entropy/ex0"] <= math.log(2) + 1e-6


def test_graft_flags_need_graft():
  with pytest.raises(SystemExit, match="need --graft"):
    check_flags(parse_args(["--arm", "x", "--trial-seeds", "0", "--graft-bypass"]))
  with pytest.raises(SystemExit, match="need --graft"):
    check_flags(parse_args(["--arm", "x", "--trial-seeds", "0", "--graft-dropout", "0.1"]))
  with pytest.raises(SystemExit, match="its own model"):
    check_flags(
      parse_args(
        [
          "--arm",
          "x",
          "--trial-seeds",
          "0",
          "--graft",
          "--exchange",
          "block1",
          "block2",
        ]
      )
    )
