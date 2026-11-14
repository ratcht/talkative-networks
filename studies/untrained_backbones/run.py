"""
Run the untrained-backbones grid, one testbed.run call per arm:

  uv run python -m studies.untrained_backbones.run

  backbone   trained (10 expert epochs) | untrained (0: the experts as initialised)
  direction  forward stage1 -> stage2 | backward stage2 -> stage1, one round
  level      1 (linear_taper) | 3 (linear_taper_gelu), from the channel ladder
  mixer      every channel ladder mixer

An untrained expert is the network its trained counterpart started from: both
are initialised from derive(expert_seed, "init"), and the channel's seeds are
the same, so the two backbones compare trial seed by trial seed. Untrained
pools get their own cache key, since the budget is part of it.

Every run is tagged backbone, level and mixer, besides the arm, trial seed and
direction testbed.run adds. Both backbones log to one experiment, since the
v3 experiment's trained runs are 10 joint epochs, not 30.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # before torch sees a GPU

from studies.channel_ladder import LEVELS, MIXERS, WIDTH  # noqa: E402
from testbed.run import main as run_arm  # noqa: E402

STUDY = "untrained-backbones"
EXPERIMENT = "v3-untrained"
BACKBONES = {"trained": 10, "untrained": 0}  # name -> expert epochs
DIRECTIONS = {"fwd": ("stage1", "stage2"), "bwd": ("stage2", "stage1")}
STUDY_LEVELS = ("1", "3")
TRIAL_SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 30


def arms():
  for backbone, expert_epochs in BACKBONES.items():
    for direction, (read, write) in DIRECTIONS.items():
      for level in STUDY_LEVELS:
        for mixer in MIXERS:
          yield [
            "--arm", f"{backbone}-{direction}-{level}-{mixer}",
            "--exchange", read, write,
            "--encoder", LEVELS[level].encoder,
            "--decoder", LEVELS[level].decoder,
            "--mixer", mixer,
            "--width", str(WIDTH),
            "--expert-epochs", str(expert_epochs),
            "--epochs", str(EPOCHS),
            "--trial-seeds", *map(str, TRIAL_SEEDS),
            "--experiment", EXPERIMENT,
            "--tag", f"study={STUDY}",
            "--tag", f"backbone={backbone}",
            "--tag", f"level={level}",
            "--tag", f"mixer={mixer}",
          ]  # fmt: skip


if __name__ == "__main__":
  for argv in arms():
    run_arm(argv)
