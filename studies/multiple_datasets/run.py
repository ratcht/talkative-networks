"""
The whole study: the expert table, the pooled reference, then the channel
trained on target vs source images.

  uv run python -m studies.multiple_datasets.run [extra testbed.run flags, e.g. --epochs 1]
"""

import subprocess
import sys

from studies.multiple_datasets import EXPERTS, STUDY, TARGET

# caps: a dataset with fewer images than its cap gives all it has
EXPERT_SAMPLES = 3600  # per expert dataset
JOINT_SAMPLES = 2900  # per source dataset; the target gives three times this
EPOCHS = 30  # of joint training
PASS = 2  # tagged on every run; bump it when rerunning with other settings

SOURCES = " ".join(EXPERTS)
COMMON = (
  f"--trial-seeds 0 1 2 3 4 --tag study={STUDY} --expert-dataset {SOURCES}"
  f" --epochs {EPOCHS} --tag pass={PASS}"
)
PER_DATASET = f"--scheme per_dataset --expert-samples {EXPERT_SAMPLES} --experts {len(EXPERTS)}"
CHANNEL = "--encoder linear_taper_gelu --decoder linear_taper_gelu"

JOINT = {
  "target": f"--joint-dataset {TARGET} --joint-samples {JOINT_SAMPLES * len(EXPERTS)}",
  "source": f"--joint-dataset {SOURCES} --joint-samples {JOINT_SAMPLES}",
}
EXCHANGE = {"fwd": "stage1 stage2", "bwd": "stage2 stage1"}
LOSSES = ("ensemble", "per_expert")
MIXERS = ("attention", "rmsnorm_attention", "self", "mean")
# cnn3_large is about the size of the three experts and their channel
POOLED_ARCHS = ("cnn3", "cnn3_large")
POOLED_EPOCHS = (10, 30)  # each pooled model is trained once per value
# the largest gate has more trained weights than the channel
GATE_READS = ("stage1", "stage2", "stage3")
GATE_WIDTHS = (8, 32)


def arms() -> list[str]:
  arms = []

  # every expert on the target and on each source test set
  for test in (TARGET, *EXPERTS):
    arms.append(
      f"--arm experts-on-{test} {PER_DATASET} --joint-dataset {TARGET} --test-dataset {test}"
    )

  # one model trained on everything the three experts were trained on, then
  # on that and the images each setting's channel is trained on
  pooled = {"experts-data": f"--scheme seed_only --joint-dataset {TARGET}"} | {
    f"{setting}-data": f"--scheme expert_and_joint {joint}"
    for setting, joint in JOINT.items()
  }
  for arch in POOLED_ARCHS:
    for trained_on, flags in pooled.items():
      for epochs in POOLED_EPOCHS:
        arms.append(
          f"--arm pooled-{arch}-{trained_on}-{epochs}ep --arch {arch} {flags}"
          f" --expert-samples {EXPERT_SAMPLES} --experts 1 --expert-epochs {epochs}"
          f" --test-dataset {TARGET} --tag arch={arch} --tag trained_on={trained_on}"
        )

  for setting, joint in JOINT.items():
    data = f"{PER_DATASET} {joint} --test-dataset {TARGET} --tag setting={setting}"
    for read in GATE_READS:
      for width in GATE_WIDTHS:
        arms.append(
          f"--arm {setting}-moe-{read}-w{width} {data} --gate linear_taper_gelu"
          f" --gate-read {read} --width {width}"
        )
    for loss in LOSSES:
      for direction, exchange in EXCHANGE.items():
        for mixer in MIXERS:
          arms.append(
            f"--arm {setting}-{loss}-{direction}-{mixer} {data} {CHANNEL}"
            f" --exchange {exchange} --mixer {mixer} --loss {loss}"
            f" --tag loss={loss} --tag dir={direction} --tag mixer={mixer}"
          )
  return arms


if __name__ == "__main__":
  # one process per arm, so an arm that crashes can't take the rest with it
  failed = []
  for arm in arms():
    command = [sys.executable, "-m", "testbed.run", *COMMON.split(), *arm.split()]
    if subprocess.run([*command, *sys.argv[1:]]).returncode != 0:
      print(f"FAILED: {arm}", flush=True)
      failed.append(arm)
  sys.exit("\n".join(["failed arms:", *failed]) if failed else 0)
