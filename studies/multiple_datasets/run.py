"""
The whole study: the expert table, the pooled reference, then the channel
trained on target vs source images.

  uv run python -m studies.multiple_datasets.run [extra testbed.run flags, e.g. --epochs 1]
"""

import subprocess
import sys

from studies.multiple_datasets import EXPERTS, STUDY, TARGET

SOURCES = " ".join(EXPERTS)
COMMON = f"--trial-seeds 0 1 2 3 4 --tag study={STUDY} --expert-dataset {SOURCES}"

# caps: a dataset with fewer images than its cap gives all it has
EXPERT_SAMPLES = 3600  # per expert dataset
JOINT_SAMPLES = (
  2000  # per source dataset; the target gives as many as the sources together
)

PER_DATASET = f"--scheme per_dataset --expert-samples {EXPERT_SAMPLES} --experts 3"
CHANNEL = "--encoder linear_taper_gelu --decoder linear_taper_gelu"

JOINT = {
  "target": f"--joint-dataset {TARGET} --joint-samples {JOINT_SAMPLES * len(EXPERTS)}",
  "source": f"--joint-dataset {SOURCES} --joint-samples {JOINT_SAMPLES}",
}
EXCHANGE = {"fwd": "stage1 stage2", "bwd": "stage2 stage1"}
LOSSES = ("ensemble", "per_expert")
MIXERS = ("attention", "rmsnorm_attention", "self", "mean")


def arms() -> list[str]:
  arms = []

  # every expert on the target and on each source test set
  for test in (TARGET, *EXPERTS):
    arms.append(
      f"--arm experts-on-{test} {PER_DATASET} --joint-dataset {TARGET} --test-dataset {test}"
    )

  # one model trained on everything the three experts were trained on
  arms.append(
    f"--arm pooled --scheme seed_only --expert-samples {EXPERT_SAMPLES} --experts 1"
    f" --joint-dataset {TARGET} --test-dataset {TARGET}"
  )

  for setting, joint in JOINT.items():
    data = f"{PER_DATASET} {joint} --test-dataset {TARGET} --tag setting={setting}"
    arms.append(f"--arm {setting}-moe {data} --gate linear_taper_gelu")
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
  for arm in arms():
    command = [sys.executable, "-m", "testbed.run", *COMMON.split(), *arm.split()]
    if subprocess.run([*command, *sys.argv[1:]]).returncode != 0:
      print(f"FAILED: {arm}", flush=True)
