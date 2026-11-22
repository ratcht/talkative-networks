"""
The experts alone, at every stream width and number of expert epochs, and a
classifier on their streams (readout.py).

  uv run python -m studies.pacs.run experts [extra testbed.run flags]
  uv run python -m studies.pacs.run readout [extra testbed.run flags]
"""

import subprocess
import sys

from studies.pacs import DOMAINS, STUDY

PASS = 1  # tagged on every run; bump it when rerunning with other settings
ARCHS = ("stream16", "stream32", "stream64", "stream128")
EXPERT_EPOCHS = (30, 100)
READOUT_ARCHS = ("stream16", "stream64")
READOUT_EXPERT_EPOCHS = 100
EPOCHS = 30  # of the classifier's training

COMMON = (
  f"--trial-seeds 0 1 2 3 4 --tag study={STUDY} --tag pass={PASS}"
  f" --scheme per_dataset --experts {len(DOMAINS) - 1}"
)


def arms() -> list[str]:
  arms = []
  for target in DOMAINS:
    sources = " ".join(d for d in DOMAINS if d != target)
    for arch in ARCHS:
      for epochs in EXPERT_EPOCHS:
        # every expert on the target and on each source test set
        for test in DOMAINS:
          arms.append(
            f"--arm {target}-{arch}-e{epochs}-experts-on-{test}"
            f" --expert-dataset {sources} --joint-dataset {sources}"
            f" --test-dataset {test} --arch {arch} --expert-epochs {epochs}"
            f" --tag target={target} --tag arch={arch} --tag expert_epochs={epochs}"
          )
  return arms


def readout_arms() -> list[str]:
  arms = []
  for target in DOMAINS:
    sources = " ".join(d for d in DOMAINS if d != target)
    for arch in READOUT_ARCHS:
      for test in DOMAINS:
        arms.append(
          f"--arm {target}-{arch}-readout-on-{test}"
          f" --expert-dataset {sources} --joint-dataset {sources}"
          f" --test-dataset {test} --arch {arch}"
          f" --expert-epochs {READOUT_EXPERT_EPOCHS} --epochs {EPOCHS}"
          f" --tag target={target} --tag arch={arch}"
        )
  return arms


if __name__ == "__main__":
  jobs = {
    "experts": ("testbed.run", arms()),
    "readout": ("studies.pacs.readout", readout_arms()),
  }
  module, job = jobs[sys.argv[1]]
  # one process per arm, so an arm that crashes can't take the rest with it
  failed = []
  for arm in job:
    command = [sys.executable, "-m", module, *COMMON.split(), *arm.split()]
    if subprocess.run([*command, *sys.argv[2:]]).returncode != 0:
      print(f"FAILED: {arm}", flush=True)
      failed.append(arm)
  sys.exit("\n".join(["failed arms:", *failed]) if failed else 0)
