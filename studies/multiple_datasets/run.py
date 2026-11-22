"""
The whole study (docs/studies/digits-five-v2.md): the expert table, the
pooled models, then the gate and the channel trained on target vs source
images.

  uv run python -m studies.multiple_datasets.run [extra testbed.run flags, e.g. --epochs 1]
"""

import subprocess
import sys

from studies.multiple_datasets import EXPERTS, STUDY, TARGET

# caps: a dataset with fewer images than its cap gives all it has
EXPERT_SAMPLES = 3600  # per expert dataset
JOINT_SAMPLES = (2900, 400)  # per source dataset; the target gives three times this
EPOCHS = 30  # of joint training, and of a pooled model's training on the joint images
PASS = 1  # tagged on every run; bump it when rerunning with other settings

SOURCES = " ".join(EXPERTS)
COMMON = (
  f"--trial-seeds 0 1 2 3 4 --tag study={STUDY} --expert-dataset {SOURCES}"
  f" --epochs {EPOCHS} --tag pass={PASS}"
)
PER_DATASET = f"--scheme per_dataset --expert-samples {EXPERT_SAMPLES} --experts {len(EXPERTS)}"
CHANNEL = "--encoder linear_taper_gelu --decoder linear_taper_gelu"

EXCHANGE = {"fwd": "stage1 stage2", "bwd": "stage2 stage1"}
LOSSES = ("ensemble", "per_expert")
MIXERS = ("rmsnorm_attention", "self", "mean")
TAPERS = (8, 4, 2)
POOLED_ARCHS = ("cnn3", "cnn3_large")
GATE_READS = ("stage1", "stage2", "stage3")
GATE_WIDTHS = (8, 32)


def joint_images() -> dict[str, str]:
  """The joint flags of every setting at every number of images, by a name
  such as target8700."""
  out = {}
  for samples in JOINT_SAMPLES:
    images = samples * len(EXPERTS)
    out[f"target{images}"] = (
      f"--joint-dataset {TARGET} --joint-samples {images}"
      f" --tag setting=target --tag images={images}"
    )
    out[f"source{images}"] = (
      f"--joint-dataset {SOURCES} --joint-samples {samples}"
      f" --tag setting=source --tag images={images}"
    )
  return out


def pooled(arch: str) -> str:
  return (
    f"--arch {arch} --expert-samples {EXPERT_SAMPLES} --experts 1"
    f" --test-dataset {TARGET} --tag arch={arch}"
  )


def pooled_arms() -> list[str]:
  """The same model, trained further on the images each channel is trained on
  (pooled.py)."""
  return [
    f"--arm pooled-{arch}-{name} {pooled(arch)} {joint}"
    for arch in POOLED_ARCHS
    for name, joint in joint_images().items()
  ]


def arms() -> list[str]:
  arms = []

  # every expert on the target and on each source test set
  for test in (TARGET, *EXPERTS):
    arms.append(
      f"--arm experts-on-{test} {PER_DATASET} --joint-dataset {TARGET} --test-dataset {test}"
    )

  # one model trained on the images the three experts were trained on
  for arch in POOLED_ARCHS:
    arms.append(f"--arm pooled-{arch} {pooled(arch)} --scheme seed_only --joint-dataset {TARGET}")

  for name, joint in joint_images().items():
    data = f"{PER_DATASET} {joint} --test-dataset {TARGET}"
    for read in GATE_READS:
      for width in GATE_WIDTHS:
        arms.append(
          f"--arm {name}-moe-{read}-w{width} {data} --gate linear_taper_gelu"
          f" --gate-read {read} --width {width}"
        )
    for loss in LOSSES:
      for direction, exchange in EXCHANGE.items():
        for mixer in MIXERS:
          for taper in TAPERS:
            arms.append(
              f"--arm {name}-{loss}-{direction}-{mixer}-r{taper} {data} {CHANNEL}"
              f" --exchange {exchange} --mixer {mixer} --loss {loss} --width {taper}"
              f" --tag loss={loss} --tag dir={direction} --tag mixer={mixer}"
              f" --tag taper={taper}"
            )
  return arms


if __name__ == "__main__":
  # one process per arm, so an arm that crashes can't take the rest with it
  failed = []
  modules = [("testbed.run", arm) for arm in arms()]
  modules += [("studies.multiple_datasets.pooled", arm) for arm in pooled_arms()]
  for module, arm in modules:
    command = [sys.executable, "-m", module, *COMMON.split(), *arm.split()]
    if subprocess.run([*command, *sys.argv[1:]]).returncode != 0:
      print(f"FAILED: {arm}", flush=True)
      failed.append(arm)
  sys.exit("\n".join(["failed arms:", *failed]) if failed else 0)
