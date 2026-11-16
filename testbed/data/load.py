"""Datasets from the Hugging Face Hub, split and normalised."""

from pathlib import Path

from datasets import load_dataset

from testbed.data.split import Data, Split, partition_digest, stratified_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = {"fashion": "zalando-datasets/fashion_mnist", "mnist": "ylecun/mnist"}


def load(
  name: str = "fashion",
  fracs: tuple[float, float, float] = (0.5, 0.4, 0.1),
  split_seed: int = 0,
  root: Path = REPO_ROOT / "data",
  device: str = "cuda",
) -> Data:
  ds = load_dataset(DATASETS[name], cache_dir=str(root)).with_format("torch")
  train, test = ds["train"][:], ds["test"][:]
  parts = stratified_split(train["label"].numpy(), fracs, split_seed)

  images = train["image"].float().div(255)
  mean, std = images[parts["expert"]].mean(), images[parts["expert"]].std()
  images = (images - mean) / std
  test_images = (test["image"].float().div(255) - mean) / std

  return Data(
    name=name,
    num_classes=ds["train"].features["label"].num_classes,
    **{k: Split(images[idx], train["label"][idx]) for k, idx in parts.items()},
    test=Split(test_images, test["label"]),
    digest=partition_digest(name, parts),
    device=device,
  )
