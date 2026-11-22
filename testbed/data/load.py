"""Datasets from the Hugging Face Hub, split and normalised."""

import hashlib
from pathlib import Path

import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from testbed.seeds import generator
from testbed.data.split import (
  Data,
  Split,
  concat,
  partition_digest,
  stratified_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def rgb32(images: torch.Tensor) -> torch.Tensor:
  return F.interpolate(images, size=32, mode="bilinear").expand(-1, 3, -1, -1)


def size64(images: torch.Tensor) -> torch.Tensor:
  return F.interpolate(images, size=64, mode="bilinear", antialias=True)


crop_flip = K.AugmentationSequential(
  K.RandomCrop((64, 64), padding=4), K.RandomHorizontalFlip()
)

DATASETS = {
  "fashion": {"path": "zalando-datasets/fashion_mnist"},
  "mnist": {"path": "ylecun/mnist", "transform": rgb32},
  "svhn": {"path": "ufldl-stanford/svhn", "name": "cropped_digits"},
  "usps": {"path": "flwrlabs/usps", "transform": rgb32},
  "mnist_m": {"path": "Mike0307/MNIST-M"},
  **{
    f"pacs_{domain}": {
      "path": "flwrlabs/pacs",
      "domain": domain,
      "test_frac": 0.2,
      "transform": size64,
      "augment": crop_flip,
    }
    for domain in ("art_painting", "cartoon", "photo", "sketch")
  },
}


def apply_transform(entry: dict, images: torch.Tensor) -> torch.Tensor:
  if "transform" in entry:
    images = entry["transform"](images)
  return images


def load_one(
  name: str, fracs: tuple[float, float, float], split_seed: int, root: Path
) -> dict:
  entry = DATASETS[name]
  dataset = load_dataset(entry["path"], entry.get("name"), cache_dir=str(root))
  if "domain" in entry:
    dataset = dataset.filter(lambda d: d == entry["domain"], input_columns="domain")
  if "test_frac" in entry:
    dataset = dataset["train"].train_test_split(
      entry["test_frac"], stratify_by_column="label", seed=split_seed
    )
  train = dataset["train"].with_format("torch")[:]
  test = dataset["test"].with_format("torch")[:]
  parts = stratified_split(train["label"].numpy(), fracs, split_seed)
  images = apply_transform(entry, train["image"].float().div(255))
  test_images = apply_transform(entry, test["image"].float().div(255))
  return {
    "num_classes": dataset["train"].features["label"].num_classes,
    **{k: Split(images[idx], train["label"][idx]) for k, idx in parts.items()},
    "test": Split(test_images, test["label"]),
    "digest": partition_digest(name, parts),
  }


def trim(split: Split, samples: int | None, seed: int) -> Split:
  keep = torch.randperm(len(split), generator=generator(seed))[:samples]
  keep = keep.sort().values
  return Split(split.x[keep], split.y[keep])


def load(
  expert_ds: list[str],
  joint_ds: list[str],
  test_ds: list[str],
  expert_samples: int | None = None,
  joint_samples: int | None = None,
  fracs: tuple[float, float, float] = (0.5, 0.4, 0.1),
  split_seed: int = 0,
  root: Path = REPO_ROOT / "data",
  device: str = "cuda",
) -> Data:
  datasets = {
    name: load_one(name, fracs, split_seed, root)
    for name in dict.fromkeys(expert_ds + joint_ds + test_ds)
  }
  expert = {
    name: trim(datasets[name]["expert"], expert_samples, split_seed)
    for name in expert_ds
  }
  joint = [
    trim(datasets[name]["joint"], joint_samples, split_seed) for name in joint_ds
  ]
  joint_val = [datasets[name]["joint_val"] for name in test_ds]
  test = [datasets[name]["test"] for name in test_ds]

  expert_sizes = {name: len(split) for name, split in expert.items()}
  joint_sizes = {name: len(split) for name, split in zip(joint_ds, joint)}
  print(f"expert images {expert_sizes}, joint images {joint_sizes}")

  expert_x = torch.cat([split.x for split in expert.values()])
  mean, std = expert_x.mean(), expert_x.std()

  def concat_splits(splits: list[Split]) -> Split:
    split = concat(splits)
    return Split((split.x - mean) / std, split.y)

  num_classes = datasets[expert_ds[0]]["num_classes"]
  assert all(dataset["num_classes"] == num_classes for dataset in datasets.values())

  augments = [DATASETS[name].get("augment") for name in expert_ds]
  assert all(augment is augments[0] for augment in augments)

  digests = [dataset["digest"] for dataset in datasets.values()]
  key = repr((expert_ds, joint_ds, test_ds, expert_samples, joint_samples, digests))
  return Data(
    name="+".join(expert_ds),
    num_classes=num_classes,
    expert=concat_splits(list(expert.values())),
    joint=concat_splits(joint),
    joint_val=concat_splits(joint_val),
    test=concat_splits(test),
    digest=hashlib.sha256(key.encode()).hexdigest()[:16],
    expert_parts={name: concat_splits([split]) for name, split in expert.items()},
    device=device,
    augment=augments[0] or nn.Identity(),
  )
