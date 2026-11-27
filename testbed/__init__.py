"""
v4 testbed: frozen experts that communicate through their residual streams.

The pipeline, one module per stage:

  data.py                   datasets as GPU tensors, split into expert / joint / joint_val / test
  experts/architectures.py  residual-stream experts (ResNet, ViT) as named stages; the
                            stream after each stage is where a channel reads and writes
  experts/recipes.py        each family's training recipe: optimiser, schedule, augmentation
  experts/schemes.py        how a pool of experts comes to differ (seeds, class skew, NCL, ...)
  experts/pool.py           pools cached on disk, keyed by everything that made them
  selection.py              which experts of a pool form the group
  communication/channel.py  encoder → mixer → decoder, reading and writing the stream
  communication/ensemble.py runs the experts with forward or backward exchanges
  train.py                  trains the channel; experts stay frozen
  metrics/                  what a run measures (docs/testbed/metrics.md)
  tracking.py               MLflow behind a small interface
  moe.py                    the soft-gated MoE baseline: a gate weighs the experts' answers
  run.py                    one arm of a study over a list of trial seeds
  results.py                each run as one number: the median around its best joint_val epoch
  merge_runs.py             copies finished runs from one MLflow database into another

Randomness comes only from named seed streams (seeds.py), so every arm run on
the same trial seeds sees the same experts and the same groups.

The testbed knows no study. What is specific to one study, such as its arms,
its run labels and its defaults, lives in that study's package in studies/.
"""
