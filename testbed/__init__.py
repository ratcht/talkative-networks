"""
v3 testbed: frozen experts that communicate.

The pipeline, one module per stage:

  data.py                   datasets as GPU tensors, split into expert / joint / joint_val / test
  experts/architectures.py  experts as named stages; the boundaries are the read/write points
  experts/schemes.py        how a pool of experts comes to differ (seeds, class skew, NCL, ...)
  experts/pool.py           pools cached on disk, keyed by everything that made them
  selection.py              which experts of a pool form the group
  communication/channel.py  encoder → mixer → decoder, with full spatial access
  communication/ensemble.py runs the experts with forward or backward exchanges
  train.py                  trains the channel; experts stay frozen
  metrics/                  what a run measures (docs/testbed/metrics.md)
  tracking.py               MLflow behind a small interface
  run.py                    one arm of a study over a list of trial seeds

Randomness comes only from named seed streams (seeds.py), so every arm run on
the same trial seeds sees the same experts and the same groups.
"""
