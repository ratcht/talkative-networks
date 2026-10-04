# 00 Patch 2, additive decoder, K = 2

Run 2026-09-07, before this folder structure existed. Recorded here from Aim and the state file.

## Setup

- Experts: `selection/manifests/encoder_study_group.json`
- Encoder `qkv`, `key_dim 16`, `value_dim 64`
- Decoder `mlp` (additive), `--patch 2`, `--k-rounds 2`
- Taps `layer1.2` / `layer3.0`
- 70 epochs, AdamW lr 1e-3, weight decay 1e-4, ensemble loss only, `fc` trainable

## Status

- Seeds 476981973, 621715752, 1882162633: done
- Seed 2044397375: interrupted
- Seed 755818579: not attempted
- No checkpoints kept. The checkpoint filename at the time did not include the seed, so each seed overwrote the previous one.

## Results

| Seed | Aim run | Val | Test |
|---|---|---|---|
| 476981973 | e62fa140eef647d19311459a | 0.6614 | 0.6504 |
| 621715752 | 0ce62e9b5a6c4e3c80bbd9d7 | 0.6568 | 0.6519 |
| 1882162633 | f718a593469940c0ae0ce319 | 0.6566 | 0.6536 |
| mean ± std | | 0.6583 ± 0.0027 | 0.6520 ± 0.0016 |

- Below untrained averaging (test 0.6652) by 1.3 points.
- Run `cb7f3f393d65417abf5c3a39` is an earlier attempt at seed 621715752 with no final result.
