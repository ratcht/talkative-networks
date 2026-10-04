# 01 Multiplicative injection (FiLM)

## Question

Does injecting the received message multiplicatively (FiLM) work better than additively?

## Method

- `--decoder film`: `FiLMDecoder` outputs `gamma` and `beta` stacked on channels. `hook_film` applies `x * (1 + gamma) + beta` at the input of `layer1.2`.
- Last decoder layer zero-initialized, so step 0 equals the non-communicating ensemble.
- The additive decoder is the same with `gamma` fixed at 0.

## Setup

- Experts: `selection/manifests/encoder_study_group.json`
- Encoder `qkv`, `key_dim 16`, `value_dim 64`, taps `layer1.2` / `layer3.0`, K = 1
- 70 epochs, AdamW lr 1e-3, weight decay 1e-4
- Ensemble loss only, `fc` trainable, no early stopping

## Arms

| Arm | Patch | Seeds |
|---|---|---|
| `film_p0` | 0 | 5 done |
| `film_p2` | 2 | 3 done, 2044397375 interrupted, 755818579 not attempted |

## Results

| Arm | Val | Test | vs averaging (test) |
|---|---|---|---|
| `film_p0` | 0.6638 ± 0.0031 | 0.6588 ± 0.0018 | -0.64 |
| `film_p2` | 0.6551 ± 0.0030 | 0.6519 ± 0.0015 | -1.33 |
| `patch2` (additive, K = 2, experiment 00) | 0.6583 ± 0.0027 | 0.6520 ± 0.0016 | -1.32 |

Per seed:

| Arm | Seed | Aim run | Val | Test |
|---|---|---|---|---|
| film_p0 | 476981973 | 79fa37dfdbce4226941af405 | 0.6638 | 0.6607 |
| film_p0 | 621715752 | 166b33782c29449a91e79a97 | 0.6614 | 0.6588 |
| film_p0 | 1882162633 | 0de7d9d1c4694103a19927f6 | 0.6690 | 0.6576 |
| film_p0 | 2044397375 | bc4d1426ebe84e20b0310d43 | 0.6614 | 0.6565 |
| film_p0 | 755818579 | 3b05438c65484ae18907edf7 | 0.6634 | 0.6606 |
| film_p2 | 476981973 | ca59d909618f4f999d576c80 | 0.6522 | 0.6520 |
| film_p2 | 621715752 | b6fd259e80aa47a3858484f7 | 0.6582 | 0.6534 |
| film_p2 | 1882162633 | 2ce7e4478f94449cbfedb00c | 0.6550 | 0.6504 |

`film_p0` summary run: `63459527c0e2484591cb3baa`.

Training curves (mean over seeds):

- `film_p0`: train accuracy 0.666 at epoch 1, 0.780 at epoch 70. Val peaks at 0.6722 (epoch 14), 0.6638 at epoch 70.
- `film_p2`: train accuracy 0.661 at epoch 1, 0.874 at epoch 70. Val peaks at 0.6661 (epoch 3), 0.6551 at epoch 70.

Answer shift on val at epoch 70:

| Arm | Corrected | Corrupted | Agreement pre (epoch 1 to 70) |
|---|---|---|---|
| `film_p0` | 5.35% | 7.01% | 0.655 to 0.580 |
| `film_p2` | 4.95% | 7.96% | 0.647 to 0.629 |

## Findings

- Both arms score below untrained averaging on test.
- Both overfit: train accuracy rises through training while val peaks early and declines. `film_p2` overfits more (decoder about 20x larger).
- Communication corrupts more member answers than it corrects.
- Agreement between members before communication falls during training, because `fc` is trained only through the post-communication output.
- Additive vs FiLM is not answered. No additive pooled K = 1 run exists locally, and the additive patch 2 run used K = 2.
