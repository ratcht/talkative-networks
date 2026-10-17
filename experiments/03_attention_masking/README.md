# 03 Attention masking

## Question

- Is the channel communicating, or are receivers mostly reading their own message back?
- Additive or multiplicative (FiLM) injection, compared under the same setup?
- Can receivers use messages better if the blocks right after the injection point are trainable?

## Changes

- `--attn {all,others,self}`: mask on the bus attention.
  - `all`: each receiver attends to every member, itself included (previous behaviour).
  - `others`: self is masked out. Any gain must come from other members.
  - `self`: only self. No information from other members, so this is a self-loop control.
- `--unfreeze layer1.2 layer1.3 layer1.4`: the 3 blocks that first process the injected message become trainable (conv weights and BatchNorm scale/shift, 42,048 parameters total). BatchNorm running stats stay fixed.

## Setup

- Experts: `selection/manifests/encoder_study_group.json`
- Encoder `qkv`, `key_dim 16`, `value_dim 64`, taps `layer1.2` / `layer3.0`, patch 0, K = 1
- `--freeze-fc`, `--patience 10`, no member loss
- 70 epochs max, AdamW lr 1e-3, weight decay 1e-4
- First 3 joint seeds: 476981973, 621715752, 1882162633
- Trainable parameters: `mlp` 34,320, `film` 37,440, plus 42,048 with `--unfreeze`

## Arms

| Arm | Decoder | Attention | Unfreeze | K |
|---|---|---|---|---|
| `mask_mlp_all` | mlp | all | | 1 |
| `mask_mlp_others` | mlp | others | | 1 |
| `mask_mlp_self` | mlp | self | | 1 |
| `mask_film_all` | film | all | | 1 |
| `mask_film_others` | film | others | | 1 |
| `mask_film_self` | film | self | | 1 |
| `mask_mlp_all_unfreeze` | mlp | all | yes | 1 |
| `mask_mlp_others_unfreeze` | mlp | others | yes | 1 |
| `mask_film_all_unfreeze` | film | all | yes | 1 |
| `mask_film_others_unfreeze` | film | others | yes | 1 |
| `mask_k0_unfreeze` | | | yes | 0 |

## Baselines

- Arms without unfreeze: untrained averaging, val 0.6686, test 0.6652. With `fc` frozen this equals the K = 0 control.
- Arms with unfreeze: `mask_k0_unfreeze`. The unfrozen blocks can improve accuracy without messages, so the communication effect is `(arm) - (k0_unfreeze)`.

## How to read it

| Result | Meaning |
|---|---|
| `others` above baseline, `self` near baseline | members use information from each other |
| `self` close to `all`, `others` near baseline | the gain in `all` is a self-loop, not communication |
| unfreeze arms above `k0_unfreeze`, frozen arms near baseline | frozen receivers cannot use messages, trainable ones can |
| `film` above `mlp` within the same attention mode | multiplicative injection is better |

## Results

`mask_k0_unfreeze` failed on all 3 seeds at the end of epoch 1: `KeyError: 'self_attn'` in `_track_routing` (`ensemble/train.py`). With K = 0 there is no attention, so the routing counts have no `self_attn` entry. The unfreeze arms have no control yet.

| Arm | Val | Test | vs averaging (test) | Best epoch |
|---|---|---|---|---|
| `mask_mlp_all` | 0.6724 ± 0.0021 | 0.6683 ± 0.0026 | +0.31 | 2, 11, 15 |
| `mask_mlp_others` | 0.6717 ± 0.0007 | 0.6666 ± 0.0028 | +0.14 | 3, 1, 3 |
| `mask_mlp_self` | 0.6713 ± 0.0017 | 0.6658 ± 0.0018 | +0.06 | 1, 1, 3 |
| `mask_film_all` | 0.6709 ± 0.0009 | 0.6664 ± 0.0026 | +0.12 | 4, 2, 11 |
| `mask_film_others` | 0.6703 ± 0.0005 | 0.6679 ± 0.0017 | +0.27 | 3, 4, 5 |
| `mask_film_self` | 0.6699 ± 0.0005 | 0.6637 ± 0.0028 | -0.15 | 9, 12, 5 |
| `mask_mlp_all_unfreeze` | 0.6721 ± 0.0010 | 0.6663 ± 0.0014 | +0.11 | 1, 3, 3 |
| `mask_mlp_others_unfreeze` | 0.6721 ± 0.0013 | 0.6650 ± 0.0025 | -0.02 | 6, 3, 1 |
| `mask_film_all_unfreeze` | 0.6715 ± 0.0029 | 0.6649 ± 0.0021 | -0.03 | 14, 4, 9 |
| `mask_film_others_unfreeze` | 0.6711 ± 0.0007 | 0.6654 ± 0.0012 | +0.02 | 9, 2, 1 |
| `mask_k0_unfreeze` | failed | failed | | |

Val is optimistic, since the epoch is selected on val. Compare on test. The unfreeze arms are compared to averaging here only because `mask_k0_unfreeze` is missing.

Per seed:

| Arm | Seed | Aim run | Val | Test |
|---|---|---|---|---|
| mask_mlp_all | 476981973 | e27de36bb98f42fa9b11af70 | 0.6748 | 0.6709 |
| mask_mlp_all | 621715752 | 7ccbf8e18f6e4ec3a9a7d6a8 | 0.6696 | 0.6647 |
| mask_mlp_all | 1882162633 | 41dac7d34fac4804ac5de0cf | 0.6728 | 0.6693 |
| mask_mlp_others | 476981973 | 8e2125579c4c400c834ca71f | 0.6708 | 0.6649 |
| mask_mlp_others | 621715752 | 63845834fda74689b2f7ab0b | 0.6726 | 0.6644 |
| mask_mlp_others | 1882162633 | 88d2d9cbf0544ed68a1e5744 | 0.6716 | 0.6705 |
| mask_mlp_self | 476981973 | ce0a6814671141c887b88866 | 0.6722 | 0.6644 |
| mask_mlp_self | 621715752 | 5677627145dd47c984fbe8a2 | 0.6728 | 0.6646 |
| mask_mlp_self | 1882162633 | 0d37bddbc58b4631bd47eb16 | 0.6690 | 0.6684 |
| mask_film_all | 476981973 | bb386651271d4be189ab34cf | 0.6720 | 0.6650 |
| mask_film_all | 621715752 | 5dfa0dde62444cf6858e9b95 | 0.6698 | 0.6642 |
| mask_film_all | 1882162633 | 54f497b7a69f4fcab6ba8f15 | 0.6708 | 0.6701 |
| mask_film_others | 476981973 | cebddc0f667e4aa99a673af9 | 0.6698 | 0.6702 |
| mask_film_others | 621715752 | 7f86dd597f1a415a814a2752 | 0.6710 | 0.6661 |
| mask_film_others | 1882162633 | cef9a72127484089894a0413 | 0.6700 | 0.6674 |
| mask_film_self | 476981973 | 14deaa507574477f93985edd | 0.6696 | 0.6622 |
| mask_film_self | 621715752 | d1a46e576ad94a3b8bddac64 | 0.6694 | 0.6613 |
| mask_film_self | 1882162633 | 5b4ae1a65c62424d9bf3af40 | 0.6706 | 0.6677 |
| mask_mlp_all_unfreeze | 476981973 | 103a0b2514a046089b8f9cd1 | 0.6716 | 0.6653 |
| mask_mlp_all_unfreeze | 621715752 | af46a8f6b40346c6aa2fd793 | 0.6734 | 0.6654 |
| mask_mlp_all_unfreeze | 1882162633 | d8cabf53fbc34f7cb8923dd7 | 0.6712 | 0.6683 |
| mask_mlp_others_unfreeze | 476981973 | b3ac314a030a4fc6a9dc79e9 | 0.6720 | 0.6684 |
| mask_mlp_others_unfreeze | 621715752 | 62dee50b5a004776ae06c07d | 0.6738 | 0.6624 |
| mask_mlp_others_unfreeze | 1882162633 | 2a51da736e3a4f4dac490d60 | 0.6706 | 0.6641 |
| mask_film_all_unfreeze | 476981973 | c98a504bccac425bb0afb43a | 0.6692 | 0.6627 |
| mask_film_all_unfreeze | 621715752 | 36bd15ba318040b5a6c1ab57 | 0.6698 | 0.6642 |
| mask_film_all_unfreeze | 1882162633 | af5e1dc61c9841468310dbb7 | 0.6756 | 0.6677 |
| mask_film_others_unfreeze | 476981973 | 585ea19dfa284894ae0739a8 | 0.6720 | 0.6672 |
| mask_film_others_unfreeze | 621715752 | fc0258326673440cabac64ab | 0.6702 | 0.6646 |
| mask_film_others_unfreeze | 1882162633 | 1e9c8d529d2b4e56bca6f9dc | 0.6712 | 0.6645 |

Answer shift on val, at the kept epoch, mean over seeds:

| Arm | Corrected | Corrupted | Agreement pre to post | Self-attention rate |
|---|---|---|---|---|
| `mask_mlp_all` | 3.28% | 5.34% | 0.678 to 0.614 | 0.448 |
| `mask_mlp_others` | 2.71% | 4.40% | 0.678 to 0.622 | 0 |
| `mask_mlp_self` | 2.89% | 4.47% | 0.678 to 0.625 | 1 |
| `mask_film_all` | 3.62% | 6.48% | 0.678 to 0.589 | 0.405 |
| `mask_film_others` | 3.56% | 6.76% | 0.678 to 0.582 | 0 |
| `mask_film_self` | 3.84% | 6.76% | 0.678 to 0.590 | 1 |
| `mask_mlp_all_unfreeze` | 0.99% | 1.48% | 0.631 to 0.617 | 0.484 |
| `mask_mlp_others_unfreeze` | 1.14% | 2.07% | 0.603 to 0.581 | 0 |
| `mask_film_all_unfreeze` | 2.92% | 4.07% | 0.592 to 0.564 | 0.315 |
| `mask_film_others_unfreeze` | 1.86% | 3.22% | 0.603 to 0.574 | 0 |

Self-attention rate: fraction of receivers whose largest attention weight is on themselves. 0.33 is uniform over 3 members. It is 0 and 1 by construction under `others` and `self`. Pre agreement differs in the unfreeze arms because the unfrozen blocks change the member outputs before messages are applied.

## Findings

- No arm separates from the others. Every frozen arm is within 0.31 points of averaging on test, and the test seed std is 0.17 to 0.28 points with 3 seeds.
- `self` is not clearly below `all` or `others`. With mlp, `self` (+0.06) is below `all` (+0.31) but the gap is about one seed std. With film, `others` (+0.27) is above `all` (+0.12). There is no consistent sign that members use information from each other.
- The answer shift under `self` matches `all` and `others` (2.7 to 3.8% corrected, 4.4 to 6.8% corrupted, similar agreement drop across the frozen arms). Most of the effect messages have on member answers does not need information from other members.
- Messages corrupt more answers than they correct in every arm.
- `film` is not better than `mlp`. It moves member answers more (higher corrected and corrupted), without a test gain.
- Unfreezing does not help. All four unfreeze arms are within 0.11 points of averaging and at or below their frozen counterparts. They also shift answers less, especially with mlp (1.0% corrected, 1.5% corrupted for `mlp_all_unfreeze`). Without `mask_k0_unfreeze`, it is unknown how much of this the unfrozen blocks reach alone.
