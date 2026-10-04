# 02 Overfitting ablation

## Question

Experiment 01 showed overfitting and members becoming individually worse. Do freezing `fc`, early stopping, or a per-member loss fix this, alone or together?

## Changes

- `--freeze-fc`: each backbone's `fc` is frozen. Only the encoders and decoders train. The K = 0 control is then exactly untrained averaging.
- `--patience N`: stop after N epochs without val accuracy improvement and restore the best epoch's weights.
- `--member-loss W`: adds `W * mean_i CE(logits_i, y)` on the post-communication member logits to the ensemble loss.

## Setup

- Base: `film_p0` from experiment 01 (FiLM decoder, patch 0, K = 1, 70 epochs, AdamW lr 1e-3, weight decay 1e-4)
- First 3 joint seeds: 476981973, 621715752, 1882162633
- "none" arm: `film_p0`, same 3 seeds

## Arms

| Arm | Flags |
|---|---|
| none (`film_p0`) | |
| `reg_freeze_fc` | `--freeze-fc` |
| `reg_early_stop` | `--patience 10` |
| `reg_member_loss` | `--member-loss 1` |
| `reg_all` | `--freeze-fc --patience 10 --member-loss 1` |

## Results

| Arm | Val | Test | vs averaging (test) | Best epoch |
|---|---|---|---|---|
| none | 0.6647 ± 0.0039 | 0.6590 ± 0.0016 | -0.62 | 70 |
| `reg_freeze_fc` | 0.6497 ± 0.0020 | 0.6486 ± 0.0032 | -1.66 | 70 |
| `reg_early_stop` | 0.6745 ± 0.0036 | 0.6669 ± 0.0056 | +0.17 | 1, 20, 14 |
| `reg_member_loss` | 0.6636 ± 0.0060 | 0.6673 ± 0.0014 | +0.21 | 70 |
| `reg_all` | 0.6724 ± 0.0020 | 0.6683 ± 0.0012 | +0.31 | 2, 1, 1 |

Val is optimistic for the early stopping arms, since the epoch is selected on val. Compare on test.

Per seed:

| Arm | Seed | Aim run | Val | Test |
|---|---|---|---|---|
| reg_freeze_fc | 476981973 | 97dde63bade64bdfba8c2285 | 0.6488 | 0.6507 |
| reg_freeze_fc | 621715752 | 18f7e3368b86443b9aff88da | 0.6482 | 0.6450 |
| reg_freeze_fc | 1882162633 | 8f22fcba530b4047973abc03 | 0.6520 | 0.6502 |
| reg_early_stop | 476981973 | 0b6e4de3fe2249439e50d611 | 0.6718 | 0.6621 |
| reg_early_stop | 621715752 | b1f3cd582fa14f3a94305240 | 0.6732 | 0.6656 |
| reg_early_stop | 1882162633 | a499adf279f34b43ac92e578 | 0.6786 | 0.6730 |
| reg_member_loss | 476981973 | 696e79a9985e4599839cd41e | 0.6570 | 0.6660 |
| reg_member_loss | 621715752 | 93995d5ee7e34ac69fff0a6d | 0.6652 | 0.6688 |
| reg_member_loss | 1882162633 | 4de3be6509f344b0bd684271 | 0.6686 | 0.6671 |
| reg_all | 476981973 | 848e729e40eb46ec8c8c24f8 | 0.6702 | 0.6669 |
| reg_all | 621715752 | c753f7adc47c4347800388b9 | 0.6742 | 0.6693 |
| reg_all | 1882162633 | ac660c3a59b84c6594cdd88b | 0.6728 | 0.6687 |

Answer shift on val, at the kept epoch:

| Arm | Corrected | Corrupted | Agreement pre to post | Self-attention rate |
|---|---|---|---|---|
| none | 5.37% | 7.02% | 0.577 to 0.534 | 0.518 |
| `reg_freeze_fc` | 5.07% | 11.16% | 0.679 to 0.530 | 0.422 |
| `reg_early_stop` | 3.94% | 5.25% | 0.622 to 0.580 | 0.533 |
| `reg_member_loss` | 5.60% | 6.36% | 0.642 to 0.606 | 0.412 |
| `reg_all` | 2.68% | 3.01% | 0.679 to 0.667 | 0.548 |

Self-attention rate: fraction of receivers whose largest attention weight is on themselves. 0.33 is uniform over 3 members.

## Findings

- `reg_all` is the only arm above averaging on every seed (+0.31 test). With `fc` frozen its K = 0 control is untrained averaging, so this comparison is fair.
- `reg_all` keeps the epoch 1 or 2 checkpoint. Corrected and corrupted both drop to about 3%, and agreement barely changes. The channel moves toward doing nothing, close to the identity initialization. The gain is likely a confidence adjustment, not members changing answers.
- `reg_freeze_fc` is the worst arm. With `fc` frozen and only the ensemble loss, messages corrupt 11.2% of member answers.
- `reg_member_loss` trains `fc`, so its gain over `film_p0` (+0.83 test) cannot be attributed to communication. It may come from retraining the heads.
- The member loss acts as a guardrail against messages harming members. It does not create a reason to communicate.
- `reg_all` self-attention rate is 0.55, so most receivers attend mainly to themselves. Communication between members is not established.
