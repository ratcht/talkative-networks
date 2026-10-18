# 04 Patch attention: results

Setup, arms and words are in [README.md](README.md). All 18 arms finished on all 3 seeds (54 runs).

Baseline: untrained probability averaging, val 0.6686, test 0.6652.

## Results

| Arm | Val | Test | vs averaging (test) | Best epoch |
|---|---|---|---|---|
| `p0_linear_add` | 0.6708 ± 0.0006 | 0.6713 ± 0.0029 | +0.61 | 1, 2, 1 |
| `p0_mlp_add` | 0.6706 ± 0.0007 | 0.6674 ± 0.0036 | +0.22 | 1, 2, 3 |
| `p1_linear_add` | 0.6647 ± 0.0008 | 0.6562 ± 0.0009 | -0.90 | 1, 1, 5 |
| `p1_linear_add_pattn` | 0.6705 ± 0.0019 | 0.6628 ± 0.0011 | -0.24 | 1, 1, 6 |
| `p1_mlp_add` | 0.6712 ± 0.0022 | 0.6633 ± 0.0024 | -0.19 | 2, 1, 6 |
| `p1_mlp_add_pattn` | 0.6724 ± 0.0031 | 0.6646 ± 0.0032 | -0.06 | 10, 3, 1 |
| `p2_linear_add` | 0.6353 ± 0.0018 | 0.6285 ± 0.0020 | -3.67 | 24, 14, 34 |
| `p2_linear_add_pattn` | 0.6522 ± 0.0056 | 0.6470 ± 0.0065 | -1.82 | 10, 14, 2 |
| `p2_mlp_add` | 0.6671 ± 0.0023 | 0.6625 ± 0.0054 | -0.27 | 9, 1, 1 |
| `p2_mlp_add_pattn` | 0.6709 ± 0.0002 | 0.6633 ± 0.0014 | -0.19 | 4, 8, 3 |
| `p1_linear_replace` | 0.5970 ± 0.0052 | 0.5880 ± 0.0020 | -7.72 | 25, 29, 27 |
| `p1_linear_replace_pattn` | 0.3575 ± 0.1848 | 0.3523 ± 0.1895 | -31.29 | 28, 70, 48 |
| `p1_mlp_replace` | 0.5883 ± 0.0248 | 0.5808 ± 0.0251 | -8.44 | 41, 15, 50 |
| `p1_mlp_replace_pattn` | 0.3427 ± 0.1651 | 0.3398 ± 0.1690 | -32.54 | 53, 58, 65 |
| `p2_linear_replace` | 0.6013 ± 0.0120 | 0.5992 ± 0.0073 | -6.60 | 54, 63, 43 |
| `p2_linear_replace_pattn` | 0.5544 ± 0.0278 | 0.5510 ± 0.0228 | -11.42 | 34, 67, 69 |
| `p2_mlp_replace` | 0.5825 ± 0.0078 | 0.5764 ± 0.0126 | -8.88 | 28, 53, 32 |
| `p2_mlp_replace_pattn` | 0.4537 ± 0.0506 | 0.4533 ± 0.0534 | -21.19 | 16, 35, 27 |

Best epochs are in seed order 476981973, 621715752, 1882162633. Val is optimistic, since the epoch is selected on val. Compare on test.

Per seed:

| Arm | Seed | Aim run | Best epoch | Val | Test |
|---|---|---|---|---|---|
| p0_linear_add | 476981973 | c8c2b812042f47c0bf4a9655 | 1 | 0.6714 | 0.6682 |
| p0_linear_add | 621715752 | 747683a0245241ce8d11db22 | 2 | 0.6702 | 0.6738 |
| p0_linear_add | 1882162633 | a9b0395533e248de8a054b78 | 1 | 0.6708 | 0.6720 |
| p0_mlp_add | 476981973 | 3f095bd823954800aa97bb57 | 1 | 0.6714 | 0.6661 |
| p0_mlp_add | 621715752 | 515ed1cb2a024d1da1734aa8 | 2 | 0.6700 | 0.6646 |
| p0_mlp_add | 1882162633 | 6a595fecd37747228901296c | 3 | 0.6704 | 0.6714 |
| p1_linear_add | 476981973 | 85803f9c05e44b21a4803bc5 | 1 | 0.6654 | 0.6558 |
| p1_linear_add | 621715752 | 8e5551bbc4de44c9a18950eb | 1 | 0.6650 | 0.6573 |
| p1_linear_add | 1882162633 | 69d7baa36cb04115a37f0cf7 | 5 | 0.6638 | 0.6556 |
| p1_linear_add_pattn | 476981973 | d46a3a6890094be688cf6fd6 | 1 | 0.6710 | 0.6630 |
| p1_linear_add_pattn | 621715752 | a0a893608f4149ef91a7a2e2 | 1 | 0.6684 | 0.6637 |
| p1_linear_add_pattn | 1882162633 | 5e4a19e67a834f3bb53163f1 | 6 | 0.6720 | 0.6616 |
| p1_mlp_add | 476981973 | b79b9116fc734afaaa1c78b2 | 2 | 0.6716 | 0.6657 |
| p1_mlp_add | 621715752 | 0ee0a869091d415193d04b83 | 1 | 0.6732 | 0.6633 |
| p1_mlp_add | 1882162633 | 48b86879290243ceb2a7f7b7 | 6 | 0.6688 | 0.6610 |
| p1_mlp_add_pattn | 476981973 | 1dfece84c2e24ed787b24c18 | 10 | 0.6694 | 0.6624 |
| p1_mlp_add_pattn | 621715752 | 89e6aef895cd4e06a6384660 | 3 | 0.6756 | 0.6682 |
| p1_mlp_add_pattn | 1882162633 | 548da8e23988434d8d07aeb4 | 1 | 0.6722 | 0.6631 |
| p2_linear_add | 476981973 | 44cc250cc0514cd88fc0e7d9 | 24 | 0.6332 | 0.6266 |
| p2_linear_add | 621715752 | 80d923dfd2f247d2877e37f6 | 14 | 0.6362 | 0.6283 |
| p2_linear_add | 1882162633 | c08ba6e0c5714f3db40cf3f4 | 34 | 0.6364 | 0.6306 |
| p2_linear_add_pattn | 476981973 | ee6b969e14954438a26534a0 | 10 | 0.6514 | 0.6454 |
| p2_linear_add_pattn | 621715752 | 9c8de797149d46ab8eab400e | 14 | 0.6470 | 0.6414 |
| p2_linear_add_pattn | 1882162633 | 2d62b47cd37b4eb2bd3d017c | 2 | 0.6582 | 0.6541 |
| p2_mlp_add | 476981973 | 2ca99553e7ab48fc8cf6cada | 9 | 0.6686 | 0.6687 |
| p2_mlp_add | 621715752 | 8c58639c23da40b4a059f5dd | 1 | 0.6644 | 0.6600 |
| p2_mlp_add | 1882162633 | fa88fcb8541b436bbdcdea64 | 1 | 0.6682 | 0.6588 |
| p2_mlp_add_pattn | 476981973 | 3b97189565384c37b4042384 | 4 | 0.6706 | 0.6640 |
| p2_mlp_add_pattn | 621715752 | ac6ed04c811a423389097100 | 8 | 0.6710 | 0.6642 |
| p2_mlp_add_pattn | 1882162633 | e61f7980e9cc4148b639615a | 3 | 0.6710 | 0.6616 |
| p1_linear_replace | 476981973 | 2d8a5759883b432eab105005 | 25 | 0.5934 | 0.5864 |
| p1_linear_replace | 621715752 | f371c4b080574a3589750f66 | 29 | 0.6030 | 0.5903 |
| p1_linear_replace | 1882162633 | 02ed56d003054c93b05eeceb | 27 | 0.5946 | 0.5874 |
| p1_linear_replace_pattn | 476981973 | b0037e47ad1f4350b77dc7ea | 28 | 0.5500 | 0.5458 |
| p1_linear_replace_pattn | 621715752 | 877b046e23e64088beae749b | 70 | 0.3410 | 0.3441 |
| p1_linear_replace_pattn | 1882162633 | f2fcff61d3fb4898b27d55e3 | 48 | 0.1816 | 0.1671 |
| p1_mlp_replace | 476981973 | 9f70203ea0ae4a7192ff9b83 | 41 | 0.6056 | 0.6023 |
| p1_mlp_replace | 621715752 | 4595f316b61842aaa87304de | 15 | 0.5598 | 0.5532 |
| p1_mlp_replace | 1882162633 | 6adaa3235edc479fb3e32728 | 50 | 0.5994 | 0.5868 |
| p1_mlp_replace_pattn | 476981973 | 1d6d9588e6b7453fb87e9c58 | 53 | 0.3050 | 0.2997 |
| p1_mlp_replace_pattn | 621715752 | ed9fd70f2d8648cbb6113a26 | 58 | 0.1998 | 0.1945 |
| p1_mlp_replace_pattn | 1882162633 | c488930b3725418b92bcffe6 | 65 | 0.5234 | 0.5252 |
| p2_linear_replace | 476981973 | b4fa5c60b7e04d6fad346966 | 54 | 0.5880 | 0.5914 |
| p2_linear_replace | 621715752 | 31ba4bcb5b974db3b5150f56 | 63 | 0.6046 | 0.6004 |
| p2_linear_replace | 1882162633 | 3d69eb86f2104438a45c1901 | 43 | 0.6112 | 0.6059 |
| p2_linear_replace_pattn | 476981973 | 07665d2b8591497f9aac1faf | 34 | 0.5664 | 0.5587 |
| p2_linear_replace_pattn | 621715752 | aa6557faa33449a0b285d0da | 67 | 0.5742 | 0.5689 |
| p2_linear_replace_pattn | 1882162633 | 583c6b5e73fb42ae833631c9 | 69 | 0.5226 | 0.5253 |
| p2_mlp_replace | 476981973 | f1c735689ef541c1ba61b8c2 | 28 | 0.5800 | 0.5704 |
| p2_mlp_replace | 621715752 | 8723bd4e4e0d423abd1dae63 | 53 | 0.5762 | 0.5679 |
| p2_mlp_replace | 1882162633 | 621aa03f20524458a419577e | 32 | 0.5912 | 0.5909 |
| p2_mlp_replace_pattn | 476981973 | 81d1168970644dc799173924 | 16 | 0.4548 | 0.4523 |
| p2_mlp_replace_pattn | 621715752 | 3328f055c7634c2ab9007409 | 35 | 0.5038 | 0.5072 |
| p2_mlp_replace_pattn | 1882162633 | 66ad538594f8448ba0ad0f5d | 27 | 0.4026 | 0.4004 |

## Other metrics

All values below are means over seeds, on val at the kept epoch unless noted.

### Answer shift

| Arm | Corrected | Corrupted | Net | Flip rate | Agreement pre to post | KL(post ‖ pre) |
|---|---|---|---|---|---|---|
| `p0_linear_add` | 3.31% | 5.77% | -2.46% | 16.6% | 0.678 to 0.597 | 0.61 |
| `p0_mlp_add` | 2.86% | 4.50% | -1.64% | 13.6% | 0.678 to 0.626 | 0.37 |
| `p1_linear_add` | 3.64% | 5.79% | -2.15% | 18.2% | 0.678 to 0.622 | 0.45 |
| `p1_linear_add_pattn` | 3.04% | 4.36% | -1.32% | 14.2% | 0.678 to 0.638 | 0.37 |
| `p1_mlp_add` | 3.20% | 5.05% | -1.85% | 15.6% | 0.678 to 0.624 | 0.46 |
| `p1_mlp_add_pattn` | 2.86% | 6.10% | -3.23% | 16.4% | 0.678 to 0.591 | 0.91 |
| `p2_linear_add` | 5.66% | 12.46% | -6.80% | 32.5% | 0.678 to 0.544 | 1.48 |
| `p2_linear_add_pattn` | 4.20% | 16.74% | -12.54% | 35.8% | 0.678 to 0.407 | 3.52 |
| `p2_mlp_add` | 2.34% | 5.23% | -2.88% | 13.8% | 0.678 to 0.606 | 0.74 |
| `p2_mlp_add_pattn` | 1.02% | 1.51% | -0.49% | 5.0% | 0.678 to 0.664 | 0.10 |
| `p1_linear_replace` | 4.77% | 28.15% | -23.38% | 54.4% | 0.678 to 0.241 | 6.53 |
| `p1_linear_replace_pattn` | 3.68% | 45.39% | -41.70% | 77.4% | 0.678 to 0.073 | 11.84 |
| `p1_mlp_replace` | 4.80% | 29.38% | -24.58% | 56.1% | 0.678 to 0.253 | 6.86 |
| `p1_mlp_replace_pattn` | 3.87% | 45.38% | -41.51% | 77.3% | 0.678 to 0.076 | 11.78 |
| `p2_linear_replace` | 6.44% | 19.14% | -12.71% | 43.9% | 0.678 to 0.448 | 3.07 |
| `p2_linear_replace_pattn` | 5.72% | 28.92% | -23.20% | 56.5% | 0.678 to 0.279 | 5.77 |
| `p2_mlp_replace` | 6.08% | 23.18% | -17.10% | 49.3% | 0.678 to 0.377 | 4.12 |
| `p2_mlp_replace_pattn` | 4.36% | 40.89% | -36.54% | 71.6% | 0.678 to 0.125 | 9.69 |

Net = corrected - corrupted, per member.

### Routing

| Arm | Self-attention rate | Top-1 weight | Joint mode share | Route flip rate | Routed correct (all / corrected / corrupted) | Block 2 entropy, start to kept epoch |
|---|---|---|---|---|---|---|
| `p0_linear_add` | 0.115 | 0.835 | 0.952 | 0.130 | 0.632 / 0.436 / 0.658 | |
| `p0_mlp_add` | 0.334 | 0.914 | 0.999 | 0.178 | 0.633 / 0.446 / 0.641 | |
| `p1_linear_add` | 0.289 | 0.704 | 0.198 | 0.324 | 0.628 / 0.468 / 0.634 | |
| `p1_linear_add_pattn` | 0.409 | 0.733 | 0.265 | 0.232 | 0.628 / 0.465 / 0.629 | 3.94 to 3.73 |
| `p1_mlp_add` | 0.296 | 0.775 | 0.296 | 0.325 | 0.635 / 0.496 / 0.632 | |
| `p1_mlp_add_pattn` | 0.325 | 0.855 | 0.463 | 0.299 | 0.627 / 0.425 / 0.650 | 3.85 to 3.11 |
| `p2_linear_add` | 0.333 | 0.986 | 0.918 | 0.037 | 0.625 / 0.459 / 0.669 | |
| `p2_linear_add_pattn` | 0.283 | 0.984 | 0.796 | 0.088 | 0.624 / 0.432 / 0.717 | 2.53 to 1.57 |
| `p2_mlp_add` | 0.481 | 0.949 | 0.870 | 0.008 | 0.623 / 0.507 / 0.560 | |
| `p2_mlp_add_pattn` | 0.041 | 0.992 | 0.849 | 0.259 | 0.634 / 0.447 / 0.605 | 2.37 to 0.22 |
| `p1_linear_replace` | 0.452 | 0.817 | 0.244 | 0.090 | 0.637 / 0.453 / 0.809 | |
| `p1_linear_replace_pattn` | 0.257 | 0.844 | 0.132 | 0.088 | 0.631 / 0.369 / 0.837 | 3.79 to 3.01 |
| `p1_mlp_replace` | 0.340 | 0.817 | 0.317 | 0.088 | 0.633 / 0.414 / 0.792 | |
| `p1_mlp_replace_pattn` | 0.349 | 0.860 | 0.175 | 0.082 | 0.631 / 0.393 / 0.825 | 3.79 to 3.12 |
| `p2_linear_replace` | 0.192 | 0.999 | 0.735 | 0.020 | 0.630 / 0.445 / 0.734 | |
| `p2_linear_replace_pattn` | 0.444 | 1.000 | 1.000 | 0.000 | 0.622 / 0.414 / 0.734 | 2.35 to 0.23 |
| `p2_mlp_replace` | 0.352 | 0.999 | 0.803 | 0.023 | 0.636 / 0.434 / 0.763 | |
| `p2_mlp_replace_pattn` | 0.338 | 1.000 | 0.985 | 0.002 | 0.634 / 0.331 / 0.844 | 2.28 to 0.39 |

- Top-1 weight and block 2 entropy are train-step values, averaged over the kept epoch. Block 2 entropy start is the mean of the first 20 steps.
- Block 1 attends over 3 members, so top-1 weight 0.33 is uniform. Block 2 maximum entropy is ln 64 = 4.16 for patch 1 and ln 16 = 2.77 for patch 2.
- Joint mode share: fraction of examples whose full routing pattern (which sender each receiver attends to most) is the most common one. 1 means the same wiring for every input.
- Routed correct: how often the sender a receiver attended to most was itself right.

### Fit and message size

| Arm | Val after epoch 1 | Train at kept epoch | Train at last epoch | Epochs run | Injection norm, start to kept epoch to last |
|---|---|---|---|---|---|
| `p0_linear_add` | 0.670 | 0.670 | 0.673 | 11 | 0.9 to 1.9 to 3.3 |
| `p0_mlp_add` | 0.669 | 0.668 | 0.672 | 12 | 0.3 to 1.9 to 3.5 |
| `p1_linear_add` | 0.664 | 0.664 | 0.683 | 12 | 14 to 37 to 107 |
| `p1_linear_add_pattn` | 0.669 | 0.667 | 0.682 | 13 | 11 to 37 to 106 |
| `p1_mlp_add` | 0.669 | 0.668 | 0.692 | 13 | 6 to 36 to 101 |
| `p1_mlp_add_pattn` | 0.668 | 0.669 | 0.682 | 15 | 5 to 53 to 105 |
| `p2_linear_add` | 0.609 | 0.793 | 0.826 | 34 | 21 to 235 to 277 |
| `p2_linear_add_pattn` | 0.615 | 0.655 | 0.672 | 19 | 16 to 140 to 189 |
| `p2_mlp_add` | 0.667 | 0.666 | 0.692 | 14 | 10 to 35 to 79 |
| `p2_mlp_add_pattn` | 0.667 | 0.668 | 0.672 | 15 | 9 to 8 to 13 |
| `p1_linear_replace` | 0.050 | 0.648 | 0.665 | 37 | 154 to 449 to 481 |
| `p1_linear_replace_pattn` | 0.052 | 0.422 | 0.429 | 55 | 134 to 552 to 573 |
| `p1_mlp_replace` | 0.136 | 0.649 | 0.667 | 45 | 57 to 432 to 465 |
| `p1_mlp_replace_pattn` | 0.032 | 0.416 | 0.426 | 67 | 65 to 684 to 716 |
| `p2_linear_replace` | 0.187 | 0.878 | 0.890 | 62 | 186 to 848 to 906 |
| `p2_linear_replace_pattn` | 0.119 | 0.689 | 0.696 | 61 | 174 to 1284 to 1354 |
| `p2_mlp_replace` | 0.099 | 0.834 | 0.873 | 48 | 88 to 743 to 822 |
| `p2_mlp_replace_pattn` | 0.079 | 0.490 | 0.482 | 36 | 107 to 953 to 1167 |

- Train accuracy of the untrained ensemble is about 0.67. The backbones were not trained on the joint split, so train accuracy far above that is the decoder memorizing it.
- Injection norm is the L2 norm of the decoder output per example. Start is the mean of the first 20 steps, last is the mean of the last 200. It is not comparable across patch sizes: for patch 0 it is a 16-vector broadcast over the grid, for patch 1 and 2 a full 16 × 32 × 32 map.

## Findings

- No arm separates from averaging. Only `p0_linear_add` (+0.61) is above it by more than one seed std, and it peaks at epoch 1 or 2, as in experiment 03. Every patch 1 and patch 2 arm is at or below averaging on test.
- Messages make members worse in every arm. Net corrected minus corrupted is negative in all 18 arms. `p0_linear_add` gains on the ensemble while its members lose 2.5% net and agreement drops from 0.678 to 0.597, so the gain is more likely added diversity between members than information passing.
- Receivers do not route to the member that is right. Routed correct is 0.62 to 0.64 in every arm, about the accuracy of one member, and it is lower on corrected answers (0.33 to 0.51). Corrections do not come from copying a correct sender.
- Block 1 routing becomes a fixed wiring in patch 0 and patch 2. Top-1 weight is 0.84 to 1.00 and the joint mode share 0.74 to 1.00, so each receiver reads the same sender for almost every input. Patch 1 keeps input-dependent routing (joint mode share 0.13 to 0.46). In `p2_linear_replace_pattn` the attention is fully saturated and query/key gradient norms fall to about 0.007.
- Block 2 stays near uniform at patch 1 (3.0 to 3.7 of 4.16), so it mostly averages a member's 64 patches into one global message. This fits both of its effects at patch 1: it moves the additive arms toward patch 0 (`p1_linear_add` -0.90 to -0.24, `p1_mlp_add` -0.19 to -0.06), and it collapses the replace arms (-31 and -33), which need the spatial layout to rebuild the `layer1.2` input. At patch 2, block 2 becomes near one-hot (0.2 to 0.4 of 2.77), except `p2_linear_add_pattn` (1.57).
- Block 2 is above its no-block-2 counterpart in all 4 additive pairs and below it in all 4 replace pairs.
- Replace does not work. All 8 replace arms are 6.6 to 32.5 points below averaging. Validation accuracy after epoch 1 is 0.03 to 0.19, best epochs are late (15 to 70), and the `replace_pattn` arms were still climbing when they stopped, with test 0.17 to 0.55 across seeds.
- Patch 2 decoders (about 1M parameters per run, vs 21k to 100k at patch 0 and 1) memorize the joint training split: `p2_linear_replace` train 0.89 vs val 0.60, `p2_mlp_replace` 0.87 vs 0.58, `p2_linear_add` 0.83 vs 0.63. Patch size and decoder size are confounded in the patch 1 vs patch 2 comparison.
- `p2_linear_add` gets worse inside epoch 1. Train loss rises from 1.65 to 1.89 over the first epoch while the injection norm grows from 0 to 67, and val falls to 0.61 and only recovers to 0.63 while overfitting. A 1M-parameter linear decoder at lr 1e-3 looks too aggressive.
- Injection norms keep growing up to the last epoch in every arm except `p2_mlp_add_pattn`, which learned to stay nearly silent (injection 8.9 to 7.9, flip rate 5%, KL 0.10) and lands within 0.2 points of averaging.
- Linear vs MLP decoder: MLP is better at patch 1 and patch 2 additive (by 0.2 to 3.4 points), linear is better at patch 0 (+0.61 vs +0.22). Within one seed std at patch 0.

## Metric issues

- `update_ratio` for the linear decoders reads 1e8 to 1e9 in epoch 1. The additive decoders start at zero weight, so the ratio divides by a norm near zero. It is an artifact, not a real update size.
- The summary run for `pa_p2_mlp_replace` (`fcab7a9581824557adb484c3` in the log) did not show up in an Aim query. Its 3 seed runs are present.
