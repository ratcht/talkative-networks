# 05 Minimal exchange: results

Setup, arms and words are in [README.md](README.md). All 48 arms finished on all 3 seeds (144 runs).

Baseline: untrained probability averaging, val 0.6686, test 0.6652.

## Results

| Arm | Val | Test | vs averaging (test) | Best epoch | Epochs run |
|---|---|---|---|---|---|
| `average_none_linear_add` | 0.6659 ± 0.0033 | 0.6616 ± 0.0009 | -0.36 | 1, 7, 1 | 11, 17, 11 |
| `average_none_linear_replace` | 0.3587 ± 0.0005 | 0.3546 ± 0.0025 | -31.06 | 36, 17, 24 | 46, 27, 34 |
| `average_none_mlp_add` | 0.6702 ± 0.0014 | 0.6684 ± 0.0016 | +0.32 | 3, 3, 2 | 13, 13, 12 |
| `average_none_mlp_replace` | 0.3276 ± 0.0143 | 0.3295 ± 0.0185 | -33.57 | 25, 24, 23 | 35, 34, 33 |
| `average_p64_linear_add` | 0.6679 ± 0.0011 | 0.6655 ± 0.0039 | +0.03 | 1, 2, 1 | 11, 12, 11 |
| `average_p64_linear_replace` | 0.3911 ± 0.0103 | 0.3937 ± 0.0076 | -27.15 | 20, 18, 25 | 30, 28, 35 |
| `average_p64_mlp_add` | 0.6697 ± 0.0030 | 0.6676 ± 0.0042 | +0.24 | 4, 3, 3 | 14, 13, 13 |
| `average_p64_mlp_replace` | 0.3486 ± 0.0357 | 0.3451 ± 0.0302 | -32.01 | 17, 14, 27 | 27, 24, 37 |
| `average_p32_linear_add` | 0.6689 ± 0.0023 | 0.6616 ± 0.0006 | -0.36 | 1, 2, 2 | 11, 12, 12 |
| `average_p32_linear_replace` | 0.4043 ± 0.0064 | 0.4060 ± 0.0082 | -25.92 | 37, 26, 36 | 47, 36, 46 |
| `average_p32_mlp_add` | 0.6709 ± 0.0015 | 0.6626 ± 0.0018 | -0.26 | 2, 1, 3 | 12, 11, 13 |
| `average_p32_mlp_replace` | 0.3758 ± 0.0112 | 0.3741 ± 0.0105 | -29.11 | 32, 30, 32 | 42, 40, 42 |
| `gate_none_linear_add` | 0.6657 ± 0.0012 | 0.6628 ± 0.0058 | -0.24 | 4, 1, 3 | 14, 11, 13 |
| `gate_none_linear_replace` | 0.3505 ± 0.0133 | 0.3451 ± 0.0075 | -32.01 | 28, 17, 16 | 38, 27, 26 |
| `gate_none_mlp_add` | 0.6705 ± 0.0028 | 0.6657 ± 0.0030 | +0.05 | 1, 4, 4 | 11, 14, 14 |
| `gate_none_mlp_replace` | 0.3061 ± 0.0264 | 0.3053 ± 0.0240 | -35.99 | 24, 38, 30 | 34, 48, 40 |
| `gate_p64_linear_add` | 0.6679 ± 0.0019 | 0.6652 ± 0.0037 | -0.00 | 1, 4, 2 | 11, 14, 12 |
| `gate_p64_linear_replace` | 0.3422 ± 0.0482 | 0.3416 ± 0.0452 | -32.36 | 20, 13, 38 | 30, 23, 48 |
| `gate_p64_mlp_add` | 0.6663 ± 0.0039 | 0.6630 ± 0.0031 | -0.22 | 1, 2, 1 | 11, 12, 11 |
| `gate_p64_mlp_replace` | 0.3111 ± 0.0193 | 0.3078 ± 0.0197 | -35.74 | 15, 17, 21 | 25, 27, 31 |
| `gate_p32_linear_add` | 0.6683 ± 0.0021 | 0.6638 ± 0.0004 | -0.14 | 1, 1, 2 | 11, 11, 12 |
| `gate_p32_linear_replace` | 0.3727 ± 0.0114 | 0.3702 ± 0.0083 | -29.50 | 32, 19, 28 | 42, 29, 38 |
| `gate_p32_mlp_add` | 0.6705 ± 0.0044 | 0.6652 ± 0.0036 | +0.00 | 4, 2, 6 | 14, 12, 16 |
| `gate_p32_mlp_replace` | 0.3501 ± 0.0037 | 0.3463 ± 0.0015 | -31.89 | 33, 35, 30 | 43, 45, 40 |
| `attention_none_linear_add` | 0.6683 ± 0.0010 | 0.6638 ± 0.0007 | -0.14 | 1, 2, 2 | 11, 12, 12 |
| `attention_none_linear_replace` | 0.3439 ± 0.0185 | 0.3432 ± 0.0163 | -32.20 | 28, 20, 22 | 38, 30, 32 |
| `attention_none_mlp_add` | 0.6679 ± 0.0022 | 0.6637 ± 0.0046 | -0.15 | 3, 2, 2 | 13, 12, 12 |
| `attention_none_mlp_replace` | 0.3273 ± 0.0036 | 0.3279 ± 0.0045 | -33.73 | 16, 16, 27 | 26, 26, 37 |
| `attention_p64_linear_add` | 0.6682 ± 0.0018 | 0.6638 ± 0.0023 | -0.14 | 1, 3, 1 | 11, 13, 11 |
| `attention_p64_linear_replace` | 0.3691 ± 0.0014 | 0.3680 ± 0.0015 | -29.72 | 19, 26, 18 | 29, 36, 28 |
| `attention_p64_mlp_add` | 0.6697 ± 0.0019 | 0.6658 ± 0.0019 | +0.06 | 1, 2, 1 | 11, 12, 11 |
| `attention_p64_mlp_replace` | 0.3310 ± 0.0038 | 0.3359 ± 0.0118 | -32.93 | 21, 18, 18 | 31, 28, 28 |
| `attention_p32_linear_add` | 0.6670 ± 0.0022 | 0.6649 ± 0.0034 | -0.03 | 1, 2, 4 | 11, 12, 14 |
| `attention_p32_linear_replace` | 0.3686 ± 0.0127 | 0.3657 ± 0.0088 | -29.95 | 19, 20, 32 | 29, 30, 42 |
| `attention_p32_mlp_add` | 0.6699 ± 0.0046 | 0.6666 ± 0.0020 | +0.14 | 5, 3, 2 | 15, 13, 12 |
| `attention_p32_mlp_replace` | 0.3396 ± 0.0356 | 0.3390 ± 0.0279 | -32.62 | 34, 30, 36 | 44, 40, 46 |
| `attention_zq_none_linear_add` | 0.6675 ± 0.0018 | 0.6668 ± 0.0023 | +0.16 | 3, 2, 4 | 13, 12, 14 |
| `attention_zq_none_linear_replace` | 0.3440 ± 0.0140 | 0.3466 ± 0.0171 | -31.86 | 38, 14, 27 | 48, 24, 37 |
| `attention_zq_none_mlp_add` | 0.6693 ± 0.0029 | 0.6629 ± 0.0028 | -0.23 | 3, 2, 2 | 13, 12, 12 |
| `attention_zq_none_mlp_replace` | 0.3232 ± 0.0107 | 0.3299 ± 0.0082 | -33.53 | 28, 21, 21 | 38, 31, 31 |
| `attention_zq_p64_linear_add` | 0.6675 ± 0.0020 | 0.6627 ± 0.0009 | -0.25 | 1, 1, 1 | 11, 11, 11 |
| `attention_zq_p64_linear_replace` | 0.3653 ± 0.0048 | 0.3667 ± 0.0047 | -29.85 | 23, 14, 21 | 33, 24, 31 |
| `attention_zq_p64_mlp_add` | 0.6712 ± 0.0009 | 0.6660 ± 0.0033 | +0.08 | 1, 3, 2 | 11, 13, 12 |
| `attention_zq_p64_mlp_replace` | 0.3289 ± 0.0078 | 0.3307 ± 0.0077 | -33.45 | 23, 18, 21 | 33, 28, 31 |
| `attention_zq_p32_linear_add` | 0.6683 ± 0.0018 | 0.6628 ± 0.0008 | -0.24 | 1, 1, 1 | 11, 11, 11 |
| `attention_zq_p32_linear_replace` | 0.3689 ± 0.0138 | 0.3697 ± 0.0149 | -29.55 | 35, 26, 25 | 45, 36, 35 |
| `attention_zq_p32_mlp_add` | 0.6703 ± 0.0045 | 0.6628 ± 0.0033 | -0.24 | 3, 1, 2 | 13, 11, 12 |
| `attention_zq_p32_mlp_replace` | 0.3491 ± 0.0060 | 0.3499 ± 0.0116 | -31.53 | 41, 31, 33 | 51, 41, 43 |

Best epochs and epochs run are in seed order 476981973, 621715752, 1882162633. No run reached the 150-epoch cap: the longest ran 51 epochs. Val is optimistic, since the epoch is selected on val. Compare on test.

### Mixing, head to head

Test difference in points between two mixes with the same projection, decoder and injection, averaged over the 6 matching pairs of each injection.

| Comparison | Additive | Replace |
|---|---|---|
| gate - average | -0.03 (-0.46 to +0.26) | -3.11 (-5.21 to -0.95) |
| attention - average | +0.02 (-0.46 to +0.39) | -2.05 (-4.03 to -0.16) |
| attention_zq - average | -0.05 (-0.54 to +0.52) | -1.82 (-3.62 to +0.04) |
| attention_zq - attention | -0.07 (-0.37 to +0.30) | +0.23 (-0.52 to +1.08) |
| attention - gate | +0.05 (-0.19 to +0.27) | +1.06 (-0.73 to +2.81) |

Mean test vs averaging, by factor:

| Factor | Additive | Replace |
|---|---|---|
| average / gate / attention / attention_zq | -0.07 / -0.09 / -0.04 / -0.12 | -29.80 / -32.91 / -31.86 / -31.63 |
| projection none / 64→64 / 64→32 | -0.07 / -0.02 / -0.14 | -32.99 / -31.65 / -30.01 |
| decoder linear / MLP | -0.14 / -0.02 | -30.09 / -33.01 |

Per seed:

| Arm | Seed | Aim run | Best epoch | Val | Test |
|---|---|---|---|---|---|
| average_none_linear_add | 476981973 | a0acbc76fc8146beb036ada9 | 1 | 0.6622 | 0.6606 |
| average_none_linear_add | 621715752 | 555c51d7b59d4730bef1880d | 7 | 0.6672 | 0.6622 |
| average_none_linear_add | 1882162633 | 203747f4ce9640a291c80fee | 1 | 0.6684 | 0.6619 |
| average_none_linear_replace | 476981973 | ed6d3b41ada944a6aacf7280 | 36 | 0.3588 | 0.3517 |
| average_none_linear_replace | 621715752 | 253165cf75b9473c83325653 | 17 | 0.3582 | 0.3559 |
| average_none_linear_replace | 1882162633 | 8e56894668a24cc8adea1845 | 24 | 0.3592 | 0.3562 |
| average_none_mlp_add | 476981973 | 8b4c3f6a603644f9b0763ea6 | 3 | 0.6716 | 0.6699 |
| average_none_mlp_add | 621715752 | 906744606a4e45419b6c00ad | 3 | 0.6688 | 0.6684 |
| average_none_mlp_add | 1882162633 | 92f6873f46884986ac16a0c8 | 2 | 0.6702 | 0.6668 |
| average_none_mlp_replace | 476981973 | 6d9b161cbe2f4bc8b68349b8 | 25 | 0.3434 | 0.3501 |
| average_none_mlp_replace | 621715752 | c3cc2647248641968f5ff5b8 | 24 | 0.3156 | 0.3144 |
| average_none_mlp_replace | 1882162633 | 3da1ab9df7274a89a92191a0 | 23 | 0.3238 | 0.3240 |
| average_p64_linear_add | 476981973 | f5d1b8a9a2094ef3ac710064 | 1 | 0.6672 | 0.6648 |
| average_p64_linear_add | 621715752 | dea9cde7f760401e90b91217 | 2 | 0.6692 | 0.6697 |
| average_p64_linear_add | 1882162633 | e5d4c5e823a34a82b334e3ab | 1 | 0.6674 | 0.6621 |
| average_p64_linear_replace | 476981973 | 6a878785d8454150ac0d0254 | 20 | 0.3818 | 0.3883 |
| average_p64_linear_replace | 621715752 | 9b667ad9fe7747e7bf63f43a | 18 | 0.3892 | 0.3903 |
| average_p64_linear_replace | 1882162633 | 1da26feb638a4be2a70ffcd1 | 25 | 0.4022 | 0.4024 |
| average_p64_mlp_add | 476981973 | 907231db331a411f89adc61a | 4 | 0.6706 | 0.6676 |
| average_p64_mlp_add | 621715752 | 87d547b9d5f443ce869cea7f | 3 | 0.6722 | 0.6718 |
| average_p64_mlp_add | 1882162633 | f51bda1276aa4001b10b3fd0 | 3 | 0.6664 | 0.6634 |
| average_p64_mlp_replace | 476981973 | 29150c5d8d954e0290d7d385 | 17 | 0.3090 | 0.3109 |
| average_p64_mlp_replace | 621715752 | d7c1e1da65c046d1abf4fa6b | 14 | 0.3584 | 0.3565 |
| average_p64_mlp_replace | 1882162633 | e8cc1e1c20d74f8781505010 | 27 | 0.3784 | 0.3680 |
| average_p32_linear_add | 476981973 | 1df83eaccb84416c944f8988 | 1 | 0.6662 | 0.6621 |
| average_p32_linear_add | 621715752 | b7df9c9e4a624d14bb30cee4 | 2 | 0.6700 | 0.6609 |
| average_p32_linear_add | 1882162633 | 5aa5db45b74a4093ae84f786 | 2 | 0.6704 | 0.6617 |
| average_p32_linear_replace | 476981973 | 3fba7492dc124ebf8fcf9218 | 37 | 0.4116 | 0.4135 |
| average_p32_linear_replace | 621715752 | 2e771e92dd27425aa99f453c | 26 | 0.4018 | 0.4072 |
| average_p32_linear_replace | 1882162633 | 11af7ec5cee942b8854602de | 36 | 0.3996 | 0.3972 |
| average_p32_mlp_add | 476981973 | 924a952ab73743a9b149c8ea | 2 | 0.6704 | 0.6645 |
| average_p32_mlp_add | 621715752 | e42ee839cbd54636a49e15e6 | 1 | 0.6726 | 0.6624 |
| average_p32_mlp_add | 1882162633 | 43ca432b3773429fb605f57b | 3 | 0.6698 | 0.6610 |
| average_p32_mlp_replace | 476981973 | d736117945b64188be27f5bb | 32 | 0.3634 | 0.3625 |
| average_p32_mlp_replace | 621715752 | 4496487a78924c2ca57c4170 | 30 | 0.3788 | 0.3830 |
| average_p32_mlp_replace | 1882162633 | fc222b73df1e43f9b9a4b1bd | 32 | 0.3852 | 0.3767 |
| gate_none_linear_add | 476981973 | b0485a6703ed47918bbc340b | 4 | 0.6644 | 0.6587 |
| gate_none_linear_add | 621715752 | d8072d69f9c54d29b94bfe4b | 1 | 0.6664 | 0.6602 |
| gate_none_linear_add | 1882162633 | 24c3038487e54f5cb869ba8f | 3 | 0.6664 | 0.6694 |
| gate_none_linear_replace | 476981973 | 14699dd4591944d1828948cd | 28 | 0.3626 | 0.3473 |
| gate_none_linear_replace | 621715752 | abbee5cb2f2444809cb7c60a | 17 | 0.3362 | 0.3367 |
| gate_none_linear_replace | 1882162633 | 7ac643b48c484df789d80b49 | 16 | 0.3526 | 0.3513 |
| gate_none_mlp_add | 476981973 | c0e56851f31844cd8a5186c5 | 1 | 0.6710 | 0.6690 |
| gate_none_mlp_add | 621715752 | 0a727fede13b4bce8b4608ca | 4 | 0.6730 | 0.6649 |
| gate_none_mlp_add | 1882162633 | 4dc1c22fa7b64e529450a722 | 4 | 0.6674 | 0.6631 |
| gate_none_mlp_replace | 476981973 | cdc67db293604ec49cf39bea | 24 | 0.2826 | 0.2830 |
| gate_none_mlp_replace | 621715752 | 2d62e71e0ea544229236ca4f | 38 | 0.3010 | 0.3021 |
| gate_none_mlp_replace | 1882162633 | 618c443faa154cdaa8362c6c | 30 | 0.3346 | 0.3307 |
| gate_p64_linear_add | 476981973 | 8cb7013785994b7bb5cdd8fb | 1 | 0.6660 | 0.6616 |
| gate_p64_linear_add | 621715752 | 083aedc10cb94669ba63c196 | 4 | 0.6698 | 0.6689 |
| gate_p64_linear_add | 1882162633 | bd1077d7b4db4803b8a07b39 | 2 | 0.6678 | 0.6650 |
| gate_p64_linear_replace | 476981973 | 99a8ab72d4f44598b779c85a | 20 | 0.3798 | 0.3753 |
| gate_p64_linear_replace | 621715752 | 282051be28e745308ea0d3e6 | 13 | 0.3590 | 0.3592 |
| gate_p64_linear_replace | 1882162633 | a9e2db739943440089367845 | 38 | 0.2878 | 0.2903 |
| gate_p64_mlp_add | 476981973 | 979e47f8d52a48d097612043 | 1 | 0.6654 | 0.6618 |
| gate_p64_mlp_add | 621715752 | ea83fba62c1e4d80bd85e0b6 | 2 | 0.6706 | 0.6666 |
| gate_p64_mlp_add | 1882162633 | af4dc6e2c6ba40859fbc6c9e | 1 | 0.6630 | 0.6607 |
| gate_p64_mlp_replace | 476981973 | bb7f7a87d1cc48d894d9d967 | 15 | 0.2888 | 0.2859 |
| gate_p64_mlp_replace | 621715752 | da4eb17cb077420c9b9ff16a | 17 | 0.3236 | 0.3242 |
| gate_p64_mlp_replace | 1882162633 | b3b8b56059334edb87b89f50 | 21 | 0.3208 | 0.3133 |
| gate_p32_linear_add | 476981973 | 87c1487af49e4f52b02899bb | 1 | 0.6662 | 0.6634 |
| gate_p32_linear_add | 621715752 | 09dd4a39c8f24f72b78b9d07 | 1 | 0.6684 | 0.6642 |
| gate_p32_linear_add | 1882162633 | 5e25bc2fca9b484785f48df4 | 2 | 0.6704 | 0.6637 |
| gate_p32_linear_replace | 476981973 | c92940e9b7d647188dfadbf7 | 32 | 0.3608 | 0.3606 |
| gate_p32_linear_replace | 621715752 | aac151fc21124c39ac95cd67 | 19 | 0.3740 | 0.3746 |
| gate_p32_linear_replace | 1882162633 | cbb91b7cc27c43cba59ce0fe | 28 | 0.3834 | 0.3754 |
| gate_p32_mlp_add | 476981973 | cc14d903c31148a49f4486d2 | 4 | 0.6724 | 0.6685 |
| gate_p32_mlp_add | 621715752 | 840775d152d4459e92475b84 | 2 | 0.6654 | 0.6614 |
| gate_p32_mlp_add | 1882162633 | 7492631016d6440888ee0c86 | 6 | 0.6736 | 0.6657 |
| gate_p32_mlp_replace | 476981973 | fd8da147701f4c869c3e862f | 33 | 0.3458 | 0.3471 |
| gate_p32_mlp_replace | 621715752 | 657f74a57687499a930a01e7 | 35 | 0.3520 | 0.3473 |
| gate_p32_mlp_replace | 1882162633 | 70471c742a294c6fa5ddbb53 | 30 | 0.3524 | 0.3446 |
| attention_none_linear_add | 476981973 | 1f3a64c9cd9d4c49a40694ec | 1 | 0.6676 | 0.6640 |
| attention_none_linear_add | 621715752 | b2d0145eefcd42c6988426d2 | 2 | 0.6678 | 0.6644 |
| attention_none_linear_add | 1882162633 | fdc1ab6077c042ae90e3aabc | 2 | 0.6694 | 0.6630 |
| attention_none_linear_replace | 476981973 | 33149ed92ef74909abc04901 | 28 | 0.3450 | 0.3425 |
| attention_none_linear_replace | 621715752 | 57ed4579e92c4b57aa03449e | 20 | 0.3248 | 0.3273 |
| attention_none_linear_replace | 1882162633 | c271b4310c404d7382049ef5 | 22 | 0.3618 | 0.3598 |
| attention_none_mlp_add | 476981973 | 7f0eb208319e40d491fac13e | 3 | 0.6676 | 0.6597 |
| attention_none_mlp_add | 621715752 | ba7d217003314a9fa957e92d | 2 | 0.6658 | 0.6628 |
| attention_none_mlp_add | 1882162633 | 41568e17ae524f9a84a5ccf3 | 2 | 0.6702 | 0.6687 |
| attention_none_mlp_replace | 476981973 | 332b5aad0e05482ab6c079f5 | 16 | 0.3272 | 0.3231 |
| attention_none_mlp_replace | 621715752 | f4121593fe674a128961eeae | 16 | 0.3238 | 0.3284 |
| attention_none_mlp_replace | 1882162633 | a295a6c4a4f84064b1bf3f60 | 27 | 0.3310 | 0.3321 |
| attention_p64_linear_add | 476981973 | dfa9f99367ca474ab4e62715 | 1 | 0.6686 | 0.6621 |
| attention_p64_linear_add | 621715752 | 26240e71a638446db0471ef4 | 3 | 0.6698 | 0.6664 |
| attention_p64_linear_add | 1882162633 | bb2e2f68a7b14868ac9d26c2 | 1 | 0.6662 | 0.6630 |
| attention_p64_linear_replace | 476981973 | 86d39e1d48c64b7cb2cf7566 | 19 | 0.3700 | 0.3677 |
| attention_p64_linear_replace | 621715752 | 831b93329b824c2793ea81be | 26 | 0.3674 | 0.3667 |
| attention_p64_linear_replace | 1882162633 | 7b7b705a71734d7fa87bcc97 | 18 | 0.3698 | 0.3696 |
| attention_p64_mlp_add | 476981973 | 86aae3522df54e84821efc4b | 1 | 0.6676 | 0.6637 |
| attention_p64_mlp_add | 621715752 | 6509c9b4faaf4dad8a646491 | 2 | 0.6704 | 0.6675 |
| attention_p64_mlp_add | 1882162633 | 6794f5fa252d4dbf87a25caf | 1 | 0.6712 | 0.6661 |
| attention_p64_mlp_replace | 476981973 | 9ee2a4b130384138bf53c2c4 | 21 | 0.3290 | 0.3259 |
| attention_p64_mlp_replace | 621715752 | 5f707378e1c14df2996dfaf5 | 18 | 0.3286 | 0.3329 |
| attention_p64_mlp_replace | 1882162633 | 3868c9514f2f45a39d79f9da | 18 | 0.3354 | 0.3490 |
| attention_p32_linear_add | 476981973 | fff47547f1b14370913da268 | 1 | 0.6676 | 0.6625 |
| attention_p32_linear_add | 621715752 | 806450de1c54426e9361fb5a | 2 | 0.6646 | 0.6634 |
| attention_p32_linear_add | 1882162633 | 32306ad3d40e4aa988507a6e | 4 | 0.6688 | 0.6687 |
| attention_p32_linear_replace | 476981973 | 80e042163e184b4abe7adf85 | 19 | 0.3568 | 0.3649 |
| attention_p32_linear_replace | 621715752 | 23eaa136b8a6445381e2d832 | 20 | 0.3670 | 0.3573 |
| attention_p32_linear_replace | 1882162633 | 49d856398009405585680d63 | 32 | 0.3820 | 0.3749 |
| attention_p32_mlp_add | 476981973 | a96786c568394aef8bfab250 | 5 | 0.6722 | 0.6686 |
| attention_p32_mlp_add | 621715752 | bef0d02c4f8444c3b8fe57e6 | 3 | 0.6646 | 0.6646 |
| attention_p32_mlp_add | 1882162633 | 5ddd18a6bc184cff96ef0ece | 2 | 0.6728 | 0.6665 |
| attention_p32_mlp_replace | 476981973 | 549e627675b240e192a835f6 | 34 | 0.3036 | 0.3101 |
| attention_p32_mlp_replace | 621715752 | e12542557dab4a4fbb92f359 | 30 | 0.3404 | 0.3412 |
| attention_p32_mlp_replace | 1882162633 | cca4b5e930f74a75bb2278be | 36 | 0.3748 | 0.3658 |
| attention_zq_none_linear_add | 476981973 | 45d938e91cca4b88ac4353a7 | 3 | 0.6658 | 0.6647 |
| attention_zq_none_linear_add | 621715752 | 39272b7dc5594c84985a3381 | 2 | 0.6694 | 0.6693 |
| attention_zq_none_linear_add | 1882162633 | 18c4ad7080cd444d911fa2a6 | 4 | 0.6674 | 0.6664 |
| attention_zq_none_linear_replace | 476981973 | 99ad0443e78247bba66a332f | 38 | 0.3528 | 0.3521 |
| attention_zq_none_linear_replace | 621715752 | 4736857f94934e65ade621fd | 14 | 0.3278 | 0.3274 |
| attention_zq_none_linear_replace | 1882162633 | c820f4a513c947fd8b3628fb | 27 | 0.3514 | 0.3603 |
| attention_zq_none_mlp_add | 476981973 | e4aea694ee6a4d03b3ce4ce7 | 3 | 0.6724 | 0.6630 |
| attention_zq_none_mlp_add | 621715752 | a16b26bfbc994f7eb7041c3b | 2 | 0.6666 | 0.6601 |
| attention_zq_none_mlp_add | 1882162633 | 3ad6a7e9060f45598f5426ac | 2 | 0.6688 | 0.6657 |
| attention_zq_none_mlp_replace | 476981973 | d37be4708b7e42e89ee45efb | 28 | 0.3130 | 0.3221 |
| attention_zq_none_mlp_replace | 621715752 | 18c2cc35fca843b988ea3e57 | 21 | 0.3344 | 0.3385 |
| attention_zq_none_mlp_replace | 1882162633 | c09908219f50488b80ce61db | 21 | 0.3222 | 0.3290 |
| attention_zq_p64_linear_add | 476981973 | 4dff291ac7d1414d8caa7240 | 1 | 0.6698 | 0.6632 |
| attention_zq_p64_linear_add | 621715752 | 32292683a9244c649929a5e7 | 1 | 0.6666 | 0.6617 |
| attention_zq_p64_linear_add | 1882162633 | d0419e45587348fd96a6cd09 | 1 | 0.6662 | 0.6633 |
| attention_zq_p64_linear_replace | 476981973 | 94baf067bab846c9a993c258 | 23 | 0.3630 | 0.3700 |
| attention_zq_p64_linear_replace | 621715752 | 057666efb3724ceba9ba48d3 | 14 | 0.3620 | 0.3613 |
| attention_zq_p64_linear_replace | 1882162633 | 1dcdcbdcadcb4d578c236f85 | 21 | 0.3708 | 0.3689 |
| attention_zq_p64_mlp_add | 476981973 | 0b719211154146bfb6da23a7 | 1 | 0.6704 | 0.6628 |
| attention_zq_p64_mlp_add | 621715752 | df4a60e1a1c54af5a132ceaf | 3 | 0.6710 | 0.6658 |
| attention_zq_p64_mlp_add | 1882162633 | 9ff244512c174a0cb8f67f8a | 2 | 0.6722 | 0.6693 |
| attention_zq_p64_mlp_replace | 476981973 | 82d36e24df0e4a63b840b49e | 23 | 0.3200 | 0.3228 |
| attention_zq_p64_mlp_replace | 621715752 | 49e89e132a5c448ab0d6b27b | 18 | 0.3322 | 0.3382 |
| attention_zq_p64_mlp_replace | 1882162633 | d353ada8ffc94f2b9f57c881 | 21 | 0.3346 | 0.3311 |
| attention_zq_p32_linear_add | 476981973 | 23854e4e2e084d958764b264 | 1 | 0.6696 | 0.6622 |
| attention_zq_p32_linear_add | 621715752 | e49a144294574ac386fd925c | 1 | 0.6662 | 0.6626 |
| attention_zq_p32_linear_add | 1882162633 | 40aa5bc49bb642c486ace699 | 1 | 0.6690 | 0.6637 |
| attention_zq_p32_linear_replace | 476981973 | fd835151276b49e595355a5c | 35 | 0.3530 | 0.3526 |
| attention_zq_p32_linear_replace | 621715752 | 7a19123815de42f68fa71ae4 | 26 | 0.3752 | 0.3772 |
| attention_zq_p32_linear_replace | 1882162633 | 5a659edb3652401c9618fc18 | 25 | 0.3784 | 0.3794 |
| attention_zq_p32_mlp_add | 476981973 | 1c8480311bb44c4481944984 | 3 | 0.6746 | 0.6648 |
| attention_zq_p32_mlp_add | 621715752 | a3d6988fe1954e1ebc43ebdd | 1 | 0.6656 | 0.6590 |
| attention_zq_p32_mlp_add | 1882162633 | 33171a526f79476096f0ebde | 2 | 0.6708 | 0.6647 |
| attention_zq_p32_mlp_replace | 476981973 | e5c31a40609f448f8269c225 | 41 | 0.3426 | 0.3435 |
| attention_zq_p32_mlp_replace | 621715752 | c8f4d6d9a8a24efbbdc1bae8 | 31 | 0.3544 | 0.3633 |
| attention_zq_p32_mlp_replace | 1882162633 | b6c8e0f58b234a0c9a1afdaf | 33 | 0.3502 | 0.3428 |

## Other metrics

All values are means over seeds, on val at the kept epoch unless noted.

### Answer shift

| Arm | Corrected | Corrupted | Net | Flip rate | Agreement pre to post | KL |
|---|---|---|---|---|---|---|
| `average_none_linear_add` | 2.96% | 5.29% | -2.33% | 15.4% | 0.678 to 0.622 | 0.55 |
| `average_none_linear_replace` | 4.17% | 47.09% | -42.92% | 79.6% | 0.678 to 0.047 | 11.58 |
| `average_none_mlp_add` | 3.45% | 9.23% | -5.78% | 22.7% | 0.678 to 0.527 | 1.54 |
| `average_none_mlp_replace` | 3.78% | 48.89% | -45.11% | 81.8% | 0.678 to 0.027 | 12.44 |
| `average_p64_linear_add` | 2.05% | 3.58% | -1.53% | 10.5% | 0.678 to 0.639 | 0.37 |
| `average_p64_linear_replace` | 4.85% | 45.00% | -40.15% | 77.0% | 0.678 to 0.064 | 10.51 |
| `average_p64_mlp_add` | 3.59% | 7.61% | -4.02% | 20.1% | 0.678 to 0.567 | 1.06 |
| `average_p64_mlp_replace` | 4.00% | 47.94% | -43.93% | 80.6% | 0.678 to 0.028 | 12.18 |
| `average_p32_linear_add` | 2.05% | 2.82% | -0.77% | 9.4% | 0.678 to 0.658 | 0.16 |
| `average_p32_linear_replace` | 4.79% | 44.18% | -39.38% | 76.0% | 0.678 to 0.072 | 10.28 |
| `average_p32_mlp_add` | 2.96% | 5.42% | -2.47% | 15.3% | 0.678 to 0.608 | 0.67 |
| `average_p32_mlp_replace` | 4.18% | 46.90% | -42.72% | 79.3% | 0.678 to 0.034 | 11.63 |
| `gate_none_linear_add` | 3.23% | 6.72% | -3.49% | 18.2% | 0.678 to 0.590 | 0.91 |
| `gate_none_linear_replace` | 3.94% | 47.68% | -43.75% | 80.2% | 0.678 to 0.041 | 11.76 |
| `gate_none_mlp_add` | 3.62% | 8.00% | -4.38% | 21.0% | 0.678 to 0.559 | 1.19 |
| `gate_none_mlp_replace` | 3.70% | 48.86% | -45.15% | 81.8% | 0.678 to 0.033 | 12.59 |
| `gate_p64_linear_add` | 2.79% | 5.03% | -2.23% | 14.3% | 0.678 to 0.621 | 0.58 |
| `gate_p64_linear_replace` | 4.02% | 47.25% | -43.23% | 79.9% | 0.678 to 0.047 | 11.66 |
| `gate_p64_mlp_add` | 3.06% | 5.53% | -2.46% | 16.1% | 0.678 to 0.615 | 0.60 |
| `gate_p64_mlp_replace` | 3.78% | 49.20% | -45.42% | 82.3% | 0.678 to 0.024 | 12.78 |
| `gate_p32_linear_add` | 1.59% | 2.03% | -0.44% | 7.2% | 0.678 to 0.670 | 0.07 |
| `gate_p32_linear_replace` | 4.34% | 46.30% | -41.97% | 78.6% | 0.678 to 0.053 | 11.17 |
| `gate_p32_mlp_add` | 3.72% | 8.17% | -4.45% | 21.3% | 0.678 to 0.557 | 1.23 |
| `gate_p32_mlp_replace` | 4.07% | 47.80% | -43.73% | 80.4% | 0.678 to 0.034 | 12.00 |
| `attention_none_linear_add` | 2.92% | 7.03% | -4.11% | 18.1% | 0.678 to 0.575 | 1.04 |
| `attention_none_linear_replace` | 3.92% | 47.40% | -43.48% | 80.0% | 0.678 to 0.044 | 11.59 |
| `attention_none_mlp_add` | 3.47% | 8.20% | -4.73% | 21.0% | 0.678 to 0.554 | 1.27 |
| `attention_none_mlp_replace` | 3.79% | 48.80% | -45.02% | 81.6% | 0.678 to 0.029 | 12.37 |
| `attention_p64_linear_add` | 2.04% | 3.98% | -1.95% | 11.3% | 0.678 to 0.631 | 0.47 |
| `attention_p64_linear_replace` | 4.16% | 46.28% | -42.12% | 78.7% | 0.678 to 0.055 | 11.25 |
| `attention_p64_mlp_add` | 2.96% | 6.12% | -3.16% | 16.4% | 0.678 to 0.591 | 0.78 |
| `attention_p64_mlp_replace` | 3.70% | 48.88% | -45.18% | 81.7% | 0.678 to 0.028 | 12.35 |
| `attention_p32_linear_add` | 2.29% | 4.44% | -2.15% | 12.5% | 0.678 to 0.625 | 0.51 |
| `attention_p32_linear_replace` | 4.33% | 46.93% | -42.60% | 79.2% | 0.678 to 0.039 | 11.45 |
| `attention_p32_mlp_add` | 3.64% | 8.32% | -4.68% | 21.4% | 0.678 to 0.551 | 1.23 |
| `attention_p32_mlp_replace` | 4.02% | 47.97% | -43.95% | 80.6% | 0.678 to 0.037 | 12.24 |
| `attention_zq_none_linear_add` | 3.65% | 8.93% | -5.28% | 22.6% | 0.678 to 0.544 | 1.37 |
| `attention_zq_none_linear_replace` | 3.96% | 47.34% | -43.38% | 79.9% | 0.678 to 0.044 | 11.58 |
| `attention_zq_none_mlp_add` | 3.59% | 7.95% | -4.36% | 20.8% | 0.678 to 0.560 | 1.18 |
| `attention_zq_none_mlp_replace` | 3.76% | 48.89% | -45.13% | 81.5% | 0.678 to 0.028 | 12.26 |
| `attention_zq_p64_linear_add` | 1.69% | 2.06% | -0.37% | 7.4% | 0.678 to 0.669 | 0.07 |
| `attention_zq_p64_linear_replace` | 4.12% | 46.64% | -42.52% | 79.0% | 0.678 to 0.048 | 11.21 |
| `attention_zq_p64_mlp_add` | 3.09% | 6.25% | -3.16% | 17.2% | 0.678 to 0.591 | 0.84 |
| `attention_zq_p64_mlp_replace` | 3.70% | 48.79% | -45.09% | 81.7% | 0.678 to 0.030 | 12.58 |
| `attention_zq_p32_linear_add` | 1.24% | 1.46% | -0.21% | 5.5% | 0.678 to 0.675 | 0.04 |
| `attention_zq_p32_linear_replace` | 4.38% | 46.77% | -42.40% | 79.1% | 0.678 to 0.046 | 11.41 |
| `attention_zq_p32_mlp_add` | 3.21% | 7.51% | -4.30% | 19.5% | 0.678 to 0.565 | 1.09 |
| `attention_zq_p32_mlp_replace` | 4.03% | 47.45% | -43.42% | 80.1% | 0.678 to 0.039 | 11.74 |

Net = corrected - corrupted, per member. KL is KL(post ‖ pre).

### Mixing weights (gate and attention arms)

| Arm | Mix weights (sender 0/1/2) | Self weight start to kept | Top-1 | Entropy start to kept | Joint mode share | Route flip | Self-attn rate | Routed correct (all/corr/corrupt) |
|---|---|---|---|---|---|---|---|---|
| `gate_none_linear_add` | 0.33 / 0.33 / 0.33 | 0.33 to 0.33 | 0.997 | 0.146 to 0.005 | 1.000 | 0.000 | 0.333 | 0.631 / 0.444 / 0.676 |
| `gate_none_linear_replace` | 0.00 / 0.33 / 0.67 | 0.33 to 0.33 | 1.000 | 0.342 to 0.000 | 1.000 | 0.000 | 0.333 | 0.634 / 0.346 / 0.853 |
| `gate_none_mlp_add` | 0.66 / 0.00 / 0.34 | 0.33 to 0.33 | 0.997 | 0.150 to 0.006 | 1.000 | 0.000 | 0.333 | 0.634 / 0.470 / 0.671 |
| `gate_none_mlp_replace` | 0.33 / 0.56 / 0.11 | 0.33 to 0.33 | 0.993 | 0.128 to 0.018 | 0.905 | 0.018 | 0.333 | 0.627 / 0.342 / 0.837 |
| `gate_p64_linear_add` | 0.33 / 0.33 / 0.33 | 0.33 to 0.33 | 0.997 | 0.214 to 0.005 | 1.000 | 0.000 | 0.333 | 0.631 / 0.468 / 0.647 |
| `gate_p64_linear_replace` | 0.67 / 0.33 / 0.00 | 0.33 to 0.33 | 1.000 | 0.183 to 0.000 | 1.000 | 0.000 | 0.333 | 0.628 / 0.388 / 0.826 |
| `gate_p64_mlp_add` | 0.58 / 0.08 / 0.34 | 0.33 to 0.33 | 0.992 | 0.142 to 0.014 | 1.000 | 0.000 | 0.333 | 0.631 / 0.443 / 0.662 |
| `gate_p64_mlp_replace` | 0.48 / 0.52 / 0.00 | 0.33 to 0.33 | 0.995 | 0.199 to 0.013 | 0.840 | 0.023 | 0.333 | 0.625 / 0.368 / 0.825 |
| `gate_p32_linear_add` | 0.38 / 0.62 / 0.01 | 0.33 to 0.33 | 0.973 | 0.285 to 0.062 | 1.000 | 0.000 | 0.333 | 0.625 / 0.495 / 0.536 |
| `gate_p32_linear_replace` | 0.23 / 0.62 / 0.15 | 0.33 to 0.33 | 0.940 | 0.418 to 0.146 | 0.679 | 0.048 | 0.333 | 0.625 / 0.355 / 0.833 |
| `gate_p32_mlp_add` | 0.33 / 0.33 / 0.33 | 0.33 to 0.33 | 1.000 | 0.188 to 0.000 | 1.000 | 0.000 | 0.333 | 0.631 / 0.420 / 0.740 |
| `gate_p32_mlp_replace` | 0.95 / 0.00 / 0.05 | 0.33 to 0.33 | 0.997 | 0.157 to 0.008 | 0.942 | 0.010 | 0.333 | 0.634 / 0.433 / 0.828 |
| `attention_none_linear_add` | 0.28 / 0.30 / 0.42 | 0.35 to 0.39 | 0.998 | 0.099 to 0.004 | 1.000 | 0.333 | 0.444 | 0.635 / 0.467 / 0.661 |
| `attention_none_linear_replace` | 0.33 / 0.33 / 0.33 | 0.23 to 0.11 | 1.000 | 0.054 to 0.000 | 1.000 | 0.000 | 0.111 | 0.632 / 0.379 / 0.835 |
| `attention_none_mlp_add` | 0.44 / 0.22 / 0.33 | 0.42 to 0.44 | 1.000 | 0.089 to 0.000 | 1.000 | 0.000 | 0.444 | 0.631 / 0.475 / 0.680 |
| `attention_none_mlp_replace` | 0.33 / 0.33 / 0.33 | 0.14 to 0.11 | 1.000 | 0.051 to 0.000 | 1.000 | 0.000 | 0.111 | 0.632 / 0.365 / 0.839 |
| `attention_p64_linear_add` | 0.23 / 0.66 / 0.11 | 0.07 to 0.01 | 0.996 | 0.098 to 0.009 | 0.999 | 0.000 | 0.000 | 0.626 / 0.493 / 0.586 |
| `attention_p64_linear_replace` | 0.44 / 0.11 / 0.44 | 0.52 to 0.44 | 1.000 | 0.051 to 0.000 | 1.000 | 0.000 | 0.444 | 0.634 / 0.400 / 0.821 |
| `attention_p64_mlp_add` | 0.22 / 0.74 / 0.03 | 0.33 to 0.25 | 0.996 | 0.099 to 0.008 | 1.000 | 0.334 | 0.222 | 0.625 / 0.433 / 0.602 |
| `attention_p64_mlp_replace` | 0.22 / 0.22 / 0.56 | 0.63 to 0.56 | 1.000 | 0.050 to 0.000 | 1.000 | 0.000 | 0.556 | 0.634 / 0.337 / 0.870 |
| `attention_p32_linear_add` | 0.41 / 0.15 / 0.44 | 0.28 to 0.34 | 0.998 | 0.126 to 0.005 | 1.000 | 0.000 | 0.333 | 0.635 / 0.522 / 0.630 |
| `attention_p32_linear_replace` | 0.22 / 0.33 / 0.44 | 0.35 to 0.33 | 1.000 | 0.079 to 0.000 | 1.000 | 0.000 | 0.333 | 0.631 / 0.334 / 0.856 |
| `attention_p32_mlp_add` | 0.33 / 0.00 / 0.67 | 0.29 to 0.33 | 1.000 | 0.136 to 0.000 | 1.000 | 0.000 | 0.333 | 0.637 / 0.450 / 0.702 |
| `attention_p32_mlp_replace` | 0.22 / 0.33 / 0.44 | 0.41 to 0.44 | 1.000 | 0.073 to 0.000 | 1.000 | 0.000 | 0.444 | 0.629 / 0.348 / 0.842 |
| `attention_zq_none_linear_add` | 0.22 / 0.44 / 0.33 | 0.51 to 0.56 | 1.000 | 0.144 to 0.000 | 1.000 | 0.000 | 0.556 | 0.631 / 0.454 / 0.671 |
| `attention_zq_none_linear_replace` | 0.33 / 0.44 / 0.22 | 0.33 to 0.33 | 1.000 | 0.076 to 0.000 | 1.000 | 0.000 | 0.333 | 0.629 / 0.336 / 0.846 |
| `attention_zq_none_mlp_add` | 0.22 / 0.22 / 0.56 | 0.30 to 0.33 | 1.000 | 0.143 to 0.000 | 1.000 | 0.000 | 0.333 | 0.635 / 0.441 / 0.687 |
| `attention_zq_none_mlp_replace` | 0.56 / 0.00 / 0.44 | 0.47 to 0.56 | 1.000 | 0.082 to 0.000 | 1.000 | 0.000 | 0.556 | 0.635 / 0.357 / 0.857 |
| `attention_zq_p64_linear_add` | 0.54 / 0.44 / 0.02 | 0.44 to 0.44 | 0.990 | 0.140 to 0.018 | 1.000 | nan | 0.444 | 0.628 / 0.491 / 0.555 |
| `attention_zq_p64_linear_replace` | 0.33 / 0.33 / 0.33 | 0.48 to 0.44 | 1.000 | 0.075 to 0.000 | 1.000 | 0.000 | 0.444 | 0.633 / 0.387 / 0.827 |
| `attention_zq_p64_mlp_add` | 0.22 / 0.44 / 0.34 | 0.38 to 0.33 | 0.997 | 0.140 to 0.006 | 1.000 | 0.000 | 0.333 | 0.629 / 0.449 / 0.690 |
| `attention_zq_p64_mlp_replace` | 0.33 / 0.56 / 0.11 | 0.33 to 0.33 | 1.000 | 0.078 to 0.000 | 1.000 | 0.000 | 0.333 | 0.629 / 0.371 / 0.831 |
| `attention_zq_p32_linear_add` | 0.21 / 0.56 / 0.23 | 0.39 to 0.24 | 0.989 | 0.162 to 0.021 | 1.000 | nan | 0.222 | 0.628 / 0.492 / 0.558 |
| `attention_zq_p32_linear_replace` | 0.00 / 0.44 / 0.56 | 0.46 to 0.56 | 1.000 | 0.091 to 0.000 | 1.000 | 0.000 | 0.556 | 0.637 / 0.420 / 0.832 |
| `attention_zq_p32_mlp_add` | 0.33 / 0.23 / 0.44 | 0.31 to 0.22 | 0.995 | 0.188 to 0.009 | 1.000 | 0.038 | 0.222 | 0.631 / 0.440 / 0.639 |
| `attention_zq_p32_mlp_replace` | 0.44 / 0.44 / 0.11 | 0.31 to 0.33 | 1.000 | 0.096 to 0.000 | 1.000 | 0.000 | 0.333 | 0.627 / 0.339 / 0.845 |

- Mix weights: mean weight each sender gets, over receivers and the kept epoch. A row like 0.33 / 0.00 / 0.67 means one receiver reads sender 0 and two read sender 2.
- Self weight and self-attention rate are 1/3 by construction for the gate: every receiver gets the same weights, so the diagonal is the gate itself.
- Entropy start is the mean of the first 20 steps. Maximum is ln 3 = 1.10. Top-1 and entropy are train-step values averaged over the kept epoch.
- Route flip is blank (nan) for runs whose kept epoch is the first, since there is no earlier epoch to compare to.

How fast the weights lock in, mean over all runs of each mix (train steps):

| Mix | Step 0 | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| gate, entropy | 1.10 | 0.73 | 0.45 | 0.32 | 0.17 | 0.09 | 0.07 |
| gate, top-1 | 0.33 | 0.59 | 0.80 | 0.87 | 0.94 | 0.96 | 0.97 |
| attention (no projection), entropy | 0.79 | 0.49 | 0.13 | 0.03 | 0.01 | 0.00 | 0.00 |
| attention (projection), entropy | 0.81 | 0.54 | 0.21 | 0.11 | 0.02 | 0.00 | 0.00 |
| attention_zq, entropy | 1.10 | 0.74 | 0.31 | 0.10 | 0.02 | 0.00 | 0.00 |
| attention_zq, top-1 | 0.33 | 0.59 | 0.87 | 0.96 | 0.99 | 1.00 | 1.00 |

Per seed, the gate picks a different member each time. Gate weights at the kept epoch, additive arms:

| Arm | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| `gate_none_linear_add` | 1.00 / 0.00 / 0.00 | 0.00 / 0.99 / 0.00 | 0.00 / 0.00 / 1.00 |
| `gate_none_mlp_add` | 0.99 / 0.00 / 0.01 | 1.00 / 0.00 / 0.00 | 0.00 / 0.00 / 1.00 |
| `gate_p64_linear_add` | 0.99 / 0.00 / 0.00 | 0.00 / 0.00 / 1.00 | 0.00 / 1.00 / 0.00 |
| `gate_p64_mlp_add` | 0.75 / 0.25 / 0.00 | 0.00 / 0.00 / 1.00 | 0.99 / 0.01 / 0.01 |
| `gate_p32_linear_add` | 0.13 / 0.86 / 0.01 | 0.02 / 0.98 / 0.01 | 0.99 / 0.01 / 0.00 |
| `gate_p32_mlp_add` | 1.00 / 0.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 0.00 / 1.00 |

Gradient norms at the kept epoch: query/key heads are 0 to 5e-6 in every replace attention arm and at most 1e-2 in additive ones. The gate's score head is 5e-5 to 5e-2 in the additive arms. Once the softmax is one-hot, almost no gradient reaches the heads.

### Fit and message size

| Arm | Val after epoch 1 | Train at kept | Train at last | Value norm | Injection norm start to kept to last |
|---|---|---|---|---|---|
| `average_none_linear_add` | 0.666 | 0.672 | 0.709 | 127.1 | 12.3 to 73.2 to 172.6 |
| `average_none_linear_replace` | 0.079 | 0.542 | 0.606 | 127.1 | 92.1 to 338.3 to 351.6 |
| `average_none_mlp_add` | 0.665 | 0.668 | 0.701 | 127.1 | 9.3 to 117.7 to 172.7 |
| `average_none_mlp_replace` | 0.070 | 0.487 | 0.559 | 127.1 | 62.6 to 491.5 to 513.0 |
| `average_p64_linear_add` | 0.667 | 0.667 | 0.717 | 116.8 | 11.8 to 42.0 to 125.2 |
| `average_p64_linear_replace` | 0.086 | 0.586 | 0.680 | 187.1 | 99.6 to 334.0 to 352.6 |
| `average_p64_mlp_add` | 0.664 | 0.671 | 0.720 | 129.0 | 9.3 to 124.4 to 148.8 |
| `average_p64_mlp_replace` | 0.066 | 0.518 | 0.615 | 177.6 | 68.0 to 359.2 to 388.4 |
| `average_p32_linear_add` | 0.668 | 0.667 | 0.691 | 68.7 | 12.1 to 33.2 to 156.1 |
| `average_p32_linear_replace` | 0.077 | 0.585 | 0.640 | 152.4 | 95.9 to 362.4 to 376.3 |
| `average_p32_mlp_add` | 0.669 | 0.667 | 0.685 | 88.7 | 9.5 to 77.0 to 142.1 |
| `average_p32_mlp_replace` | 0.060 | 0.535 | 0.595 | 146.4 | 62.9 to 470.1 to 489.5 |
| `gate_none_linear_add` | 0.664 | 0.669 | 0.712 | 127.1 | 9.3 to 92.3 to 168.5 |
| `gate_none_linear_replace` | 0.090 | 0.512 | 0.594 | 127.1 | 103.5 to 349.4 to 366.5 |
| `gate_none_mlp_add` | 0.665 | 0.671 | 0.715 | 127.1 | 8.7 to 98.5 to 161.5 |
| `gate_none_mlp_replace` | 0.066 | 0.541 | 0.598 | 127.1 | 69.8 to 431.4 to 453.0 |
| `gate_p64_linear_add` | 0.666 | 0.670 | 0.726 | 123.0 | 11.4 to 60.1 to 139.2 |
| `gate_p64_linear_replace` | 0.093 | 0.528 | 0.606 | 128.7 | 108.0 to 359.3 to 377.6 |
| `gate_p64_mlp_add` | 0.666 | 0.669 | 0.711 | 130.5 | 9.7 to 55.9 to 142.5 |
| `gate_p64_mlp_replace` | 0.065 | 0.456 | 0.550 | 131.9 | 79.5 to 353.4 to 379.7 |
| `gate_p32_linear_add` | 0.667 | 0.668 | 0.692 | 77.0 | 10.3 to 22.7 to 143.2 |
| `gate_p32_linear_replace` | 0.083 | 0.512 | 0.569 | 87.6 | 85.9 to 313.7 to 329.2 |
| `gate_p32_mlp_add` | 0.663 | 0.674 | 0.698 | 87.5 | 8.8 to 114.5 to 146.3 |
| `gate_p32_mlp_replace` | 0.061 | 0.495 | 0.543 | 86.8 | 69.2 to 541.6 to 565.7 |
| `attention_none_linear_add` | 0.666 | 0.667 | 0.716 | 127.1 | 11.9 to 71.2 to 165.2 |
| `attention_none_linear_replace` | 0.082 | 0.527 | 0.594 | 127.1 | 108.2 to 360.4 to 376.7 |
| `attention_none_mlp_add` | 0.662 | 0.667 | 0.708 | 127.1 | 10.1 to 115.8 to 162.9 |
| `attention_none_mlp_replace` | 0.072 | 0.482 | 0.565 | 127.1 | 72.4 to 456.8 to 489.0 |
| `attention_p64_linear_add` | 0.667 | 0.669 | 0.705 | 120.8 | 9.9 to 48.7 to 152.9 |
| `attention_p64_linear_replace` | 0.101 | 0.557 | 0.576 | 128.2 | 112.5 to 369.7 to 468.8 |
| `attention_p64_mlp_add` | 0.667 | 0.668 | 0.713 | 127.0 | 8.4 to 47.2 to 141.3 |
| `attention_p64_mlp_replace` | 0.080 | 0.501 | 0.581 | 138.7 | 78.5 to 412.9 to 449.1 |
| `attention_p32_linear_add` | 0.666 | 0.668 | 0.684 | 74.0 | 10.2 to 47.0 to 144.4 |
| `attention_p32_linear_replace` | 0.073 | 0.481 | 0.525 | 92.6 | 99.4 to 380.9 to 401.2 |
| `attention_p32_mlp_add` | 0.665 | 0.672 | 0.695 | 87.0 | 8.9 to 108.2 to 141.2 |
| `attention_p32_mlp_replace` | 0.047 | 0.492 | 0.541 | 91.6 | 65.3 to 518.4 to 546.4 |
| `attention_zq_none_linear_add` | 0.662 | 0.668 | 0.723 | 127.1 | 11.6 to 115.7 to 162.0 |
| `attention_zq_none_linear_replace` | 0.078 | 0.536 | 0.606 | 127.1 | 105.5 to 372.9 to 388.4 |
| `attention_zq_none_mlp_add` | 0.662 | 0.667 | 0.709 | 127.1 | 10.3 to 111.0 to 165.3 |
| `attention_zq_none_mlp_replace` | 0.074 | 0.522 | 0.596 | 127.1 | 73.4 to 479.7 to 497.5 |
| `attention_zq_p64_linear_add` | 0.668 | 0.666 | 0.705 | 120.9 | 10.1 to 23.4 to 154.2 |
| `attention_zq_p64_linear_replace` | 0.100 | 0.542 | 0.627 | 129.9 | 105.8 to 364.5 to 381.2 |
| `attention_zq_p64_mlp_add` | 0.668 | 0.672 | 0.690 | 127.3 | 8.8 to 78.8 to 152.9 |
| `attention_zq_p64_mlp_replace` | 0.065 | 0.516 | 0.610 | 131.2 | 76.3 to 429.1 to 459.3 |
| `attention_zq_p32_linear_add` | 0.668 | 0.668 | 0.694 | 77.0 | 10.4 to 17.9 to 134.3 |
| `attention_zq_p32_linear_replace` | 0.078 | 0.509 | 0.560 | 95.2 | 91.3 to 396.9 to 410.7 |
| `attention_zq_p32_mlp_add` | 0.665 | 0.669 | 0.689 | 89.4 | 9.0 to 79.9 to 153.4 |
| `attention_zq_p32_mlp_replace` | 0.056 | 0.499 | 0.548 | 96.4 | 65.2 to 536.8 to 565.3 |

- Train accuracy of the untrained ensemble is about 0.67 (the backbones were not trained on the joint split).
- Value norm is the L2 norm of a member's flattened projected map, 64 · s numbers.
- Injection norm is the L2 norm of the decoder output per example (16 × 32 × 32). Start is the mean of the first 20 steps, last the mean of the last 200.

## Findings

- No arm separates from averaging. All 24 additive arms are within 0.36 points of it on test, against a seed std of 0.04 to 0.58 points. Over the 72 additive seeds the mean change is -0.08 points (sd 0.30), and 24 of 72 are above averaging. The best arm, `average_none_mlp_add` (+0.32 ± 0.16), uses no learned mixing at all.
- The way members are mixed makes no difference with additive injection. Paired against the same projection and decoder, gate, attention and attention_zq are each within 0.6 points of average, and their mean differences are 0.02 to 0.07 points.
- The learned mixes are not mixing. Gate and attention become one-hot within about 10 training steps (attention within 3 to 5; entropy from 1.10 to below 0.1) and stay there. The joint mode share is at least 0.999 in 32 of the 36 gate and attention arms (the other 4, all gate replace arms, are 0.68 to 0.94), so each receiver reads the same single sender for every input. The gate picks a different member on every seed. So "gate" is copying one member chosen at random early on, and "attention" is each receiver copying one fixed member.
- Starting attention uniform does not help. attention_zq starts at exactly 1/3 each and is one-hot by step 3 to 5, about as fast as standard-init attention. Its test is within 0.4 points of attention for every additive pair (mean -0.07).
- Why it locks in so fast (likely cause): the score heads read the raw flattened map, 4,096 or 2,048 unnormalized numbers with RMS about 2. AdamW moves every weight by about the learning rate at the start, so one step can shift a score by several units, which already makes a 3-way softmax nearly one-hot. Then the softmax passes almost no gradient back (query/key gradient norms down to 1e-12 and below), and the choice freezes.
- Messages make members worse, and the ensemble gains where they make members disagree. Net corrected minus corrupted is negative in every arm. Across the 72 additive seeds, test gain correlates with lower agreement after the round (r = -0.55), with higher flip rate (+0.46) and with more net corruption (-0.53). This matches experiment 04: the small gains look like added diversity between members, not information passing.
- Replace fails much worse than in experiment 04: -25.9 to -36.0 points, vs -6.6 to -8.9 for 04's replace arms without patch attention. It underfits rather than overfits: train accuracy at the kept epoch is 0.40 to 0.65 per seed, below the untrained ensemble's 0.67 in every run, and at most 0.73 at the last epoch. Patience stopped every replace run between 23 and 51 epochs, so more epochs would not have helped. Likely cause: `MinimalDecoder` uses one positions map for all channels, so each output channel is a mix of the same 64 upsampled channel maps. 04's patch decoder mapped each cell's channels to its own 16 × 4 × 4 block jointly. Rebuilding the `layer1.2` input from the 8 × 8 read is already the known limit, and this decoder makes it harder.
- Within replace, average (-29.8) is above gate (-32.9) and attention (-31.6 to -31.9). This is the opposite of the plan's reason for keeping replace: the arm where every receiver gets the same message is the least bad. 64→32 (-30.0) is above no projection (-33.0), and linear (-30.1) is above MLP (-33.0).
- The parameter gap between attention and the others (about 600k vs 110k to 230k) did not show up in accuracy, because the attention heads stop learning after a few steps.

## Metric issues

- `self_weight` and `self_attn_rate` are 1/3 by construction for the gate arms (identical weights for every receiver). They only carry information for attention.
- `val_route_flip_rate` is missing for runs whose kept epoch is the first.
