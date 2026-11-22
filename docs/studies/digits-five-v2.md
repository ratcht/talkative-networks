# Study plan: experts from different digit datasets, second version

The first experiment of this study is reported in
[digits-five.md](digits-five.md). In it, no mixer in which the
experts hear each other was more accurate than the `self` mixer, and the
references were not fair to compare with: the pooled model was the size of
one expert and never saw the images the channel was trained on, and the gate
of the MoE had several times fewer trained weights than the channel. This
plan repeats the experiment with fair references and adds two conditions
under which the `self` mixer is expected to weaken: fewer images for the
channel, and a narrower taper.

## Setup

| item | value |
|---|---|
| expert datasets | MNIST, SVHN, USPS |
| target dataset | MNIST-M |
| expert architecture | `cnn3` |
| expert images per dataset (cap) | 3,600 |
| expert epochs | 10 |
| channel encoder and decoder | `linear_taper_gelu` |
| channel, gate and pooled phase 2 epochs | 30 |
| trial seeds | 0, 1, 2, 3, 4 |
| study tag | `multiple-datasets-v2` |
| pass tag | 1 |

## Images the channel trains on

| setting | large | small |
|---|---|---|
| target | 8,700 from MNIST-M | 1,200 from MNIST-M |
| source | 2,900 each from MNIST, SVHN, USPS | 400 each from MNIST, SVHN, USPS |

## Arms

| arms | axes | count |
|---|---|---:|
| experts alone | tested on MNIST-M, MNIST, SVHN, USPS | 4 |
| pooled, phase 1 only | `cnn3`, `cnn3_large` | 2 |
| pooled, phase 1 then phase 2 | `cnn3`, `cnn3_large` × 2 settings × 2 image counts | 8 |
| gate | 3 read points × 2 widths × 2 settings × 2 image counts | 24 |
| channel | 3 mixers × 3 tapers × 2 losses × 2 directions × 2 settings × 2 image counts | 144 |
| total | | 182 |

| axis | values |
|---|---|
| mixer | `rmsnorm_attention`, `self`, `mean` |
| taper `r` | 8, 4, 2 |
| loss | joint, individual |
| direction | forward, backward |
| gate read point | `stage1`, `stage2`, `stage3` |
| gate width | 8, 32 |

With 5 seeds that is 910 runs.

## Pooled model

| phase | data | epochs |
|---|---|---:|
| 1 | the experts' 10,800 images | 10 |
| 2 | the same images the channel trains on in that setting and image count | 30 |

Phase 1 is expert training (`--scheme seed_only`). Phase 2 is the testbed's
joint training with the model's own weights as the trained part, run by the
study's `pooled.py`, so it is evaluated after every epoch and read at its
best `joint_val` epochs, the same way as a channel run.
`cnn3_large` is `cnn3` at twice the width, about the size of the three
experts and their channel at `r = 8`.

## Code to write

| change | where |
|---|---|
| phase 2 of the pooled model, built from the testbed's functions | `studies/multiple_datasets/pooled.py` |
| arms updated to the tables above, `attention` removed | the study's runner |

Already in the code: `cnn3_large`, the weight counts, the cap of 3,600 and the
30 epochs.

## Not in this job

The classifier on all experts' final features, and PACS, which becomes its
own study.
