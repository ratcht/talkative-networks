# Does communication help an ensemble of experts?

Findings from the channel ladder study: 1,200 training runs on one image task.

## Summary

Three frozen image classifiers ("experts") were given a small trainable add-on
so they could exchange information before answering. Group accuracy rose from
88.2% to about 90.5%, but communication wasn't the reason: the same add-on does
at least as well when each expert hears only itself.

Letting experts choose whom to listen to never helped. Forward, it costs about
0.1 points, and a rerun on a second machine reproduced that. Stopping `attention`
from locking on to one sender didn't change accuracy. Hearing others makes the
experts better individually, but the group barely moves. A larger add-on helped
in one direction only, and the gain lasted only without communication. A larger
gate didn't help at all.

## Scoring

Every condition ran on the same 15 groups of experts for 30 epochs (passes
through the training images). A run's score is its median accuracy on 10,000
test images, over the 5 epochs around its best epoch on 6,000 separate validation
images. Differences are in percentage points, compared group by group (same
experts). ± is twice the standard error of those paired differences, roughly a
95% range; a range based on the t distribution, which is more exact for 15
groups, would be about 7% wider. (13/15) means ahead on 13 of 15 groups.

Some tables read fixed epochs instead, the same ones for every run, so that no
epoch is picked for any run. Accuracy findings also hold on the validation images
at fixed epochs unless marked weak or noted. Routing and message numbers come
from the validation images only. The gate isn't in those tables: it sends no
messages, and its weights weren't logged.

## Overview

| phase | question | runs |
|---|---|---|
| 1 | Where should the channel read and write? | 225 |
| 2a, 2b | Does a nonlinearity (GELU) help, forward and backward? | 180 |
| 2c | Does hearing everyone equally help? | 120 |
| 3 | Does a wider channel help? Does a wider gate? | 585 |
| 4 | Is forward lock-on caused by message size? | 90 |

Scores (%) at level 1:

| who each expert hears | forward, r = 8 | backward, r = 8 | backward, r = 32 |
|---|---|---|---|
| No channel | 88.22 | 88.22 | 88.22 |
| Gate, reading stage 3 | 89.83 | 89.83 | 89.88 |
| `self` | 90.44 | 90.66 | 90.86 |
| `mean` | 90.44 | 90.80 | 90.88 |
| `attention` | 90.40 | 90.60 | 90.76 |
| `attention`, rerun | 90.39 | not run | not run |
| `cosine_attention` | 90.40 | not run | not run |

The gate has no direction, so its score repeats across the two r = 8 columns. In
the r = 32 column it is the gate at r = 32.

## Phase 1: where to read and write

Level 1 with `attention` and `self` at six read and write points, and the gate at
three read points, all at r = 8.

| read, write | direction | `self` | `attention` | `attention` − `self` | all 30 epochs, test | all 30 epochs, validation |
|---|---|---|---|---|---|---|
| stage 1, stage 2 | forward | 90.52 | 90.48 | −0.04 ± 0.11 (7/15) | −0.10 ± 0.05 | −0.10 ± 0.06 |
| stage 1, stage 3 | forward | 90.31 | 90.22 | −0.09 ± 0.09 (7/15) | −0.08 ± 0.05 | −0.11 ± 0.04 |
| stage 2, stage 3 | forward | 90.44 | 90.40 | −0.04 ± 0.10 (4/15) | −0.09 ± 0.04 | −0.13 ± 0.05 |
| stage 2, stage 1 | backward | 90.72 | 90.62 | −0.09 ± 0.09 (2/15) | −0.09 ± 0.04 | −0.09 ± 0.04 |
| stage 3, stage 2 | backward | 90.63 | 90.56 | −0.07 ± 0.06 (4/15) | −0.03 ± 0.06 | −0.03 ± 0.04 |
| stage 3, stage 1 | backward | 90.66 | 90.60 | −0.06 ± 0.06 (6/15) | −0.05 ± 0.05 | −0.03 ± 0.05 |

`attention` trails `self` at all six. The later phases used each direction's
smallest loss by the score, stage 2 to stage 3 forward and stage 3 to stage 1
backward, though both were ties with the runner-up.

| gate reads | score |
|---|---|
| Stage 1 | 89.62 |
| Stage 2 | 89.72 |
| Stage 3 | 89.83 |

Stage 3 beats stage 2 by 0.11 ± 0.11 (12/15), so later phases used stage 3.

### 1. The channel beats both baselines, but not through communication

`self` and `attention` score about the same, and both sit above the gate.

| condition | score | spread between groups (standard deviation) |
|---|---|---|
| No channel | 88.22 | 2.07 |
| Gate, which weighs answers without changing them | 89.83 | 0.74 |
| `attention`, forward, level 1 | 90.40 | 0.30 |
| `self`, forward, level 1 | 90.44 | 0.26 |
| `attention`, backward, level 1 | 90.60 | 0.25 |
| `self`, backward, level 1 | 90.66 | 0.30 |

The spread shows how much the 15 groups differ from each other, mostly in how
lopsided their experts are. It isn't the uncertainty of a comparison: every
condition runs on the same groups, so conditions are compared group by group.
Their ranges overlap even when one condition is ahead on every group.

Lead over the gate at r = 8, group by group. Levels 2 and 3 come from phases 2a
and 2b, and `mean` from phase 2c:

| level | `attention`, forward | `self`, forward | `mean`, forward | `attention`, backward | `self`, backward | `mean`, backward |
|---|---|---|---|---|---|---|
| 1 | +0.57 ± 0.40 (13/15) | +0.61 ± 0.38 (12/15) | +0.60 ± 0.38 (12/15) | +0.76 ± 0.37 (14/15) | +0.82 ± 0.39 (15/15) | +0.97 ± 0.37 (15/15) |
| `2-encoder` | +0.53 ± 0.43 (11/15) | +0.62 ± 0.39 (14/15) | +0.57 ± 0.43 (13/15) | +0.76 ± 0.37 (14/15) | +0.71 ± 0.39 (14/15) | +0.81 ± 0.36 (14/15) |
| `2-decoder` | +0.53 ± 0.39 (12/15) | +0.65 ± 0.41 (12/15) | +0.74 ± 0.41 (12/15) | +0.78 ± 0.39 (13/15) | +0.77 ± 0.40 (14/15) | +0.89 ± 0.36 (14/15) |
| 3 | +0.54 ± 0.39 (11/15) | +0.64 ± 0.39 (13/15) | +0.60 ± 0.43 (12/15) | +0.72 ± 0.37 (13/15) | +0.74 ± 0.38 (14/15) | +0.83 ± 0.36 (14/15) |

Every channel condition is ahead of the gate on 11–15 of 15 groups, by 0.5–1.0
on average. The ± is wide because the lead is uneven between groups, not because
the result is unclear. For forward level 1 `attention`, the lead runs from −0.07
to +2.98 across groups, with a median of +0.37. Backward it runs from −0.16 to
+2.90, with a median of +0.66.

`self` leads by as much as `attention`, so the gain comes from having a trained
add-on, not from experts hearing each other. The gate is also smaller: 13,900
weights, against 18,500–31,100 per expert for the channel at r = 8. A wider gate
doesn't close the gap (finding 13).

### 2. Where the channel writes sets the score; where it reads sets the cost of `attention` (weak)

Without communication, the score depends on how early the channel writes: an
earlier write has more of the expert after it. `self`, level 1, by write point:

| `self` writes after | score |
|---|---|
| Stage 1 | 90.69 |
| Stage 2 | 90.58 |
| Stage 3 | 90.38 |

Writing after stage 1 beats stage 3 by 0.31 ± 0.06 (15/15). Communication
behaves differently. `attention`'s loss to `self`, over all 30 epochs, depends on
where the channel reads, not the direction:

| `attention` − `self` | test | validation |
|---|---|---|
| Reading stage 1 or 2 (four placements) | −0.09 ± 0.03 | −0.11 ± 0.03 |
| Reading stage 3 (two placements) | −0.04 ± 0.05 | −0.03 ± 0.03 |
| Difference | +0.05 ± 0.05 (8/15) | +0.08 ± 0.04 (14/15) |
| Reading stage 2, backward minus forward | +0.00 ± 0.05 | +0.04 ± 0.06 |

The frozen experts are least alike at stage 3, and a simple classifier gains most
there from seeing all three experts' internal states instead of one:

| stage | similarity between experts (1 = identical) | gain from seeing all three experts |
|---|---|---|
| 1 | 0.98 | +0.57 ± 0.14 |
| 2 | 0.98 | +0.98 ± 0.15 |
| 3 | 0.94 | +1.13 ± 0.14 |

Reading stage 3 only removes the loss; it never turns into a gain. This is
exploratory: the stage 3 contrast was chosen after seeing the table, and by the
score it is +0.00 ± 0.07. Stage 3 can only be read backward, so read point and
direction can't be fully separated. It also means later phases' forward against
backward comparisons differ in read point too (stage 2 against stage 3).

## Phases 2a and 2b: does a nonlinearity help?

Levels 2 and 3 with `attention` and `self`, forward (2a) and backward (2b), at
r = 8. Level 1 comes from phase 1.

### 3. Choosing whom to hear doesn't help

`attention` − `self` at r = 8, by the score:

| level | forward | backward |
|---|---|---|
| 1 | −0.04 ± 0.10 (4/15) | −0.06 ± 0.06 (6/15) |
| `2-encoder` | −0.09 ± 0.09 (3/15) | +0.05 ± 0.09 (11/15) |
| `2-decoder` | −0.11 ± 0.10 (3/15) | +0.01 ± 0.13 (10/15) |
| 3 | −0.10 ± 0.10 (3/15) | −0.03 ± 0.09 (6/15) |

No GELU level beats level 1 with `attention` at r = 8. The forward loss is
clearer over all 30 epochs, where it holds at every width (finding 14), and it
reappears in fresh runs (finding 19).

### 4. Routing is sharpest with linear senders, and softer backward

A GELU when sending softens routing. Backward is softer than forward at every
level. From softest to sharpest, the order is `2-encoder`, 3, 1, `2-decoder` in
both directions, though the gaps in common wiring are much smaller backward.
Epochs 26–30, r = 8:

| level | spread of attention, forward | backward | weight on top sender, forward | backward | images on the most common wiring, forward | backward |
|---|---|---|---|---|---|---|
| `2-encoder` | 0.63 | 0.70 | 0.69 | 0.67 | 47% | 29% |
| 3 | 0.49 | 0.55 | 0.77 | 0.75 | 56% | 33% |
| 1 | 0.01 | 0.35 | 1.00 | 0.85 | 85% | 33% |
| `2-decoder` | 0.00 | 0.22 | 1.00 | 0.91 | 95% | 40% |

Spread runs from 0 (one sender) to 1.10 (all three equally). Phase 4 traced the
sharpness to message size (finding 18).

### 5. `attention` doesn't favour senders who are right

However concentrated routing gets, experts hear themselves about as often as
chance (33%), though single runs vary widely (7–63% backward). The sender heard
is right about as often at every level. Epochs 26–30 for hearing itself and
21–30 for the sender, r = 8:

| level | hears itself, forward | backward | sender heard is right, forward | backward |
|---|---|---|---|---|
| 1 | 35% | 30% | 85.1% | 85.9% |
| `2-encoder` | 28% | 28% | 84.9% | 85.8% |
| `2-decoder` | 26% | 26% | 85.6% | 86.4% |
| 3 | 25% | 32% | 85.3% | 85.4% |

That sits between hearing a random partner and always hearing the more accurate
one, never above that, and far below choosing by class or by image:

| if each receiver heard | sender heard is right |
|---|---|
| A random partner | 84.7% |
| Always its more accurate partner | 86.5% |
| The best partner for each class | 91.5% |
| A right partner whenever one exists | 92.8% |

### 6. Linear senders are louder, and only forward message sizes keep moving

What receivers hear is stable backward but shifts a lot forward. Linear senders
are louder than GELU senders in both directions, and `self` shows the same gap,
so this isn't caused by routing. The signal written into experts grows everywhere.
r = 8, `attention`; each cell is the value after epoch 1, then after epoch 30:

| level | size heard, forward | backward | written signal, forward | backward |
|---|---|---|---|---|
| 1 | 25.9 to 17.0 | 10.2 to 12.4 | 3.3 to 12.4 | 4.2 to 35.7 |
| `2-decoder` | 33.1 to 32.1 | 12.9 to 13.4 | 3.3 to 8.5 | 4.1 to 30.6 |
| `2-encoder` | 17.8 to 6.1 | 7.3 to 7.8 | 3.1 to 9.4 | 4.1 to 32.4 |
| 3 | 24.4 to 8.2 | 10.5 to 9.0 | 3.4 to 9.0 | 3.9 to 29.6 |

- **Size heard** is `msg_norm/aggregated`: the length of the message a receiver
  gets after mixing, averaged over images and receivers.
- **Written signal** is `msg_norm/injection`: the length of what the decoder adds
  to the receiving expert's features. It is 0 before training.

Both are measured on the validation images and averaged over the 15 groups.
Forward keys end in `/ex0` and backward keys in `/ex0/round0`. Epochs 1 and 30
are MLflow steps 1 and 30.

GELU senders also give queries a stronger training signal, but this looks like a
result of softer routing, not its cause. Forward, linear and GELU senders start
with similar query gradients, but their attention spreads differently by epoch
5, and their gradients separate only later. Backward, where all levels are soft,
the gap is small:

| level | query gradient, forward, epoch 5 | forward, epochs 26–30 | backward, epochs 26–30 | spread of attention, forward, epoch 5 |
|---|---|---|---|---|
| 1 | 0.019 | 0.013 | 0.065 | 0.015 |
| `2-decoder` | 0.015 | 0.005 | 0.052 | 0.005 |
| `2-encoder` | 0.019 | 0.050 | 0.072 | 0.316 |
| 3 | 0.012 | 0.037 | 0.065 | 0.207 |

- **Query gradient** is `grad_norm/encoder.query/ex0`, in both directions: the
  size of the gradient reaching the query head's weights, averaged over the
  training steps of the epoch. Epochs 26–30 is the mean of those five epochs.
- **Spread of attention** is `attn_entropy/ex0`: how evenly receivers split their
  attention, from 0 (all on one sender) to 1.10 (all three equally), on the
  validation images.

### 7. Much of the gain may not involve messages (weak)

Most of the gain arrives in the first epoch, when most of the channel's signal is
the same for every image. It may mostly be correcting each expert's class tilt.
This is untested. r = 8, all four levels together; `mean` comes from phase 2c:

| condition | gain over no channel, epoch 1 | final gain | share in epoch 1 | written signal that's the same for every image, epoch 1 |
|---|---|---|---|---|
| Forward, `self` | 1.82 | 2.24 | 81% | 80% |
| Forward, `attention` | 1.79 | 2.16 | 83% | 80% |
| Forward, `mean` | 1.81 | 2.24 | 81% | 80% |
| Backward, `self` | 1.83 | 2.38 | 77% | 80% |
| Backward, `attention` | 1.80 | 2.37 | 76% | 80% |
| Backward, `mean` | 1.78 | 2.49 | 71% | 80% |
| Gate | 1.28 | 1.62 | 79% | not applicable |

## Phase 2c: does hearing everyone equally help?

`mean` at all four levels, at both chosen read and write points, r = 8. It is
compared with `self` and `attention` from phases 1, 2a and 2b.

### 8. Hearing others changes the experts, not the group

Experts get better and more alike the more of the others they hear, but the group
doesn't follow. Level 1, r = 8:

| condition | mean expert | what averaging adds | group | agreement between experts | at least one expert right |
|---|---|---|---|---|---|
| No channel | 84.32 | 3.90 | 88.22 | 81.77 | 95.20 |
| Gate | 84.32 | 5.52 | 89.83 | 81.77 | 95.20 |
| Forward, `self` | 88.20 | 2.24 | 90.44 | 88.04 | 95.41 |
| Forward, `attention` | 88.57 | 1.83 | 90.40 | 89.36 | 95.07 |
| Forward, `mean` | 88.91 | 1.53 | 90.44 | 90.34 | 94.78 |
| Backward, `self` | 88.53 | 2.13 | 90.66 | 88.62 | 95.37 |
| Backward, `attention` | 88.97 | 1.63 | 90.60 | 90.09 | 94.93 |
| Backward, `mean` | 89.26 | 1.55 | 90.80 | 90.66 | 94.88 |

- **Mean expert** is `test/expert_accuracy`: each expert's own accuracy, answering
  alone after the channel has changed it, averaged over the three experts.
- **Group** is `test/accuracy`: the accuracy of the experts' averaged answer
  (average their class probabilities, take the top class). It is the score.
- **What averaging adds** is group − mean expert. It isn't logged.
- **Agreement between experts** is `test/joint_answer_shift/agreement_post`: on
  each image, the share of the three pairs of experts that give the same answer,
  averaged over images.
- **At least one expert right** is the sum of three `test/answer_grid` cells:
  `all_right`, `some_right_avg_right` and `some_right_avg_wrong`. It is the share
  of images where some expert's own answer is right, the ceiling for any way of
  picking among the experts' answers.

Each column is read like the score (median over the 5 epochs around the best
validation epoch, on the test images) and averaged over the 15 groups. No
channel is step 0, before the channel has injected anything: the frozen experts,
averaged.

The gate leaves the experts as they are, so all its gain comes from a better
weighted average. Differences from `self`, all four levels together:

| measure | `attention`, forward | `attention`, backward | `mean`, forward | `mean`, backward |
|---|---|---|---|---|
| Mean expert | +0.29 ± 0.08 (14/15) | +0.46 ± 0.08 (15/15) | +0.65 ± 0.09 (15/15) | +0.75 ± 0.09 (15/15) |
| Agreement | +1.12 ± 0.28 (15/15) | +1.49 ± 0.15 (15/15) | +2.03 ± 0.24 (15/15) | +2.29 ± 0.27 (15/15) |
| What averaging adds | −0.38 ± 0.10 (1/15) | −0.47 ± 0.06 (0/15) | −0.65 ± 0.08 (0/15) | −0.63 ± 0.08 (0/15) |
| Right expert outvoted | −0.28 ± 0.11 (1/15) | −0.41 ± 0.08 (0/15) | −0.57 ± 0.11 (0/15) | −0.69 ± 0.09 (0/15) |
| No expert right | +0.34 ± 0.10 (15/15) | +0.39 ± 0.07 (15/15) | +0.54 ± 0.09 (15/15) | +0.57 ± 0.11 (15/15) |
| Group | −0.09 ± 0.06 (3/15) | −0.01 ± 0.06 (7/15) | +0.00 ± 0.04 (6/15) | +0.11 ± 0.06 (13/15) |

The group's errors are the "outvoted" and "no expert right" images added together.
Fewer right experts get outvoted, but about as many images end up with no expert
right, so the two nearly cancel. The training goal rewards only the group answer,
which may be why better individuals add so little.

### 9. A wide gap to training isn't overfitting by itself, but hearing others overfits

Training accuracy keeps climbing in every condition. With `self`, validation just
stays flat: accuracy holds and loss rises only a little. `mean` has a cup instead:
its loss falls further, then climbs well past `self`, and its accuracy turns down.
Backward, r = 8, all four levels together:

| measure | epochs 1–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 |
|---|---|---|---|---|---|---|
| Training accuracy, `self` | 91.3 | 92.7 | 93.6 | 94.3 | 94.9 | 95.4 |
| Training accuracy, `attention` | 91.3 | 92.7 | 93.7 | 94.5 | 95.1 | 95.6 |
| Training accuracy, `mean` | 91.4 | 92.9 | 94.0 | 94.9 | 95.5 | 96.1 |
| Training accuracy, gate | 90.5 | 91.2 | 91.7 | 92.2 | 92.6 | 92.9 |
| Validation accuracy, `self` | 90.72 | 90.99 | 91.06 | 91.07 | 91.07 | 91.10 |
| Validation accuracy, `attention` | 90.75 | 91.00 | 91.06 | 91.07 | 91.03 | 91.04 |
| Validation accuracy, `mean` | 90.78 | 91.06 | 91.13 | 91.12 | 91.08 | 91.04 |
| Validation accuracy, gate | 90.03 | 90.16 | 90.17 | 90.22 | 90.19 | 90.19 |
| Validation loss, `self` | 0.260 | 0.253 | 0.253 | 0.255 | 0.259 | 0.262 |
| Validation loss, `attention` | 0.259 | 0.253 | 0.253 | 0.257 | 0.263 | 0.269 |
| Validation loss, `mean` | 0.258 | 0.250 | 0.253 | 0.259 | 0.269 | 0.280 |
| Validation loss, gate | 0.272 | 0.268 | 0.268 | 0.269 | 0.272 | 0.275 |

Width makes this much worse for conditions that hear others (finding 16).

### 10. What's learned on training images doesn't carry over

On training images, channels teach experts to answer most of what none of them
knew. On validation images none of that appears. The gate's experts don't change,
so its only way to fit is to stop right experts being outvoted. It goes furthest
there on training images but stays highest on validation images. Backward, level
1, r = 8, epoch 1 to epoch 30, percent of images:

| condition | no expert right, training | no expert right, validation | right expert outvoted, training | right expert outvoted, validation |
|---|---|---|---|---|
| `self` | 4.29 to 0.35 | 4.57 to 4.30 | 5.26 to 3.41 | 4.85 to 4.63 |
| `attention` | 4.32 to 0.48 | 4.68 to 4.87 | 5.29 to 3.23 | 4.73 to 4.04 |
| `mean` | 4.34 to 0.48 | 4.62 to 5.24 | 5.20 to 2.89 | 4.75 to 3.68 |
| Gate | 4.35 to 4.35 | 4.52 to 4.52 | 5.63 to 2.65 | 5.55 to 5.29 |

### 11. `mean`'s backward lead over `self` (weak)

The score gives +0.11 ± 0.06 (13/15), the only positive communication result.
Forward it is 0.00 ± 0.04. Backward, fixed epochs show a smaller lead on
validation images, and it fades:

| epochs | test | validation |
|---|---|---|
| 1–5 | +0.05 ± 0.02 (14/15) | +0.06 ± 0.03 (12/15) |
| 6–10 | +0.09 ± 0.05 (11/15) | +0.07 ± 0.06 (13/15) |
| 11–15 | +0.12 ± 0.04 (13/15) | +0.07 ± 0.03 (12/15) |
| 16–20 | +0.09 ± 0.05 (12/15) | +0.05 ± 0.06 (10/15) |
| 21–25 | +0.09 ± 0.08 (12/15) | +0.01 ± 0.05 (7/15) |
| 26–30 | +0.01 ± 0.09 (8/15) | −0.06 ± 0.06 (4/15) |

It may also be capacity, not communication: a `mean` receiver hears up to three
times as many numbers, and `self` at three times the width reaches the same score
(90.79 against 90.80). It may also be the read point, since `mean` was only run
reading stage 2 forward and stage 3 backward (finding 2). The true size is
probably 5–10 test images in 10,000.

### 12. `attention` trailing `mean` (weak)

`attention` − `mean`, levels 1 and 3 together, all 30 epochs. The wider rows come
from phase 3. It is small, and not seen at every width on validation images.

| width | forward, test | forward, validation | backward, test | backward, validation |
|---|---|---|---|---|
| r = 8 | −0.08 ± 0.04 | −0.06 ± 0.05 | −0.09 ± 0.04 | −0.04 ± 0.03 |
| r = 16 | −0.07 ± 0.04 | −0.06 ± 0.05 | −0.06 ± 0.05 | −0.01 ± 0.03 |
| r = 24 | −0.08 ± 0.06 | −0.09 ± 0.04 | −0.08 ± 0.05 | −0.06 ± 0.04 |
| r = 32 | −0.09 ± 0.04 | −0.06 ± 0.06 | −0.06 ± 0.05 | −0.08 ± 0.06 |

## Phase 3: does a wider channel help?

Levels 1 and 3 with `attention`, `self` and `mean`, at both chosen read and write
points, at r = 16, 24 and 32. r = 8 comes from phases 1 and 2. The gate was added
at each width on 2026-11-02, after the rest of the study had run.

### 13. A wider gate doesn't close the gap

| r | gate weights | channel weights, forward | backward | gate score | channel score, forward | backward | gate − gate at r = 8 | channel − gate at the same r, forward | backward |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 13,900 | 55,600 | 93,300 | 89.83 | 90.43 | 90.64 | | +0.59 ± 0.39 (12/15) | +0.81 ± 0.37 (14/15) |
| 16 | 27,800 | 111,000 | 186,200 | 89.89 | 90.40 | 90.72 | +0.05 ± 0.07 (10/15) | +0.51 ± 0.36 (13/15) | +0.83 ± 0.34 (15/15) |
| 24 | 41,800 | 166,300 | 279,200 | 89.86 | 90.44 | 90.72 | +0.02 ± 0.10 (7/15) | +0.58 ± 0.40 (12/15) | +0.87 ± 0.38 (15/15) |
| 32 | 55,700 | 221,700 | 372,200 | 89.88 | 90.38 | 90.78 | +0.04 ± 0.08 (11/15) | +0.50 ± 0.38 (11/15) | +0.90 ± 0.35 (15/15) |

The channel columns pool levels 1 and 3 and all three mixers. Channel weights
count all three experts' encoders and decoders. They are the same at both levels
and for every mixer, because a GELU adds no weights and the mixers have none.
With `self` and `mean`, the query and key heads (864 weights at r = 8, 3,168 at
r = 32) get no training signal. Backward has more weights because it writes
to stage 1, the largest write point. Weights don't explain the lead: the gate at
r = 32 has as many as the forward channel at r = 8 (55,700 against 55,600), and
that channel still leads it by +0.55 ± 0.38 (12/15). Over all 30 epochs
the wider gates gain only +0.03 to +0.04 on test and +0.03 to +0.07 on validation.
The extra width goes into fitting training images instead: training accuracy
rises from 92.9 at r = 8 to 93.9 at r = 32 in epochs 26–30, while validation
accuracy stays near 90.2. Backward, every condition stays ahead of the same-width
gate on all 15 groups at r ≥ 16; `self` at level 1 leads by +0.98 ± 0.34 at
r = 32.

### 14. `attention` doesn't beat `self` at any width

Over all 30 epochs, `attention` − `self`, levels 1 and 3 together:

| width | forward, test | forward, validation | backward, test | backward, validation |
|---|---|---|---|---|
| r = 8 | −0.11 ± 0.04 | −0.11 ± 0.03 | −0.01 ± 0.05 | −0.02 ± 0.04 |
| r = 16 | −0.11 ± 0.05 | −0.14 ± 0.05 | +0.01 ± 0.04 | +0.00 ± 0.05 |
| r = 24 | −0.14 ± 0.04 | −0.15 ± 0.03 | −0.04 ± 0.03 | −0.03 ± 0.04 |
| r = 32 | −0.13 ± 0.04 | −0.13 ± 0.04 | −0.02 ± 0.05 | −0.08 ± 0.05 |

### 15. Width helps backward only, and lasts only without communication

Backward scores by width:

| condition | r = 8 | r = 16 | r = 24 | r = 32 |
|---|---|---|---|---|
| Level 1, `self` | 90.66 | 90.75 | 90.79 | 90.86 |
| Level 1, `mean` | 90.80 | 90.88 | 90.81 | 90.88 |
| Level 1, `attention` | 90.60 | 90.74 | 90.76 | 90.76 |
| Level 3, `self` | 90.58 | 90.59 | 90.66 | 90.72 |
| Level 3, `mean` | 90.67 | 90.70 | 90.69 | 90.72 |
| Level 3, `attention` | 90.55 | 90.66 | 90.64 | 90.71 |
| Gate | 89.83 | 89.89 | 89.86 | 89.88 |

Those scores read each run near its best epoch. Late in training only `self`
keeps its gain. Backward gain over r = 8 at epochs 21–30, levels 1 and 3 together,
test / validation:

| who each expert hears | r = 16 | r = 24 | r = 32 |
|---|---|---|---|
| `self` | +0.05 / +0.07 | +0.11 / +0.13 | +0.16 / +0.15 |
| `attention` | +0.03 / +0.07 | +0.02 / +0.05 | +0.03 / −0.02 |
| `mean` | −0.01 / +0.03 | −0.00 / +0.09 | +0.00 / +0.05 |
| Gate | +0.01 / +0.01 | +0.02 / +0.07 | +0.03 / +0.02 |

Forward, width changes nothing, and the channel doesn't use the extra room. Level
1, `self`, epochs 21–30:

| measure | r = 8 | r = 16 | r = 24 | r = 32 |
|---|---|---|---|---|
| Message directions in use, forward | 4.2 | 5.3 | 5.4 | 5.4 |
| Patterns written, forward | 2.1 | 2.2 | 2.3 | 2.3 |
| Message directions in use, backward | 5.1 | 7.7 | 9.1 | 9.9 |
| Patterns written, backward | 4.1 | 5.6 | 6.3 | 6.8 |

At r = 32 the backward channel has 124,000 weights per expert, against an expert's
98,000.

### 16. With more width, hearing others overfits

Backward, rise in validation loss, then change in validation accuracy, from each
condition's best epochs to epochs 26–30:

| condition | r = 8 | r = 16 | r = 24 | r = 32 |
|---|---|---|---|---|
| Level 1, `self` | +0.014, −0.01 | +0.023, 0.00 | +0.025, 0.00 | +0.026, 0.00 |
| Level 1, `attention` | +0.019, −0.01 | +0.041, −0.05 | +0.048, −0.12 | +0.056, −0.13 |
| Level 1, `mean` | +0.031, −0.08 | +0.056, −0.07 | +0.063, −0.05 | +0.070, −0.12 |
| Level 3, `self` | +0.007, 0.00 | +0.016, 0.00 | +0.018, −0.06 | +0.022, 0.00 |
| Level 3, `attention` | +0.014, −0.03 | +0.037, −0.12 | +0.058, −0.25 | +0.077, −0.25 |
| Level 3, `mean` | +0.029, −0.11 | +0.070, −0.29 | +0.089, −0.30 | +0.098, −0.29 |
| Gate | +0.007, −0.03 | +0.012, −0.10 | +0.015, −0.03 | +0.015, −0.06 |

Scores are read before most of this happens.

### 17. Nonlinearity doesn't help at any width

Level 3 − level 1, r ≥ 16, all 30 epochs:

| who each expert hears | forward, test | forward, validation | backward, test | backward, validation |
|---|---|---|---|---|
| `attention` | +0.09 ± 0.03 (15/15) | −0.00 ± 0.04 (7/15) | −0.06 ± 0.03 (3/15) | −0.10 ± 0.03 (1/15) |
| `self` | +0.06 ± 0.03 (13/15) | +0.00 ± 0.04 (7/15) | −0.08 ± 0.02 (2/15) | −0.05 ± 0.03 (3/15) |
| `mean` | −0.01 ± 0.03 (7/15) | −0.05 ± 0.04 (4/15) | −0.14 ± 0.04 (0/15) | −0.16 ± 0.05 (1/15) |
| All three | +0.05 ± 0.02 (14/15) | −0.02 ± 0.03 (5/15) | −0.09 ± 0.02 (0/15) | −0.10 ± 0.03 (1/15) |

Backward it costs about 0.1, including for `self`. Forward, test and validation
disagree, so there's no finding. Level 3 never clearly beat level 1, so level 2
wasn't rerun at larger widths.

## Phase 4: is forward lock-on caused by message size?

Forward, r = 8: levels 1 and `2-encoder` with `cosine_attention`, which fixes the
length of queries and keys, and all four levels with `attention` again, on a
second machine.

### 18. Forward lock-on is a side effect of message size, and removing it changes nothing

Longer queries and keys sharpen `attention`. Phase 4 measured them for the first
time, then used `cosine_attention` to fix their length. At each run's score
window:

| condition | query and key size | spread of attention | weight on top sender | images on the most common wiring | score |
|---|---|---|---|---|---|
| `attention`, `2-decoder` | 48–50 | 0.002 | 1.00 | 89% | 90.41 |
| `attention`, level 1 | 31 | 0.006 | 1.00 | 88% | 90.39 |
| `attention`, level 3 | 7.5 | 0.39 | 0.83 | 72% | 90.29 |
| `attention`, `2-encoder` | 6 | 0.51 | 0.75 | 63% | 90.29 |
| `cosine_attention`, `2-encoder` | length has no effect | 0.88 | 0.60 | 35% | 90.33 |
| `cosine_attention`, level 1 | length has no effect | 0.82 | 0.63 | 21% | 90.40 |

So GELU senders looked softer because their queries and keys are smaller. Without
size, training barely moves routing from where it starts. Level 1:

| measure | `attention`, untrained to trained | `cosine_attention`, untrained to trained |
|---|---|---|
| Spread of attention | 1.08 to 0.006 | 0.88 to 0.82 |
| Weight on the top sender | 0.39 to 0.997 | 0.59 to 0.63 |
| Images on the most common wiring | 28% to 88% | 26% to 21% |
| Sender heard is right | 84.5% to 84.1% | 84.3% to 85.6% |

The loss to `self` remains, even though queries and keys now get about ten times
the training signal. Softer routing isn't better routing.

| level 1, r = 8 | score | all 30 epochs, test | all 30 epochs, validation |
|---|---|---|---|
| `cosine_attention` − `attention` | +0.01 ± 0.10 (5/15) | +0.05 ± 0.05 | +0.00 ± 0.08 |
| `cosine_attention` − `self` | −0.04 ± 0.10 (6/15) | −0.06 ± 0.04 | −0.11 ± 0.03 |
| `cosine_attention` − gate | +0.57 ± 0.39 (13/15) | +0.51 ± 0.39 | +0.56 ± 0.34 |
| `attention`, rerun − gate | +0.56 ± 0.40 (12/15) | +0.46 ± 0.37 | +0.56 ± 0.33 |

### 19. Fresh runs repeat the forward loss, and show how much a rerun moves a score

The four `attention` arms were rerun on a second machine, with the same experts
and settings. `attention` − `self`, all 30 epochs:

| level | first run, test | rerun, test | first run, validation | rerun, validation |
|---|---|---|---|---|
| 1 | −0.09 ± 0.04 | −0.11 ± 0.04 | −0.13 ± 0.05 | −0.11 ± 0.06 |
| `2-encoder` | −0.10 ± 0.05 | −0.14 ± 0.04 | −0.09 ± 0.05 | −0.11 ± 0.04 |
| `2-decoder` | −0.13 ± 0.06 | −0.14 ± 0.10 | −0.13 ± 0.04 | −0.16 ± 0.08 |
| 3 | −0.12 ± 0.05 | −0.12 ± 0.05 | −0.09 ± 0.06 | −0.10 ± 0.06 |

The rerun itself moved each arm's score:

| level, `attention`, r = 8 | first run | rerun | change |
|---|---|---|---|
| 1 | 90.40 | 90.39 | −0.01 ± 0.09 (5/15) |
| `2-encoder` | 90.36 | 90.29 | −0.07 ± 0.12 (7/15) |
| `2-decoder` | 90.37 | 90.41 | +0.04 ± 0.08 (9/15) |
| 3 | 90.38 | 90.29 | −0.08 ± 0.09 (5/15) |

### 20. Query and key growth comes from the layer they share with the message (weak)

An expert's query and key stay equal in size, even when sizes differ many times
between experts. In the first epoch, messages, queries and keys all grow ×8–11.
That points to the shared reading layer, not the query and key heads, whose logged
step sizes allow at most ×1.3 in that epoch. After epoch 1, growth belongs to
senders nobody hears. Level 1, `attention`, median key size:

| sender | epoch 1 | epoch 5 | epoch 30 |
|---|---|---|---|
| Never heard | 17.6 | 30.8 | 70.3 |
| Partly heard | 16.6 | 21.8 | 23.0 |
| Well heard | 14.6 | 15.0 | 13.9 |

The least-heard sender has the largest key on 14 of 15 groups with `attention` and
with `cosine_attention`. Under `cosine_attention` size can't affect who is heard,
so being unheard seems to come first. This rests on logs only, with no direct test.

## What this means

On this task, the channel works as a trained add-on but not as a way to
communicate. Experts do change when they hear each other, yet the group answer
doesn't improve, and `attention` doesn't pick the experts who know the answer,
whether its routing is locked or soft. All conditions stop 4–5 points below the
95.20% of test images that at least one expert gets right, so the experts know
more than any of these methods extracts.

## Limits

- **Small effects.** Scores vary by about a point between groups, and 0.1 is 10
  test images. ± covers differences between groups, not which images happen to
  be in the test set: switching to validation images moved one comparison by
  0.07.
- **Rerun noise.** A rerun on a second machine moved scores by −0.08 to +0.04
  (finding 19). So treat differences under about 0.1 as unresolved unless they
  also hold on validation images.
- **Scoring** flatters conditions that peak early, as `attention` and `mean` do.
- **Scope.** Only channel − `self`, channel − gate and phase 4's checks were
  planned; the rest is exploratory. One task, one small network, one learning
  rate, read and write points picked from six near-ties, and a training goal
  that rewards only the group answer. Phase 4 was forward only, at r = 8. The
  gate at r = 16–32 was added to phase 3 afterwards, on 2026-11-02, from
  uncommitted code whose training code for the gate is unchanged.

## Open questions

- Is the first-epoch gain a fixed correction? Test with a learned constant
  signal in place of the channel.
- Would training each expert on its own answer improve `attention`'s choices?
- Do `attention`'s choices per image matter? On an exact rerun of a trained run
  where routing is soft (backward, or `cosine_attention`), give each image
  another image's routing weights at test time. If accuracy holds, the choices
  carry nothing, however a sender might help.
- Is `mean`'s backward lead about the read point? Test `mean` reading stage 2 and
  writing after stage 1.
- Why does the shared reading layer jump in the first epoch, and why do unheard
  senders keep growing?

## Source

MLflow experiments `channel-ladder-phase1`, `phase2a`, `phase2b`, `phase2c`,
`phase3-r16`, `r24`, `r32` and `phase4`. Phases 1 to 3 ran on a laptop and phase
4 on a lab machine. The wider gates are the arms `moe-stage3-r16`, `-r24` and
`-r32` in phase 3's experiments, also run on the laptop; their code is saved in
`cache/channel_ladder_phase3_gate/`. The similarity and classifier numbers in
finding 2 come from the cached frozen experts, not from logged runs.
