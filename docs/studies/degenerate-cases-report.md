# What happens when the experts aren't a normal group?

Findings from the degenerate cases study: 585 training runs on one image task,
read against the channel ladder's level 1.

## Summary

The channel ladder kept one kind of expert group and changed the channel. This
study kept the channel (the ladder's level 1, r = 8) and changed the group, in
four ways:
- three copies of one expert;
- two trained experts plus one untrained;
- three experts trained to disagree;
- three untrained experts.

The channel can carry what experts know. An untrained expert that listens to two
trained ones gains 6 points forward and 19 backward. The group gains far less,
0.2 to 0.4, and forward even that fades by 60 epochs. That's because what the
untrained expert learns, the others already know.

Otherwise, hearing others helps the group only a little:
- **Experts trained to disagree:** about 0.3 at the strongest disagreement, and
  only backward.
- **Copies:** it never helps. But the lead it was meant to explain, `mean` over
  `self` with different experts, doesn't hold up on validation images.

`attention` never clearly beats `mean`. Its only clear gain for the group over
each expert's own add-on (`self`) comes when no expert knows anything: about 1
point after 150 epochs with three untrained experts. There `mean` is too uneven
to compare. With one untrained member, `attention` lifts that expert as much as
`mean` does, and no more.

A gate, run only with experts trained to disagree, is beaten by every channel at
every level of disagreement.

## Scoring

Scoring follows the channel ladder report. A run's score is its median accuracy
on 10,000 test images over the 5 epochs around its best epoch on 6,000 separate
validation images. There are two additions:
- The 60- and 150-epoch cases are also scored on their first 30 epochs, by the
  same rule.
- The untrained expert in "two trained + one untrained" is scored on its own
  accuracy, around its own best validation epoch.

Differences are in percentage points, compared group by group. Each case has 15
groups, one per trial seed. Clones and the two-trained groups are built from the
ladder's own 15 groups.
- **±** is twice the standard error of the paired differences, as in the ladder
  report. The study's design doc gives t-based ranges, about 7% wider.
- **(13/15)** means ahead on 13 of 15 groups.

Every accuracy finding was also checked at fixed epochs, the same ones for every
run, on test and on the validation images:
- the 30-epoch cases: all 30 epochs;
- the 60-epoch case: epochs 21–30 and 51–60;
- the 150-epoch case: epochs 21–30 and 141–150.

Findings hold on validation at fixed epochs unless marked weak or noted. Routing
is compared with each run's untrained channel at step 0. Findings that open with
"Planned" answer the four questions fixed before the runs; the rest is
exploratory.

## Overview

| case | the group | epochs | runs | planned question |
|---|---|---|---|---|
| Clones | three copies of the first expert of the ladder's group | 30 | 90 | Does the ladder's backward `mean` lead need experts that know different things? |
| Two trained + one untrained | the ladder's group with its third expert untrained | 60 | 90 | Can the channel carry information at all? |
| Negative correlation | three experts trained to disagree, at strengths λ = 0, 2.5 and 4 | 30 | 315 | Does hearing others matter more when experts disagree? |
| Untrained | three experts with random weights | 150 | 90 | Does listening help when nobody knows anything? |

Every case runs `self`, `mean` and `attention`, forward and backward, at level 1
with r = 8. That channel has 55,600 weights forward and 93,300 backward, for all
three experts. Negative correlation also runs the gate (13,900 weights) at each
λ.

Scores (%):

| group | before training | forward `self` | `mean` | `attention` | backward `self` | `mean` | `attention` | gate |
|---|---|---|---|---|---|---|---|---|
| The ladder's group (reference) | 88.22 | 90.44 | 90.44 | 90.40 | 90.66 | 90.80 | 90.60 | 89.83 |
| Clones | 82.65 | 89.39 | 89.42 | 89.37 | 89.57 | 89.50 | 89.49 | not run |
| Two trained + one untrained, first 30 epochs | 87.04 | 89.60 | 89.98 | 89.88 | 90.00 | 90.14 | 89.86 | not run |
| Two trained + one untrained, 60 epochs | 87.04 | 89.61 | 90.02 | 89.87 | 90.01 | 90.21 | 89.94 | not run |
| Negative correlation, λ = 0 | 90.36 | 90.96 | 90.91 | 90.94 | 91.19 | 91.17 | 91.06 | 90.37 |
| Negative correlation, λ = 2.5 | 90.28 | 90.67 | 90.70 | 90.62 | 90.75 | 90.88 | 90.77 | 90.20 |
| Negative correlation, λ = 4 | 89.50 | 90.29 | 90.33 | 90.17 | 90.25 | 90.51 | 90.25 | 89.83 |
| Untrained, first 30 epochs | 9.65 | 85.50 | 84.51 | 86.24 | 77.55 | 71.06 | 77.00 | not run |
| Untrained, 150 epochs | 9.65 | 86.97 | 87.12 | 87.86 | 82.85 | 82.37 | 84.29 | not run |

- **Before training** is `test/accuracy` at step 0: the three experts' answers
  averaged, with the channel injecting nothing.
- **The ladder's group** is the channel ladder's level-1 runs on the same 15
  groups: phase 1, and phase 2c for `mean`. Its gate is the ladder's gate
  reading stage 3.
- **Negative correlation's λ = 0** experts are trained without the ladder's class
  skew, so they aren't the ladder's group.

## Clones

Each group is three copies of the first expert of the ladder's group. Before
training they give the same answers, so the group is just that expert. Each copy
has its own add-on, started from a different random state, so the copies can
drift apart once training starts.

### 1. With copies, hearing the others never helps, but the comparison with different experts is unsettled (weak)

Planned. Backward, the ladder's `mean` led `self` by 0.14. `mean` gives each
expert three times the readings, so there are two possible causes:
- If the lead came from those extra readings, copies would show it too.
- If it came from the others knowing different things, copies wouldn't.

`mean` − `self`:

| | by the score | all 30 epochs, test | all 30 epochs, validation |
|---|---|---|---|
| The ladder's group, backward | +0.14 ± 0.08 (13/15) | +0.08 ± 0.06 (12/15) | +0.01 ± 0.07 (7/15) |
| Clones, backward | −0.06 ± 0.10 (5/15) | −0.05 ± 0.05 (3/15) | −0.06 ± 0.08 (3/15) |
| The ladder's group minus clones, backward | +0.21 ± 0.13 (11/15) | +0.13 ± 0.07 (12/15) | +0.06 ± 0.11 (10/15) |
| The ladder's group minus clones, forward | −0.03 ± 0.10 (7/15) | +0.02 ± 0.06 (10/15) | −0.02 ± 0.07 (7/15) |

With copies, `mean` never beats `self` in any reading. But the different experts'
lead shrinks at fixed epochs and is gone on validation images, so the gap
between the two is unclear there too.

`attention`'s backward deficit to `mean` behaves the same way:
- For the ladder's group it is −0.20 ± 0.06 (1/15) by the score, but −0.03 ± 0.05
  on validation.
- For copies it is about 0.

### 2. Backward, hearing different experts lifts each expert more than hearing copies

Each expert's own accuracy, `mean` − `self`:

| | by the score | all 30 epochs, test | all 30 epochs, validation |
|---|---|---|---|
| The ladder's group, forward | +0.70 ± 0.14 (15/15) | +0.47 ± 0.04 (15/15) | +0.45 ± 0.05 (15/15) |
| Clones, forward | +0.71 ± 0.15 (15/15) | +0.48 ± 0.09 (15/15) | +0.47 ± 0.09 (15/15) |
| The ladder's group, backward | +0.73 ± 0.09 (15/15) | +0.69 ± 0.05 (15/15) | +0.66 ± 0.06 (15/15) |
| Clones, backward | +0.52 ± 0.15 (15/15) | +0.40 ± 0.06 (15/15) | +0.37 ± 0.08 (15/15) |
| The ladder's group minus clones, backward | +0.21 ± 0.18 (12/15) | +0.30 ± 0.08 (14/15) | +0.29 ± 0.10 (14/15) |

- **Forward**, hearing copies lifts each expert as much as hearing different
  experts. The difference is −0.01 ± 0.21, and −0.03 ± 0.09 on validation. So
  that gain comes from the extra readings, not from what the others know.
- **Backward**, about 0.3 of the 0.7 needs different experts.

In neither direction does it reach the group (finding 1).

Each expert's own accuracy is `test/expert_accuracy`, the average of the three
experts' own accuracies (`joint_val/expert_accuracy` on validation).

### 3. Each copy's own add-on makes the copies different

| condition | mean expert | what averaging adds | group | agreement between experts | at least one expert right |
|---|---|---|---|---|---|
| Clones, before training | 82.65 | 0.00 | 82.65 | 100.00 | 82.65 |
| Clones, forward `self` | 87.53 | 1.86 | 89.39 | 87.80 | 95.03 |
| Clones, forward `mean` | 88.24 | 1.18 | 89.42 | 90.20 | 94.39 |
| Clones, forward `attention` | 88.08 | 1.29 | 89.37 | 89.88 | 94.36 |
| Clones, backward `self` | 87.98 | 1.59 | 89.57 | 89.02 | 94.76 |
| Clones, backward `mean` | 88.49 | 1.01 | 89.50 | 91.10 | 94.00 |
| Clones, backward `attention` | 88.31 | 1.18 | 89.49 | 90.31 | 94.26 |
| The ladder's group, before training | 84.32 | 3.90 | 88.22 | 81.77 | 95.20 |
| The ladder's group, forward `self` | 88.20 | 2.24 | 90.44 | 88.04 | 95.41 |

- **Mean expert** is `test/expert_accuracy`: each expert's own accuracy,
  answering alone after the channel has changed it, averaged over the three.
- **Group** is `test/accuracy`, the score.
- **What averaging adds** is group − mean expert.
- **Agreement between experts** is `test/joint_answer_shift/agreement_post`: on
  each image, the share of the three pairs of experts that give the same answer,
  averaged over images. Before training it is `agreement_pre`.
- **At least one expert right** is the sum of the `test/answer_grid` cells
  `all_right`, `some_right_avg_right` and `some_right_avg_wrong`.

Each column is read like the score and averaged over the 15 groups. Later tables
with these columns read them the same way.

Copies that start identical end up almost as varied as three separately trained
experts:
- At least one copy is right on 94.0–95.0% of test images, up from 82.65%.
- The ladder's group reaches 94.8–95.4%. Copies are 0.4–0.9 below it (0–2/15),
  and 0.6–1.0 below on validation (0/15).

`self` does this as much as the other mixers. So it comes from the add-ons, each
correcting its copy differently, not from communication.

Still, copies end 1.0 to 1.3 points below the ladder's group in every arm (0/15;
1.2 to 1.4 on validation). Before training they were 5.6 points below.

### 4. Forward, `attention` locks on to one sender even when every sender reads the same features

`attention` arms, from before training to each run's score window:

| condition | spread of attention | weight on top sender | images on the most common wiring | sender changed since the last epoch |
|---|---|---|---|---|
| Clones, forward | 1.08 to 0.004 | 0.39 to 1.00 | 20% to 97% | 2% |
| The ladder's group, forward | 1.09 to 0.008 | 0.39 to 1.00 | 26% to 97% | 1% |
| Clones, backward | 1.08 to 0.24 | 0.41 to 0.90 | 23% to 35% | 9% |
| The ladder's group, backward | 1.08 to 0.34 | 0.40 to 0.86 | 21% to 31% | 12% |

Forward, each receiver ends up hearing one sender on almost every image. That
happens with copies just as with different experts, although here every sender
reads the same features. So the lock-on doesn't depend on one sender knowing more.
Backward, routing stays softer and depends on the image, in both groups.

- **Spread of attention** is `attn_entropy`: 0 when each receiver puts all its
  weight on one sender, 1.10 when it weights all three equally.
- **Weight on top sender** is `attn_top1`.
- **Images on the most common wiring** is `routing_mode_share/joint`. Take, for
  each image, the sender each receiver hears most; this is the share of images
  where that pattern matches the most common one. 100% is the same routing for
  every image. A fresh random choice per image would give about 4%.
- **Sender changed since the last epoch** is `route_flip_rate`: the share of
  image–receiver pairs whose most-heard sender changed since the previous epoch.
  It isn't logged at step 0.

Values are medians over the 15 runs, each read over its score window. Forward
keys end in `/ex0`, backward in `/ex0/round0`.

## Two trained + one untrained

The ladder's group with its third expert replaced by one with random weights,
which guesses (10.19%) before training. 60 epochs.

### 5. An untrained expert gains 6 points forward and 19 backward by listening

Planned. The untrained expert's own accuracy, read around its own best validation
epoch:

| | `self` | `mean` | `attention` | `mean` − `self` | `attention` − `self` | `attention` − `mean` |
|---|---|---|---|---|---|---|
| Forward, 60 epochs | 79.39 | 85.52 | 86.02 | +6.13 ± 2.53 (14/15) | +6.63 ± 1.31 (15/15) | +0.49 ± 2.14 (4/15) |
| Forward, first 30 epochs | 78.06 | 84.00 | 84.67 | +5.94 ± 3.21 (13/15) | +6.61 ± 1.69 (15/15) | +0.67 ± 2.19 (7/15) |
| Backward, 60 epochs | 67.00 | 85.76 | 85.93 | +18.76 ± 1.94 (15/15) | +18.93 ± 2.10 (15/15) | +0.17 ± 1.25 (5/15) |
| Backward, first 30 epochs | 63.54 | 84.34 | 84.88 | +20.79 ± 1.67 (15/15) | +21.33 ± 1.94 (15/15) | +0.54 ± 1.77 (7/15) |

On validation, at epochs 21–30 and 51–60, listening adds 6.1 to 6.9 points
forward and 19.2 to 22.1 backward (14–15/15).

With only its own add-on, the untrained expert already learns a lot from its
random features. Listening adds 6 to 19 points on top of that, and `attention`
does no better than `mean`.

The number is `test/expert_accuracy/expert2`: its median over the 5 epochs around
its best `joint_val/expert_accuracy/expert2` epoch. The untrained expert is
always the third.

### 6. The group gains far less, and stays below three trained experts

| | forward | backward |
|---|---|---|
| `mean` − `self`, by the score | +0.41 ± 0.13 (14/15) | +0.20 ± 0.12 (13/15) |
| `mean` − `self`, epochs 21–30, validation | +0.29 ± 0.10 (15/15) | +0.15 ± 0.10 (11/15) |
| `mean` − `self`, epochs 51–60, validation | +0.11 ± 0.13 (9/15) | +0.32 ± 0.12 (13/15) |
| `attention` − `self`, by the score | +0.26 ± 0.14 (11/15) | −0.07 ± 0.18 (6/15) |
| `attention` − `self`, epochs 51–60, validation | +0.11 ± 0.13 (11/15) | +0.12 ± 0.10 (11/15) |

The untrained expert gains 6 to 19 points, the group 0.2 to 0.4. Forward, the
group's gain with `mean` fades on validation by epochs 51–60.

Every arm also ends below the ladder's three trained experts over the same 30
epochs:

| first 30 epochs, minus the ladder's same arm | by the score | epochs 21–30, validation |
|---|---|---|
| Forward `self` | −0.85 ± 0.11 (0/15) | −0.81 ± 0.12 (0/15) |
| Forward `mean` | −0.46 ± 0.15 (1/15) | −0.31 ± 0.09 (1/15) |
| Forward `attention` | −0.53 ± 0.12 (0/15) | −0.54 ± 0.08 (0/15) |
| Backward `self` | −0.66 ± 0.10 (0/15) | −0.74 ± 0.09 (0/15) |
| Backward `mean` | −0.66 ± 0.07 (0/15) | −0.53 ± 0.10 (0/15) |
| Backward `attention` | −0.74 ± 0.16 (0/15) | −0.66 ± 0.12 (0/15) |

### 7. What the untrained expert learns, the others already know

First 30 epochs. Here the untrained expert is read in the group's window, not its
own, so its `self` numbers are lower than in finding 5: under `self` it is still
learning when the group peaks.

| condition | untrained expert | trained experts | mean expert | what averaging adds | group | agreement between experts | at least one expert right |
|---|---|---|---|---|---|---|---|
| Before training | 10.19 | 84.46 | 59.70 | 27.34 | 87.04 | 33.81 | 93.41 |
| Forward `self` | 76.47 | 88.13 | 84.15 | 5.45 | 89.60 | 79.67 | 95.62 |
| Forward `mean` | 83.12 | 88.87 | 86.94 | 3.04 | 89.98 | 85.98 | 95.16 |
| Forward `attention` | 84.45 | 88.55 | 87.11 | 2.77 | 89.88 | 86.20 | 95.23 |
| Backward `self` | 42.60 | 88.74 | 73.38 | 16.61 | 90.00 | 57.34 | 95.69 |
| Backward `mean` | 83.57 | 89.10 | 87.22 | 2.92 | 90.14 | 86.47 | 95.10 |
| Backward `attention` | 84.02 | 88.67 | 87.09 | 2.76 | 89.86 | 86.07 | 95.24 |

When it listens (`mean` − `self`), the untrained expert agrees far more with the
other two:
- **Agreement:** +6.3 ± 2.4 forward and +29 ± 6.5 backward (15/15). On
  validation at epochs 21–30 it is +6.0 and +16.0.
- **At least one expert right** shrinks a little: −0.45 ± 0.19 (2/15) forward and
  −0.59 ± 0.19 (0/15) backward. On validation it is −0.68 and −0.53 (0/15).

So its new right answers are on images the trained experts already get right. On
its own (`self`), it is less accurate but more different, and the group scores
about the same. The trained experts gain a little from listening too: +0.74 ±
0.24 (14/15) forward and +0.36 ± 0.27 (13/15) backward with `mean`.

Columns as in finding 3. **Untrained expert** is `test/expert_accuracy/expert2`;
**trained experts** is the average of `expert0` and `expert1`.

### 8. Forward, `attention` doesn't lock into a fixed wire, unlike with three trained experts

| condition | spread of attention | weight on top sender | images on the most common wiring | sender changed since the last epoch |
|---|---|---|---|---|
| Forward | 1.09 to 0.013 | 0.38 to 0.995 | 24% to 45% | 4% |
| Backward | 1.09 to 0.058 | 0.38 to 0.976 | 25% to 46% | 3% |
| The ladder's group, forward | 1.09 to 0.008 | 0.39 to 1.00 | 26% to 97% | 1% |

Each receiver still hears one sender at a time, but forward, which one depends on
the image:
- The most common wiring covers 45% of images, against 97% with three trained
  experts.
- Each receiver's most-heard sender gets 60–81% of images.

From before training, the most common wiring rose from 24% to 45% of images. So
routing did become more consistent, but far less than the ladder group's 26% to
97%. Routes hardly change between epochs (4%), so each image's sender is a
settled choice, not a fresh coin flip. Whom the untrained expert hears wasn't
logged. Columns as in finding 4.

## Negative correlation

Three experts trained together to disagree, at strength λ, without the ladder's
class skew. λ = 0 is the control. The rule for picking λ was fixed before any
channel ran, except that λ = 4 was picked after loosening it. Before any channel:

| λ | each expert alone | the three averaged | agreement between experts |
|---|---|---|---|
| 0 | 89.68 | 90.36 | 94.1 |
| 2.5 | 61.51 | 90.28 | 32.0 |
| 4 | 38.73 | 89.50 | 6.5 |

At λ = 4 each expert alone is right on 39% of images and the three rarely agree.
Yet their average is right on 89.5%, which is 0.86 ± 0.22 below λ = 0 (0/15).

### 9. Backward, hearing the others gains more as the experts disagree more

Planned. `mean` − `self`, minus the same gap at λ = 0:

| | by the score | all 30 epochs, test | all 30 epochs, validation |
|---|---|---|---|
| Backward, λ = 2.5 | +0.15 ± 0.15 (9/15) | +0.15 ± 0.07 (12/15) | +0.19 ± 0.06 (15/15) |
| Backward, λ = 4 | +0.29 ± 0.16 (12/15) | +0.27 ± 0.10 (14/15) | +0.28 ± 0.06 (15/15) |
| Forward, λ = 2.5 | +0.08 ± 0.10 (8/15) | +0.05 ± 0.06 (10/15) | +0.07 ± 0.06 (12/15) |
| Forward, λ = 4 | +0.08 ± 0.14 (11/15) | +0.09 ± 0.04 (13/15) | +0.08 ± 0.05 (11/15) |

Backward `mean` − `self` itself goes −0.02, +0.13 and +0.26 by the score as λ
rises. On validation it is −0.09, +0.11 and +0.20, the last on 15/15. Forward, a
smaller gain of about 0.08 shows at fixed epochs but not by the score, so forward
is weak.

Two cautions:
- λ = 4 was picked after the dial was read, and its experts start 0.86 points
  lower.
- The gap grows because `self` loses more than `mean` as λ rises (finding 11),
  not because `mean` gains.

### 10. Every channel undoes the disagreement within the first epoch

λ = 4:

| condition | each expert alone | agreement between experts | group |
|---|---|---|---|
| Before training | 38.73 | 6.53 | 89.50 |
| Forward `self` | 88.25 | 88.45 | 90.29 |
| Forward `mean` | 88.95 | 90.91 | 90.33 |
| Forward `attention` | 88.52 | 89.66 | 90.17 |
| Backward `self` | 87.92 | 87.54 | 90.25 |
| Backward `mean` | 89.21 | 91.10 | 90.51 |
| Backward `attention` | 88.44 | 89.02 | 90.25 |
| Gate | 38.73 | 6.53 | 89.83 |

After one epoch, validation agreement is already 81–88%, and each expert alone is
right on 84–88%. The training goal rewards only the averaged answer. Yet every
add-on, `self`'s included, turns its expert into a good one on its own, and the
disagreement is gone. At λ = 2.5, agreement goes from 32.0 to 88.9–91.2. The gate
leaves the experts as they are.

Columns as in finding 3; **each expert alone** is the mean expert. "At least one
expert right" is left out because it can't be compared across λ: experts that
spread their guesses cover more classes by chance.

### 11. Every arm ends lower as λ rises, `self` most

| λ = 4 minus λ = 0 | by the score | all 30 epochs, validation |
|---|---|---|
| Before training | −0.86 ± 0.22 (0/15) | −0.77 ± 0.28 (2/15) |
| Gate | −0.53 ± 0.17 (1/15) | −0.78 ± 0.13 (0/15) |
| Forward `self` | −0.67 ± 0.11 (0/15) | −0.70 ± 0.09 (0/15) |
| Forward `mean` | −0.59 ± 0.09 (0/15) | −0.62 ± 0.08 (0/15) |
| Forward `attention` | −0.77 ± 0.08 (0/15) | −0.72 ± 0.11 (0/15) |
| Backward `self` | −0.94 ± 0.11 (0/15) | −1.18 ± 0.09 (0/15) |
| Backward `mean` | −0.65 ± 0.10 (0/15) | −0.89 ± 0.09 (0/15) |
| Backward `attention` | −0.81 ± 0.12 (0/15) | −1.01 ± 0.10 (0/15) |

Experts trained to disagree are a worse starting point for every method. Backward,
`self` loses most.

On validation, the "before training" row is step 0 on the validation images. λ =
0 and λ = 4 are different experts on the same trial seed.

### 12. The gate is beaten by every channel at every λ

| channel − gate | λ = 0 | λ = 2.5 | λ = 4 |
|---|---|---|---|
| Forward `self` | +0.60 ± 0.15 (15/15) | +0.46 ± 0.12 (14/15) | +0.46 ± 0.13 (15/15) |
| Forward `mean` | +0.55 ± 0.15 (15/15) | +0.50 ± 0.09 (15/15) | +0.49 ± 0.14 (14/15) |
| Forward `attention` | +0.58 ± 0.13 (15/15) | +0.41 ± 0.09 (15/15) | +0.33 ± 0.14 (12/15) |
| Backward `self` | +0.83 ± 0.14 (15/15) | +0.55 ± 0.08 (15/15) | +0.42 ± 0.09 (15/15) |
| Backward `mean` | +0.80 ± 0.16 (15/15) | +0.68 ± 0.10 (15/15) | +0.68 ± 0.13 (15/15) |
| Backward `attention` | +0.69 ± 0.15 (15/15) | +0.57 ± 0.09 (15/15) | +0.42 ± 0.11 (15/15) |
| Gate − before training | +0.01 ± 0.07 (9/15) | −0.08 ± 0.12 (6/15) | +0.34 ± 0.28 (10/15) |

On validation over all 30 epochs, every channel arm leads the gate by 0.46 to
0.91 (14–15/15). Even when the experts disagree, the gate weights them image by
image and still does barely better than averaging them: +0.07, −0.06 and +0.06
on validation.

### 13. Backward, `attention` commits to one sender more as experts disagree, and falls further behind `mean`

| condition | spread of attention | weight on top sender | images on the most common wiring | `attention` − `mean`, by the score | all 30 epochs, validation |
|---|---|---|---|---|---|
| Forward, λ = 0 | 1.09 to 0.010 | 0.39 to 1.00 | 27% to 93% | +0.03 ± 0.07 (7/15) | +0.02 ± 0.04 (10/15) |
| Forward, λ = 2.5 | 1.09 to 0.002 | 0.39 to 1.00 | 29% to 99% | −0.08 ± 0.08 (5/15) | −0.07 ± 0.04 (4/15) |
| Forward, λ = 4 | 1.09 to 0.001 | 0.38 to 1.00 | 25% to 99% | −0.16 ± 0.10 (4/15) | −0.07 ± 0.06 (5/15) |
| Backward, λ = 0 | 1.08 to 0.66 | 0.40 to 0.70 | 19% to 17% | −0.11 ± 0.08 (4/15) | −0.07 ± 0.05 (6/15) |
| Backward, λ = 2.5 | 1.08 to 0.23 | 0.40 to 0.91 | 21% to 30% | −0.11 ± 0.09 (4/15) | −0.01 ± 0.08 (7/15) |
| Backward, λ = 4 | 1.08 to 0.10 | 0.39 to 0.96 | 21% to 47% | −0.26 ± 0.12 (2/15) | −0.18 ± 0.11 (3/15) |

Forward, `attention` is a fixed wire at every λ. Backward, it gets sharper and
more fixed as λ rises, and at λ = 4 it trails `mean` on validation too. Routing
columns as in finding 4.

## Untrained

Three experts with random weights, so before training the group guesses (9.65%).
Only the add-on learns. 150 epochs.

### 14. The add-on alone learns a lot from random features

Test accuracy with `self`, averaged over the 15 groups at fixed epochs:

| | epoch 1 | 10 | 30 | 60 | 150 | three trained experts, `self` |
|---|---|---|---|---|---|---|
| Forward | 67.2 | 83.1 | 85.4 | 86.3 | 87.1 | 90.44 |
| Backward | 40.0 | 71.4 | 77.6 | 79.7 | 82.9 | 90.66 |

With each expert hearing only itself, the add-on reads frozen random features and
reaches 87% forward and 83% backward. By the score that is 3.5 and 7.8 points
below the ladder's trained experts after their 30 epochs.

Backward is still climbing at 150 epochs. `self`'s score window ends after epoch
140 on 10 of 15 groups.

### 15. By 150 epochs, `attention` beats each expert's own add-on

Planned, with finding 16. `attention` − `self`:

| | by the score | validation, last 10 epochs read |
|---|---|---|
| Forward, first 30 epochs | +0.75 ± 0.77 (10/15) | +0.65 ± 0.78 (11/15) |
| Forward, 150 epochs | +0.88 ± 0.16 (15/15) | +0.53 ± 0.22 (14/15) |
| Backward, first 30 epochs | −0.55 ± 1.80 (5/15) | −1.32 ± 1.90 (4/15) |
| Backward, 150 epochs | +1.44 ± 0.80 (14/15) | +1.61 ± 0.93 (14/15) |

Validation reads epochs 21–30 for the first 30 epochs and 141–150 for 150
epochs. At 30 epochs the difference is within noise.

The 150-epoch numbers are lower bounds. Backward `attention` was still gaining in
the last 10 epochs: +0.17 ± 0.10 (13/15) over the 10 before.

### 16. `mean` learns slower and unevenly

Planned. `mean` − `self`:

| | by the score | validation, last 10 epochs read | groups more than 2 points behind `self` |
|---|---|---|---|
| Forward, first 30 epochs | −0.99 ± 1.66 (8/15) | −1.37 ± 1.68 (7/15) | 3 |
| Forward, 150 epochs | +0.14 ± 0.92 (12/15) | −0.29 ± 0.92 (8/15) | 1 |
| Backward, first 30 epochs | −6.50 ± 4.23 (4/15) | −7.75 ± 3.90 (3/15) | 9 |
| Backward, 150 epochs | −0.47 ± 2.29 (12/15) | −0.51 ± 2.24 (12/15) | 3 |

By 150 epochs, group by group:
- **Backward**, `mean` is 0.5 to 3.3 points ahead of `self` on 12 groups, and 6.0,
  6.9 and 12.6 behind on the other three.
- **Forward**, it is 0.1 to 1.1 ahead on 12 groups and 6.1 behind on one.

So `mean` usually ends a little ahead, but on a few groups it lags far behind,
which makes its average unreadable. At 150 epochs `attention` leads `mean` by
0.74 forward and 1.91 backward. Both ranges include 0, for the same reason.

### 17. Forward, `attention` gains by making the experts better and alike; backward, it also widens what they cover

`attention` − `self` at 150 epochs:

| | mean expert | agreement between experts | at least one expert right | group |
|---|---|---|---|---|
| Forward | +4.94 ± 1.62 (14/15) | +12.08 ± 3.40 (14/15) | −1.57 ± 0.33 (0/15) | +0.88 ± 0.16 (15/15) |
| Backward | +1.83 ± 1.72 (11/15) | +2.03 ± 3.40 (9/15) | +0.76 ± 0.30 (13/15) | +1.44 ± 0.80 (14/15) |

On validation at epochs 141–150, "at least one expert right" is −1.85 ± 0.50
(0/15) forward and +0.69 ± 0.21 (14/15) backward.
- **Forward:** the experts get much better alone and more alike, and between them
  cover fewer images.
- **Backward:** they cover more.

Columns as in finding 3.

### 18. Here `attention` routes image by image, in both directions

| condition | spread of attention | weight on top sender | images on the most common wiring | sender changed since the last epoch |
|---|---|---|---|---|
| Forward | 1.10 to 0.27 | 0.35 to 0.89 | 69% to 35% | 4% |
| Backward | 1.10 to 0.75 | 0.35 to 0.68 | 92% to 28% | 4% |

Before training, attention is spread evenly, yet the same sender edges ahead on
most images: 69% and 92% of images share one wiring. After training, the wiring
depends on the image and stays put between epochs. Forward, this group and the
two-trained group are the only ones where routing doesn't end on one wiring for
almost every image. Columns as in finding 4.

## What this means

- **The channel can carry information, but a group of trained experts has little
  to receive.**
  - An untrained expert gains 6 to 19 points by listening.
  - The add-on alone reaches 83–87% from random features.
  - Yet the group gains 0.2 to 0.4, because what the untrained expert learns is
    already in the group. Forward, even that fades by 60 epochs.
- **Otherwise, hearing others helps the group only a little.**
  - With experts trained to disagree: about 0.3 at the strongest disagreement,
    backward only. It holds on validation. Forward the gain is smaller and
    unclear.
  - With copies, `mean` never beat `self`. But the ladder's own backward lead, the
    one the copies were compared with, didn't hold on validation, so copies leave
    that question open.
- **`attention` never clearly beats `mean`.**
  - Its only clear gain for the group over `self` comes with three untrained
    experts: about 1 point after 150 epochs, where it routes image by image.
    There `mean` is too uneven to compare.
  - With one untrained member, it lifts that expert as much as `mean` does, and no
    more.
  - Forward, with trained experts, it usually ends as a fixed wire, even when
    every sender reads the same features.
- **Training pulls groups of trained experts to the same agreement.**
  - They start at 100% for copies, 94% (λ = 0) or 6.5% (λ = 4) for the
    negative-correlation experts, and 82% for the ladder's group.
  - They end at about 88–93%.
  - The add-on does this on its own: `self` does it too.
- **The gate never competes.** Every channel arm beats it at every λ, and it
  barely beats averaging the experts.

## Limits

- **Planned versus exploratory.** Only findings 1, 5, 9, 15 and 16 answer
  questions fixed before the runs. The rest is exploratory, and many differences
  were read.
- **λ = 4** was picked after the dial was read, under a loosened rule. Its
  experts start 0.86 lower.
- **Lower bounds.** The untrained case was still climbing backward at 150 epochs.
- **Narrow groups.**
  - Clones copy only the first expert of each group.
  - The untrained expert is always the third.
  - Negative correlation's experts are trained without the ladder's class skew,
    so λ = 0 isn't the ladder's group.
- **Routing.** Whom each expert hears wasn't logged. So the routing findings say
  how sharp and how fixed routing is, not who is heard.
- **Scope.**
  - One task, one small network, one learning rate, and level 1 at r = 8 only.
  - The runs came from uncommitted code (`git_dirty` is true on every run). That
    code is saved in `cache/degenerate_cases/`.
- **Small effects.** As in the ladder, treat differences under about 0.1 as
  unresolved unless they also hold on validation images.

## Open questions

- Does the ladder's backward `mean` lead exist? It needs a rerun before copies can
  say what it is made of.
- Does `mean`'s backward gain with disagreeing experts hold at a λ picked in
  advance? λ = 3's experts are cached.
- Whom does the untrained expert hear, and do the trained experts stop hearing it?
  `attn_heard` on an exact lab replay would show it.
- Why does `mean` stall on some untrained groups?
- Does `attention`'s lead in the untrained case grow or shrink after 150 epochs?

## Source

- **This study:** the MLflow experiments `degenerate-cases-clones`,
  `degenerate-cases-two-trained-one-untrained`,
  `degenerate-cases-negative-correlation` and `degenerate-cases-untrained`, 585
  runs, all finished.
- **Reference:** the channel ladder's level-1 runs at r = 8, in
  `channel-ladder-phase1` and `channel-ladder-phase2c`.
- **Design and rules:** `docs/studies/degenerate-cases.md`. Its own results
  section uses t-based ranges.
- **Analysis:** the scripts, their outputs and the checks fixed before reading
  are in `cache/degenerate_cases/analysis/`.
