# Study: degenerate cases

How does the communicating ensemble behave when its experts aren't the standard
group? The [channel ladder](channel-ladder.md) held the group fixed and varied
the channel. This study holds the channel fixed and varies the group, in four
cases. The study's code is `studies/degenerate_cases/`.

## The experiment

Proposed and accepted 2026-11-01. The rules below are fixed before any run.

**Held from the ladder.** Every arm is level 1 at $r = 8$, at the ladder's
chosen pairs (forward stage2 → stage3, backward stage3 → stage1), with the
mixers `attention`, `self` and `mean`. That is 6 arms per group, on trial seeds
0–14, at learning rate 1e-3 with the joint loss. Arms compare trial seed by
trial seed, and each run is read as the median test accuracy over the 5 epochs
around its best joint_val epoch (`testbed/results.py`), except where a case says
otherwise.

| case | the group | epochs | arms | runs |
|---|---|---|---|---|
| `clones` | three copies of the first expert of the ladder's group | 30 | 6 | 90 |
| `two-trained-one-untrained` | the ladder's group, with the third expert untrained | 60 | 6 | 90 |
| `negative-correlation` | three experts trained to disagree, at three λ | 30 | 18, plus 3 MoE | 315 |
| `untrained` | three experts with random weights | 150 | 6 | 90 |

The joint learning rate is constant, so a long run holds a shorter run's epochs
exactly. The `untrained` case is read at 30 epochs as well as at 150, and
`two-trained-one-untrained` at 30 as well as at 60, each by applying the rule
to those first epochs only.

**`clones`: is the ladder's one positive finding about knowledge or width?**
Backward `mean` − `self` was +0.143 ± 0.081 in the ladder, but `mean` also gives
each receiver 3 × 8 = 24 readings. Clones have nothing different to share and
still get those readings.

| if the +0.14 comes from | different experts | clones |
|---|---|---|
| different knowledge | +0.14 | about 0 |
| more readings | +0.14 | about +0.14 |

The result is the ladder's `mean` − `self` minus the clones' `mean` − `self`,
trial seed by trial seed. Clones stop being identical once training starts,
because the channels' initialisations differ, so gain over bagging means little
here. With identical channel initialisations too, every mixer gives the same
answers (`tests/test_invariants.py`).

**`two-trained-one-untrained`: can the channel carry useful information at
all?** This is the study's positive control. The result is the untrained
expert's own accuracy under `mean` and `attention` minus under `self`. The
number is `test/expert_accuracy/expert2`, its median over the 5 epochs around
the best epoch of `joint_val/expert_accuracy/expert2`, because the ensemble's
best epoch can come while the untrained expert is still learning. Without a gain
here, a null in another case says little about the channel.

**`negative-correlation`: does hearing others matter more when experts start
out disagreeing?** The result is `mean` − `self` at each λ minus the same gap at
λ = 0. λ = 0 is the control, not the ladder: these experts have no class skew. The
MoE, reading stage3, is the competitor at each λ. Read agreement before and
after training too, because the joint loss rewards agreement. Oracle − bagging
can't be compared across λ, since different guesses cover more classes.

λ is picked before any channel runs, from what the experts do on joint_val alone
(`studies/degenerate_cases/dial.py`):

- the middle λ is the smallest with agreement below 50% on every trial seed;
- the far λ is the largest whose bagging, averaged over trial seeds, is within
  1.5 points of λ = 0's.

The dial ran on 2026-11-01, on all 15 trial seeds, and the study runs λ = 0, 2.5
and 4. On joint_val:

| λ | mean expert accuracy | bagging | agreement (lowest, highest trial seed) |
|---|---|---|---|
| 0 | 89.92 | 90.56 | 94.0 (93.3, 94.8) |
| 2 | 74.61 | 90.58 | 58.6 (49.1, 71.2) |
| 2.5 | 61.61 | 90.48 | 32.1 (30.6, 33.6) |
| 3 | 54.64 | 90.21 | 23.1 (20.2, 24.9) |
| 4 | 38.76 | 89.79 | 6.6 (3.4, 14.8) |
| 5 | 32.01 | 88.32 | 0.7 (0.0, 3.4) |

The limit on bagging was changed after this table was read. It was 0.5 points,
which picks λ = 3, but 2.5 and 3 are nearly the same setting, while λ = 4 is a
real step further at 0.77 points below λ = 0. The change rests on this table
alone, though the pilot had already put λ = 4 through the channel on one trial
seed. λ = 4 is therefore weaker evidence than a pick made in advance, and its
experts know slightly less: read its results with that 0.77 in mind. λ = 3's
pools are cached and pair with the rest, so it can be added later.

**`untrained`: does listening help when nobody knows anything?** Bagging is
guessing here, so gain over bagging means nothing. The results are `mean` −
`self` and `attention` − `self`, at 30 and at 150 epochs. In the pilot, backward
`self` and `attention` were still climbing at 150 epochs, so read their numbers
as lower bounds.

**Forward `attention` is a fixed choice of sender.** In the ladder's forward
runs, each receiver settles on one sender by the first epoch. The study leaves
that as it is, so that it pairs with the ladder, and reads forward `attention`
− `mean` as one fixed sender against the average, not as choosing per input.

**Cost on the laptop**, one run at a time: about 1.2 hours for `clones`, 2.4 for
`two-trained-one-untrained`, 4 for `negative-correlation` and 6 for `untrained`,
plus about 30 minutes for the λ dial. On the lab machine it is about half that,
one run per GPU.

## Running the study

Pick λ first, and put what it prints in `LAMS`
(`studies/degenerate_cases/__init__.py`):

```bash
uv run python -m studies.degenerate_cases.dial
```

`studies/degenerate_cases/plan.py` prints the command for every arm, in the order
to run them. Flags given to it are added to every line:

```bash
uv run python -m studies.degenerate_cases.plan > arms.txt
bash arms.txt
```

`studies/degenerate_cases/run.py` runs one arm over the trial seeds:

```bash
uv run python -m studies.degenerate_cases.run --case clones --direction forward --mixer self
uv run python -m studies.degenerate_cases.run --case negative-correlation --lam 2.5 \
  --direction backward --mixer mean
uv run python -m studies.degenerate_cases.run --case negative-correlation --lam 2.5 --moe
```

Each case is its own MLflow experiment, `degenerate-cases-<case>`, and runs are
tagged with the study, the case and, where they have one, λ. Every other flag is
`testbed.run`'s, and a flag given here overrides the study's default.

## Results

All 585 runs finished, 2026-11-01 to 02 on the laptop, in about 14 hours. They
are the MLflow experiments `degenerate-cases-<case>`, read by
`studies/degenerate_cases/read.py`. Every difference is taken trial seed by
trial seed and shown as its mean over the 15 trial seeds, in accuracy points,
± the 95% half-width, then on how many trial seeds it is positive. Clones and
the mixed group used the ladder's own pools on every trial seed. The code ran
uncommitted on top of 9ab5621 (`git_dirty` is true on every run); what it was
is saved in `cache/degenerate_cases/`.

Many differences are read below, and each interval is for one difference on
its own. The four questions above are the results; everything else is
exploratory.

### Clones: the ladder's one positive finding is about knowledge

| | backward | forward |
|---|---|---|
| `mean` − `self`, the ladder's three different experts | +0.14 ± 0.08 (13/15) | −0.01 ± 0.08 (6/15) |
| `mean` − `self`, clones | −0.06 ± 0.11 (5/15) | +0.03 ± 0.08 (7/15) |
| the ladder's gap minus the clones' | **+0.21 ± 0.14** (11/15) | −0.03 ± 0.11 (7/15) |

Backward, hearing the others adds about 0.14 when they know different things,
and nothing when they are copies. So the gap needs experts that know different
things; the extra readings `mean` gets don't make it on their own. Forward,
hearing the others adds nothing either way.

- **This settles which, not how much.** The +0.14 is the one positive the
  ladder found among many readings, and it hasn't been rerun backward, so its
  size may be overstated. The clones show it needs different experts, not that
  it is 0.14.
- **It disagrees with an earlier hint.** The ladder's `mean` at $r = 8$ was
  only +0.010 ± 0.123 above `self` at $r = 24$, which pointed to width. Clones
  are the cleaner test: `self` at $r = 24$ also widens what each expert writes,
  while clones change only what the others know.
- **A wider add-on of each expert's own gets there too.** In the ladder,
  backward `self` climbs with the width (90.66, 90.75, 90.79 and 90.86 at
  $r = 8$, 16, 24 and 32) while `mean` stays at 90.80 to 90.88. Hearing
  different experts is one way to reach that accuracy at $r = 8$, not
  something only communication can give.

- `attention` − `mean`, backward, is −0.20 ± 0.06 (1/15) for the ladder's
  experts but −0.01 ± 0.11 for clones. `attention` misses what `mean` gets only
  when there is something different to hear.
- Clones end about 1 point below three different experts in every arm (forward
  −1.02 to −1.05, backward −1.09 to −1.30, 0/15 each), from 5.6 points below at
  step 0 (bagging 82.65 against 88.22).

### Two trained + one untrained: the channel carries information

The untrained expert's own accuracy, read around its own best joint_val epoch,
over 60 epochs. At step 0 it is 10.19, guessing.

| | `self` | `mean` | `attention` | `mean` − `self` | `attention` − `self` |
|---|---|---|---|---|---|
| forward | 79.39 | 85.52 | 86.02 | +6.13 ± 2.72 (14/15) | +6.63 ± 1.41 (15/15) |
| backward | 67.00 | 85.76 | 85.93 | +18.76 ± 2.08 (15/15) | +18.93 ± 2.25 (15/15) |

Listening lifts the untrained expert by 6 points forward and 19 backward, on
almost every trial seed, so the channel can carry what the others know. The
first 30 epochs give the same picture (+5.94 and +6.61 forward, +20.79 and
+21.33 backward). `attention` does no better than `mean` (+0.49 ± 2.29 forward,
+0.17 ± 1.34 backward).

The ensemble gains much less: `mean` − `self` is +0.41 ± 0.14 (14/15) forward
and +0.20 ± 0.13 (13/15) backward, and every arm stays 0.4 to 0.8 below the
ladder's three trained experts. What the untrained expert learns by listening
is mostly what the other two already know.

### Negative correlation: hearing others may gain a little as experts disagree, backward only

`mean` − `self` at each λ, and minus the same gap at λ = 0:

| λ | agreement at step 0 | backward | backward, minus λ = 0 | forward | forward, minus λ = 0 |
|---|---|---|---|---|---|
| 0 | 94.1 | −0.02 ± 0.12 (7/15) | | −0.05 ± 0.06 (6/15) | |
| 2.5 | 32.0 | +0.13 ± 0.09 (11/15) | +0.15 ± 0.16 (9/15) | +0.03 ± 0.08 (7/15) | +0.08 ± 0.10 (8/15) |
| 4 | 6.5 | +0.26 ± 0.14 (13/15) | **+0.29 ± 0.17** (12/15) | +0.03 ± 0.13 (7/15) | +0.08 ± 0.15 (11/15) |

Backward, `mean` gains over `self` as the experts disagree more, the same
direction as the clones. Forward, there is no gain at any λ. Read λ = 4 with
two things in mind: it was picked after the dial was read, and its experts
start 0.86 points lower on test (bagging 89.50 against 90.36). The gap grows
because `self` falls further than `mean` (backward, `self` 91.19 to 90.25,
`mean` 91.17 to 90.51); every arm ends lower as λ rises.

- **The joint loss undoes the disagreement in every arm.** At λ = 4 agreement
  goes from 6.5 at step 0 to 87.5 (`self`) to 91.1 (`mean`) after training.
- **The MoE is below every channel arm at every λ**, by 0.33 to 0.83 points
  (0 to 3 of 15). It is about bagging at λ = 0 (90.37 against 90.36) and only
  0.33 above it at λ = 4.

### Untrained: `attention` helps, `mean` is erratic

| | first 30 epochs | first 150 epochs |
|---|---|---|
| forward, `attention` − `self` | +0.75 ± 0.83 (10/15) | +0.88 ± 0.17 (15/15) |
| forward, `mean` − `self` | −0.99 ± 1.78 (8/15) | +0.14 ± 0.99 (12/15) |
| backward, `attention` − `self` | −0.55 ± 1.93 (5/15) | +1.44 ± 0.86 (14/15) |
| backward, `mean` − `self` | −6.50 ± 4.54 (4/15) | −0.47 ± 2.45 (12/15) |

With experts that know nothing, `attention` gains about 1 point over each
expert's own add-on by 150 epochs, on almost every trial seed. `mean` varies so
much between trial seeds that its gap is unreadable, and at 30 epochs it is far
behind. Backward, the window ends in the last 10 epochs for 9 to 10 of the 15
runs in each arm, so those numbers are lower bounds. Unlike in every other
case, `attention` beats `mean` here (+0.74 forward, +1.91 backward, both with
wide intervals), and here, forward too, it routes example by example (below).
The add-on alone reaches 86.97 forward and 82.85 backward, 3.5 and 7.8 points
below three trained experts.

### Where attention routes, and where it is a fixed wire

Medians over the 15 runs of each `attention` arm, each run read over its own
window. Entropy is 0 for one sender and log 3 = 1.10 for all three equally.
Mode share is the share of examples routed the most common way: 1 is the same
routing for every example.

| case | forward entropy | forward mode share | backward entropy | backward mode share |
|---|---|---|---|---|
| clones | 0.004 | 0.97 | 0.24 | 0.35 |
| negative correlation, λ = 0 | 0.010 | 0.93 | 0.66 | 0.17 |
| negative correlation, λ = 2.5 | 0.002 | 0.99 | 0.23 | 0.30 |
| negative correlation, λ = 4 | 0.001 | 0.99 | 0.10 | 0.47 |
| two trained + one untrained | 0.013 | 0.45 | 0.06 | 0.46 |
| untrained | 0.27 | 0.35 | 0.75 | 0.28 |

- **Forward, with clones and with negative correlation, attention is the fixed
  wire the ladder found.** Each receiver hears one sender, the same for almost
  every example, so forward `attention` − `mean` there is one fixed sender
  against the average.
- **Forward with an untrained expert, it routes.** In the mixed group it still
  picks one sender at a time, but which one changes with the example; in the
  untrained group it is also softer.
- **Backward, it routes in every case**, and commits to one sender more as the
  experts disagree more (entropy 0.66, 0.23, 0.10 as λ goes 0, 2.5, 4), while
  falling further behind `mean` (−0.11, −0.11, −0.26).

### Overall

- Hearing other experts helps only when they know something different, and
  only backward and by little: about 0.14 to 0.29 points. That is an upper end:
  0.14 is the ladder's one positive among many readings, and 0.29 comes from
  λ = 4, picked after the dial was read.
- The channel can carry a lot: an untrained expert gains 6 to 19 points by
  listening. The ensemble gains little from it, because what that expert
  learns is already in the group.
- `attention` never beats `mean` by a detectable amount in a group that has
  trained experts. It helps only when no expert knows anything, where it
  routes example by example in both directions.
- Pilot hints from one or two trial seeds overstated several effects: λ = 4
  forward `mean` − `self` was +0.42 in the pilot and is +0.03 over 15 trial
  seeds, and clones cost about 1 point, not 0.5.
