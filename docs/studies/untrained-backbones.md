# Study: untrained backbones

**Question.** When the experts are never trained (frozen at their random
initialisation), how accurate does the ensemble become if only the
communication channel between them is trained? How does that compare with the
same channel on trained experts, and how much of it depends on the experts
communicating?

**Status.** One run of the grid below, 5 trial seeds per arm. The
results are measured. The explanation in [Interpretation](#interpretation) is
a hypothesis that has not been tested yet.

## Background

### The ensemble

| term | meaning |
|---|---|
| expert | a small CNN classifier (`cnn3`: three conv stages, then a two-layer head; 98,442 weights). The ensemble has M = 3 experts. Experts are frozen: no expert weight changes after expert training. |
| stage, point | each expert is cut into named stages `stage1`, `stage2`, `stage3`, `head`. A point is the output of a stage. On 28×28 inputs, `stage1` is 16×14×14 = 3,136 numbers, `stage2` 32×7×7 = 1,568, `stage3` 64×3×3 = 576. |
| trained backbone | experts trained for 10 epochs on their own split, each on its own class mix drawn from a Dirichlet(α = 1) (class skew), so they know different things. |
| untrained backbone | the same experts with 0 training epochs: each is exactly the network its trained counterpart started from, with the same initial weights. |
| bagging | the ensemble's answer with no communication: the average of the experts' class probabilities. |

### The communication channel

An **exchange** reads every expert's features at a read point, mixes them
across experts, and adds an **injection** to every expert's features at a
write point. Everything the expert computes after the write point then runs
on the modified features. Only the channel trains.

| direction | read → write | what runs after the write |
|---|---|---|
| forward | `stage1` → `stage2` | `stage3` and `head` |
| backward | `stage2` → `stage1` | `stage2` again, then `stage3` and `head` (one round) |

The channel has three parts ([Channel](../testbed/channel.md) defines them in
full):

| part | one per | what it does |
|---|---|---|
| encoder | expert | reads the expert's whole feature map at the read point (no pooling) through r = 8 learned linear templates, and builds a message (query, key, value) from those 8 readings |
| mixer | exchange | decides what each expert receives from all the messages |
| decoder | expert | turns what its expert received into the strengths of r = 8 learned patterns, each the size of the whole write point, and injects their sum |

The decoder's patterns start at zero, so an untrained channel injects nothing
and every run starts exactly at bagging.

**Levels** set where the channel is nonlinear. Both levels have exactly the
same weights.

| level | encoder | decoder | nonlinearity |
|---|---|---|---|
| 1 | `linear_taper` | `linear_taper` | none apart from the mixer's softmax |
| 3 | `linear_taper_gelu` | `linear_taper_gelu` | a GELU on the encoder's 8 readings and one on the decoder's 8 strengths |

**Mixers** decide what each expert receives. Expert $i$ sends value $v_i$,
query $q_i$ and key $k_i$, and receives $m_i$:

| mixer | $m_i$ | role |
|---|---|---|
| `attention` | $\sum_j w_{ij} v_j$, with $w_{ij} = \operatorname{softmax}_j (q_i \cdot k_j / \sqrt{d_k})$, $j$ including $i$ | the channel's own mixer |
| `rmsnorm_attention` | the same, with $q_i$ and $k_j$ each divided by their RMS first (QK-norm), so the attention logits are bounded | attention that cannot collapse onto one sender through growing query and key lengths ([QK-norm study](rmsnorm-attention.md)) |
| `mean` | $\frac{1}{M} \sum_j v_j$, the same for every expert | control: hears every expert, chooses no one |
| `self` | $v_i$, the expert's own message | control: no information crosses between experts |

Under `self`, each expert's encoder and decoder form a trained path from its
own read point to its own write point. Forward, that is a learned bypass
around `stage2`: `stage2`'s output plus a learned function of `stage1`'s
output. At level 1 it is affine and of rank at most 8. Backward, it is a
learned feedback from `stage2` into `stage1`. The other three mixers keep this
own path: each expert's own message is part of what it receives. So
**mixer − `self`** measures what mixing in the other experts' messages adds on
top of each expert's own trained path.

### Training and scoring

- **Data.** Fashion-MNIST. The 60k training images are split per class into
  `expert` (30k, trains the experts), `joint` (24k, trains the channel) and
  `joint_val` (6k, picks epochs). The 10k test set is only read.
- **Loss.** NLL of the ensemble's averaged prediction. Adam, learning rate
  1e-3, batch 128, 30 epochs.
- **Run score** (`testbed/results.py`). Take the epoch with the highest
  `joint_val` accuracy and the 5 trained epochs centred on it, shifted inward
  at the ends of the run. A run's score for a metric is its median over those
  5 epochs. Nothing selects on test.
- **Pairing.** A trial seed fixes the experts, the channel's initial weights
  and the batch order. Every arm on the same trial seed gets the same experts.
  An untrained expert is its trained counterpart's starting point. So arms
  are compared seed by seed, and between backbones too.

### Metrics used here

All are defined in [Metrics](../testbed/metrics.md).

| metric | meaning |
|---|---|
| `test/accuracy` | the ensemble's test accuracy: argmax of the averaged probabilities |
| `test/expert_accuracy` | share of (example, expert) pairs where the expert's own answer, after communication, is correct |
| `msg_norm/injection` | mean L2 norm of an injection, per expert and example, on `joint_val` |
| `joint/accuracy` − `joint_val/accuracy` | how far training-split accuracy runs ahead of held-out accuracy (overfitting) |

## Design

| axis | values |
|---|---|
| backbone | trained (10 expert epochs), untrained (0) |
| direction | forward (`stage1` → `stage2`), backward (`stage2` → `stage1`, 1 round) |
| level | 1, 3 |
| mixer | `attention`, `rmsnorm_attention`, `mean`, `self` |
| trial seeds | 0–4 |

That is 32 arms and 160 runs. Fixed: `cnn3`, M = 3, class skew α = 1, r = 8,
$d_k$ = 16, $d_v$ = 64.

**Channel size.** At r = 8 both exchanges have 39,024 channel weights per
expert ([Channel](../testbed/channel.md) §5). That is 117,072 for the three
experts, 1.19× the weights of one expert. Most of them sit in the side that
touches `stage1`: the forward encoder (25,960) and the backward decoder
(25,608).

## Results

Values are the mean ± SE over the 5 trial seeds, in accuracy points.
Differences marked "paired" are taken seed by seed first. With n = 5,
differences below about ±0.2 points on trained backbones cannot be resolved.
On untrained backbones the per-seed spread is larger (see
[Details](#details)).

### R1. Training only the channel takes untrained backbones from chance to 81–87%

Test accuracy:

| backbone | direction | level | bagging (step 0) | `attention` | `rmsnorm_attention` | `mean` | `self` |
|---|---|---|---|---|---|---|---|
| trained | forward | 1 | 87.60 ± 1.40 | 90.40 ± 0.12 | 90.37 ± 0.13 | 90.47 ± 0.12 | 90.39 ± 0.07 |
| trained | forward | 3 | 87.60 ± 1.40 | 90.29 ± 0.11 | 90.16 ± 0.15 | 90.35 ± 0.14 | 90.36 ± 0.15 |
| trained | backward | 1 | 87.60 ± 1.40 | 90.41 ± 0.08 | 90.56 ± 0.09 | 90.61 ± 0.09 | 90.52 ± 0.09 |
| trained | backward | 3 | 87.60 ± 1.40 | 90.37 ± 0.18 | 90.42 ± 0.14 | 90.41 ± 0.18 | 90.39 ± 0.18 |
| untrained | forward | 1 | 10.00 ± 0.00 | 86.99 ± 0.24 | 87.45 ± 0.07 | 87.16 ± 0.16 | 86.20 ± 0.20 |
| untrained | forward | 3 | 10.00 ± 0.00 | 86.59 ± 0.27 | 87.09 ± 0.15 | 87.00 ± 0.28 | 85.18 ± 1.14 |
| untrained | backward | 1 | 10.00 ± 0.00 | 83.22 ± 1.30 | 84.66 ± 0.31 | 83.27 ± 0.95 | 83.35 ± 0.18 |
| untrained | backward | 3 | 10.00 ± 0.00 | 81.41 ± 1.88 | 85.26 ± 0.31 | 83.40 ± 0.25 | 83.24 ± 0.99 |

- **Untrained bagging is exactly chance.** It is 10.00 on every seed, and the
  test set has 1,000 images per class. That is consistent with the ensemble
  of random networks predicting one class for every image.
- **Training only the channel reaches 81.4–87.5% on untrained backbones.**
  That is 2.9–9.0 points below the same arm on trained backbones (90.2–90.6).
- **Forward beats backward on untrained backbones**, by 1.8–5.2 points
  depending on the mixer and level. On trained backbones the two directions
  are within 0.3 points.

### R2. The experts' own path alone reaches 83–86%

The `self` control, where no information crosses between experts, reaches
86.20 and 85.18 forward and 83.35 and 83.24 backward (levels 1 and 3). That
is 1.2–2.0 points below the best mixer in each setting. Most of the gain from chance therefore
does not depend on communication between experts. It is reached by each
expert's own trained path from read point to write point, through otherwise
random layers.

### R3. On untrained backbones, mixing in the other experts adds 1.2–2.0 points with `rmsnorm_attention`

Lead over `self` on test, paired:

| backbone | direction | level | `attention` | `rmsnorm_attention` | `mean` |
|---|---|---|---|---|---|
| trained | forward | 1 | +0.01 ± 0.06 | −0.02 ± 0.08 | +0.07 ± 0.09 |
| trained | forward | 3 | −0.08 ± 0.08 | −0.21 ± 0.03 | −0.01 ± 0.05 |
| trained | backward | 1 | −0.11 ± 0.04 | +0.04 ± 0.06 | +0.10 ± 0.05 |
| trained | backward | 3 | −0.02 ± 0.09 | +0.03 ± 0.10 | +0.02 ± 0.08 |
| trained | pooled | | −0.05 ± 0.04 | −0.04 ± 0.04 | +0.05 ± 0.01 |
| untrained | forward | 1 | +0.79 ± 0.39 | +1.25 ± 0.25 | +0.97 ± 0.22 |
| untrained | forward | 3 | +1.42 ± 1.26 | +1.91 ± 1.09 | +1.83 ± 1.17 |
| untrained | backward | 1 | −0.13 ± 1.15 | +1.30 ± 0.25 | −0.08 ± 0.87 |
| untrained | backward | 3 | −1.83 ± 0.95 | +2.01 ± 0.97 | +0.15 ± 0.84 |
| untrained | pooled | | +0.06 ± 0.60 | +1.62 ± 0.26 | +0.72 ± 0.40 |

"Pooled" averages each seed over the four direction × level settings, then
takes the mean ± SE over seeds.

- **Trained backbones.** No mixer leads `self` by more than 0.1 points. This
  repeats the [QK-norm](rmsnorm-attention.md) and
  [individual-loss](individual-loss.md) studies, now at 30 epochs.
- **Untrained backbones.** `rmsnorm_attention` leads `self` in all four
  settings, +1.62 ± 0.26 pooled. `mean` leads forward but not backward.
  Plain `attention` is the least stable: its backward runs include seeds far
  below the rest (74.38 and 80.73 at level 3, 78.27 at level 1).

### R4. Communication raises the experts' own accuracy on untrained backbones

`test/expert_accuracy`:

| backbone | direction | level | `attention` | `rmsnorm_attention` | `mean` | `self` |
|---|---|---|---|---|---|---|
| trained | forward | 1 | 88.29 ± 0.49 | 88.37 ± 0.49 | 88.39 ± 0.57 | 88.11 ± 0.50 |
| trained | forward | 3 | 88.07 ± 0.62 | 88.00 ± 0.54 | 88.29 ± 0.63 | 87.90 ± 0.52 |
| trained | backward | 1 | 88.33 ± 0.51 | 88.77 ± 0.51 | 88.84 ± 0.48 | 88.30 ± 0.57 |
| trained | backward | 3 | 88.11 ± 0.64 | 88.46 ± 0.57 | 88.49 ± 0.63 | 88.07 ± 0.59 |
| untrained | forward | 1 | 81.03 ± 0.81 | 81.57 ± 0.54 | 81.44 ± 0.72 | 77.55 ± 0.71 |
| untrained | forward | 3 | 78.65 ± 1.00 | 79.38 ± 1.66 | 78.85 ± 1.97 | 74.49 ± 1.57 |
| untrained | backward | 1 | 71.21 ± 0.81 | 76.26 ± 1.67 | 74.05 ± 1.88 | 71.25 ± 0.69 |
| untrained | backward | 3 | 67.81 ± 2.53 | 72.74 ± 0.62 | 70.20 ± 1.72 | 71.96 ± 2.10 |

On untrained backbones, forward, each expert's own answer is 3.5–4.9 points
better when it hears the others than under `self`. The ensemble's lead (R3) is
smaller than that. On trained backbones the experts' own accuracy moves by
at most 0.6 points between mixers.

### R5. The injection is 20–65× larger on untrained backbones

`msg_norm/injection` on `joint_val`:

| backbone | direction | level | `attention` | `rmsnorm_attention` | `mean` | `self` |
|---|---|---|---|---|---|---|
| trained | forward | 1 | 20.7 ± 1.9 | 17.6 ± 2.4 | 19.8 ± 3.5 | 21.6 ± 2.5 |
| trained | forward | 3 | 19.5 ± 1.7 | 17.5 ± 2.9 | 19.1 ± 2.0 | 20.6 ± 1.3 |
| trained | backward | 1 | 25.5 ± 4.9 | 30.2 ± 2.0 | 29.1 ± 3.5 | 27.9 ± 3.4 |
| trained | backward | 3 | 20.5 ± 3.5 | 24.2 ± 2.4 | 23.3 ± 1.3 | 21.2 ± 2.0 |
| untrained | forward | 1 | 441 ± 14 | 468 ± 19 | 419 ± 9 | 591 ± 13 |
| untrained | forward | 3 | 462 ± 23 | 458 ± 18 | 418 ± 15 | 562 ± 26 |
| untrained | backward | 1 | 1,054 ± 37 | 1,434 ± 115 | 1,080 ± 54 | 1,359 ± 53 |
| untrained | backward | 3 | 1,310 ± 78 | 1,580 ± 108 | 1,174 ± 59 | 1,341 ± 65 |

On untrained backbones the channel writes about 20–30× more forward, and
about 35–65× more backward, than it does on trained backbones. The size of
the backbone's own features at the write point is not logged, so how the
injection compares with them is not measured (see [Next](#next)).

### R6. The untrained runs had not converged at 30 epochs

| backbone | best `joint_val` epoch, all 80 runs | `joint_val` accuracy, epoch 20 → 30 |
|---|---|---|
| trained | 6–30 | within ±0.25 points |
| untrained | 20–30, 74 of 80 runs at epoch 26 or later | still rising: +0.4 to +2.4 points |

The full per-arm curves are in [Details](#details). On untrained backbones
the run score is read at the end of training, while accuracy is still
rising, so R1–R4 understate what each arm would reach with longer training.

**Overfitting** (`joint` − `joint_val` accuracy at the picked epochs) is
+1.8 to +3.5 points on untrained backbones and +1.7 to +4.0 on trained ones.
It is not specific to either backbone.

## Interpretation

This section is a hypothesis that is consistent with R1–R6. It has not been
tested.

**H1. On untrained backbones the channel is itself the classifier, and the
random layers after the write point serve as a fixed decoder.**

The forward model, with the backbone untrained, is

```
image → stage1 (random) → channel: 3,136 → 8 readings → 8 patterns (trained) → stage3, head (random)
                        ↘ stage2 (random) ↗ added to
```

- **The channel sees the image.** The encoder reads all of `stage1`, a
  random conv + ReLU + max-pool map of 3,136 numbers computed from the 784
  pixels. Random convolutional features keep much of the information in an
  image, so a trained readout of them is a classifier in its own right. For
  reference, a linear classifier on raw Fashion-MNIST pixels scores about 84%
  (logistic regression in the Fashion-MNIST benchmark, Xiao et al. 2017).
  That number is external, not measured in this testbed.
- **The channel is not small.** 117k trained weights (1.19 experts) on 24k
  images for 30 epochs.
- **The random tail can be steered.** The decoder's patterns are learned, and
  R5 shows they are written at a scale far above the trained-backbone case.
  The hypothesis is that the injection dominates the random features at the
  write point, so the frozen `stage3` and head act as a fixed nonlinear map
  from the channel's 8-number code to the logits. Training places each class's
  code where that fixed map scores the class highest. A precedent for few
  trained parameters steering a random network: Frankle, Schwab & Morcos
  (2021) train only BatchNorm's scale and shift in randomly initialised
  ResNets and reach far above chance.

What H1 accounts for:

- **R1 and R2.** Most of the accuracy needs no communication, because each
  expert's own path is the whole classifier.
- **Forward beats backward.** Backward reads at `stage2`, after two random
  conv stages, and writes back through a deeper random tail (`stage2`,
  `stage3`, head).
- **R3 and R4.** Under `self`, each expert's answer depends on its own 8
  readings. A mixer lets it depend on the readings of all three experts, up to
  24 numbers. That matters when the channel is the classifier and not when the
  backbones already classify. This part of H1 is the least supported: it
  predicts that the lead over `self` shrinks as r grows.

What H1 predicts, and would be refuted by:

| prediction | refuted if |
|---|---|
| the injection is much larger than the backbone's own features at the write point on untrained backbones, and not on trained ones | the two are comparable on untrained backbones |
| removing the backbone's own features at the write point at evaluation, keeping only the injection, leaves untrained accuracy almost unchanged and collapses trained accuracy | untrained accuracy drops substantially |
| the same 8-wide bottleneck on the read point's features, with a trained head in place of the random tail, scores at least as well as `self` | it scores well below `self` |
| the lead over `self` on untrained backbones shrinks at r = 32 | it stays or grows |

## Caveats

- **Not converged (R6).** The untrained accuracies, and the gaps between
  mixers on untrained backbones, may change with longer training.
- **n = 5.** On untrained backbones single seeds sit 4–9 points below the
  rest of their arm (e.g. `self` forward level 3, seed 2: 80.66; `attention`
  backward level 3, seed 1: 74.38). Only the `rmsnorm_attention` lead over
  `self` (R3) is resolved at this n. 24 paired comparisons at this n will
  include nominal hits by chance.
- **One expert recipe.** All results are for `cnn3` experts with class skew
  α = 1, M = 3, r = 8, on Fashion-MNIST.
- **"Untrained" means random initialisation.** It does not mean experts
  trained on unrelated data, or experts with zeroed weights.

## Next

Proposed, not yet built. Each is a tracked arm or metric in this study.

1. **Log the backbone's own feature norm at the write point**, next to
   `msg_norm/injection`, so R5 can be read as a ratio (H1, first
   prediction).
2. **Backbone knockout at evaluation.** Score each trained channel with the
   backbone's own features at the write point removed (H1, second
   prediction).
3. **Bottleneck-classifier baseline.** The `self` channel's encoder and
   decoder width (r = 8) on the same read point, with a trained head in place
   of the frozen random tail (H1, third prediction).
4. **Longer untrained runs** (60–100 epochs), until the best `joint_val`
   epoch falls inside the run (R6).
5. **r = 32 on untrained backbones** (H1, fourth prediction).

## Details

- **MLflow.** Experiment `v3-untrained`, tracking URI
  `sqlite:///<repo root>/mlflow.db`.
- **Tags.** `study=untrained-backbones`, `backbone` (`trained`,
  `untrained`), `direction` (`forward`, `backward`), `level` (`1`, `3`),
  `mixer`, `trial_seed`, and
  `arm` = `{backbone}-{fwd,bwd}-{level}-{mixer}`. The run name is
  `{arm}-trial{seed}`.
- **Params.** The two backbones differ only in `experts.budget.epochs` (10 or
  0) and so in `pool.key`. Trained pools are the ones the QK-norm and
  individual-loss studies used.
- **Exchange metrics** are logged on `joint_val` per site: `ex0` forward,
  `ex0/round0` backward, e.g. `msg_norm/injection/ex0/round0`.
- **Run:** `uv run python -m studies.untrained_backbones.run` (GPU 1 by
  default). It runs all 32 arms in one process.
- **Tables:** `uv run python -m studies.untrained_backbones.results`. It
  prints every table on this page from `mlflow.db`, plus the per-arm
  convergence curves and the per-seed accuracies quoted in
  [Caveats](#caveats).
