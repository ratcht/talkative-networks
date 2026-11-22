# Study: experts from different datasets

**Question.** When each expert is trained on a different digit dataset, does
a trained channel between them classify a dataset none of them saw better
than the same experts without communication?

Report: [digits-five.md](../../docs/studies/digits-five.md). Plan of the second version: [digits-five-v2.md](../../docs/studies/digits-five-v2.md).

> **Running now (started 2026-11-16).** One job runs every arm below:
>
> ```
> nohup uv run python -m studies.multiple_datasets.run > studies/multiple_datasets/run.log 2>&1 &
> ```
>
> 39 arms on trial seeds 0–4, 195 runs, one process per arm. In the morning:
>
> | check | command |
> |---|---|
> | arms that crashed | `grep FAILED studies/multiple_datasets/run.log` |
> | one summary line per arm | `grep "mean over" studies/multiple_datasets/run.log` |
> | everything else | MLflow, tags `study=multiple-datasets` and `pass=2` |
>
> Not in this job: the feature-combiner baseline (not built; the soft-gated
> MoE stands in), Synthetic Digits, and masking the home expert.

## Background

Until now every expert was trained on the same dataset and the experts
differed by seed or class mix. Here each expert knows one dataset, and the
ensemble is tested on a fifth that none of them was trained on.

### The datasets

| role | dataset | images | native shape |
|---|---|---|---|
| expert 0 | MNIST | clean handwritten digits | 1×28×28 |
| expert 1 | SVHN | house numbers in street photographs | 3×32×32 |
| expert 2 | USPS | scanned postal digits | 1×16×16 |
| target | MNIST-M | MNIST digits over patches of colour photographs | 3×32×32 |

Every image is brought to 3×32×32: MNIST and USPS are resized and their grey
channel repeated three times. One mean and standard deviation, taken from the
experts' training images, normalise everything, MNIST-M included.

Each dataset's training set is split per class into three parts that share no
image, and its official test set is kept apart:

| part | share | used for |
|---|---|---|
| expert | 50% | training that dataset's expert |
| joint | 40% | training the channel |
| joint_val | 10% | picking the epochs a run is read at |

### The experts

Expert *i* is trained only on dataset *i* (`--scheme per_dataset`), on 3,600
images drawn at random from that dataset's expert part
(`--expert-samples 3600`). USPS has 3,646, so 3,600 gives every expert the
same number of images, epochs and steps. The experts are then frozen.

### The grid

| axis | values |
|---|---|
| channel's training images | target: 8,700 from MNIST-M; source: 2,900 each from MNIST, SVHN and USPS |
| loss | joint (`--loss ensemble`), individual (`--loss per_expert`) |
| direction | forward (`stage1 → stage2`), backward (`stage2 → stage1`, one round) |
| mixer | `attention`, `rmsnorm_attention`, `self`, `mean` |
| trial seeds | 0–4, so n = 5 per arm |

The encoder and decoder are `linear_taper_gelu` in every arm. In both
settings the channel trains on 8,700 images, joint_val is MNIST-M's, and the
test set is MNIST-M's. In the source setting the channel never trains on an
MNIST-M image.

| mixer | what each receiver hears |
|---|---|
| `attention` | a weighted average of every expert's message, weighted by attention |
| `rmsnorm_attention` | the same, with the query and key normalised first |
| `self` | only its own message: the control with no communication |
| `mean` | the plain average of every expert's message |

### The references

| arm | what it is |
|---|---|
| `experts-on-{mnist_m, mnist, svhn, usps}` | the three experts, no channel, tested on each dataset |
| `pooled-{arch}-experts-data-{epochs}ep` | one model trained on the 10,800 images the three experts were trained on |
| `pooled-{arch}-{target, source}-data-{epochs}ep` | one model trained on those and the 8,700 images that setting's channel is trained on |
| `{target, source}-moe-{read}-w{width}` | the soft-gated MoE: a gate weighs the frozen experts' answers |

The pooled model is a `cnn3`, the size of one expert, or a `cnn3_large`,
`cnn3` at twice the width and about the size of the three experts and their
channel. It trains for 10 or 30 epochs. The gate reads at `stage1`, `stage2`
or `stage3` with width 8 or 32, and the largest has more trained weights than
the channel. Of each, the one to report is the best on joint_val. Every run
logs `weights.experts` and `weights.trained`.

Bagging, the plain average of the frozen experts, is step 0 of every run.

### Data sizes

Both sizes are caps per dataset, set at the top of
`studies/multiple_datasets/run.py`. A dataset with fewer images than the cap
gives all it has, and each run prints how many images every dataset gave.

| constant | value | what it caps | what limits it |
|---|---|---|---|
| `EXPERT_SAMPLES` | 3,600 | images each expert trains on | USPS has 3,646 in its expert part |
| `JOINT_SAMPLES` | 2,900 | images each source dataset gives the channel | USPS has 2,916 in its joint part |

The target setting gives the channel `JOINT_SAMPLES` × 3 MNIST-M images, so
both settings train it on the same total, 8,700. `EPOCHS`, also there, is the
number of joint epochs, 30.

Why these values. Each expert is its own model, so experts may differ in how
much data they have; 3,600 happens to make them equal. The channel is one
model trained on a mix of datasets, so an uneven mix tilts it toward the
larger ones; 2,900 is the most that keeps the three sources at a third each. Raising
`JOINT_SAMPLES` above 2,916 makes USPS the smallest share, and raising
`EXPERT_SAMPLES` above 3,646 gives the USPS expert less data than the others.

## Hypotheses

- **H1, target.** Trained on MNIST-M images, a channel that communicates
  beats `self` on MNIST-M.
- **H2, source.** Trained only on source images, the channel still raises
  accuracy on MNIST-M over bagging, and a channel that communicates beats
  `self`.
- **H3, loss.** In the source setting the individual loss gives a larger lead
  over `self` than the joint loss. On every source image one expert was
  trained on that image's dataset; the joint loss lets the average rest on
  that expert, and the individual loss asks every expert to be right.

A mixer's **lead over `self`** is its accuracy minus the accuracy of `self`
with the same training images, loss, direction and seed.

## Results

Not filled in yet. Accuracy points (×100), mean ± SE over the 5 trial seeds.

One trial seed on the laptop before the job, for orientation only: the three
per-dataset experts averaged 41.3 on MNIST-M, and the `pooled` model scored
53.5.

### The experts alone

Each expert's own accuracy at step 0 (`test/expert_accuracy/expert{i}`), and
bagging (`test/accuracy`), from the four `experts-on-*` arms:

| | MNIST-M | MNIST | SVHN | USPS |
|---|---|---|---|---|
| MNIST expert | | | | |
| SVHN expert | | | | |
| USPS expert | | | | |
| bagging | | | | |

Accuracy on MNIST-M:

| arm | accuracy |
|---|---|
| bagging, per-dataset experts | |
| `pooled` | |

### H1 and H2: the channel on MNIST-M

Test accuracy on MNIST-M:

| setting | loss | direction | attention | rmsnorm_attention | self | mean |
|---|---|---|---|---|---|---|
| target | joint | fwd | | | | |
| target | joint | bwd | | | | |
| target | individual | fwd | | | | |
| target | individual | bwd | | | | |
| source | joint | fwd | | | | |
| source | joint | bwd | | | | |
| source | individual | fwd | | | | |
| source | individual | bwd | | | | |

| setting | MoE |
|---|---|
| target | |
| source | |

Lead over `self`, paired by seed, pooled over direction:

| setting | loss | attention | rmsnorm_attention | mean |
|---|---|---|---|---|
| target | joint | | | |
| target | individual | | | |
| source | joint | | | |
| source | individual | | | |

### H3: individual − joint in the source setting

| mixer | joint | individual | individual − joint, paired |
|---|---|---|---|
| attention | | | |
| rmsnorm_attention | | | |
| self | | | |
| mean | | | |

### Each expert after communication

`test/expert_accuracy/expert{i}` on MNIST-M at the picked epochs, to see
which experts the channel improves:

| setting | loss | MNIST expert | SVHN expert | USPS expert |
|---|---|---|---|---|
| target | joint | | | |
| target | individual | | | |
| source | joint | | | |
| source | individual | | | |

## Caveats

- In the source setting the epochs a run is read at are picked on MNIST-M's
  joint_val, so the summary number uses target labels. The test curves do not.
- The home expert is not masked. In the source setting the channel can do
  well by copying the expert that knows the image's dataset.
- The baseline is the soft-gated MoE, not the feature combiner of the plan.
- MNIST-M is built from MNIST digits, so the MNIST expert has seen its shapes.
- Without Synthetic Digits, SVHN is the only expert trained on colour images.
- USPS is upscaled from 16×16, and 3,600 images is a tenth of SVHN's expert
  part.
- n = 5.
- A first pass at 10 joint epochs and 2,000 images per source dataset was
  stopped: `self` was still improving at its last epoch. Those runs are in
  MLflow without a `pass` tag; this pass is tagged `pass=2`.

## Next

- The feature-combiner baseline, in both settings.
- Masking the home expert in the source setting.
- Synthetic Digits as a fourth expert.
- Holding out each dataset in turn as the target.

## Details

- Arms are built in `studies/multiple_datasets/run.py`. Arm names:
  `experts-on-{dataset}`, `pooled-{arch}-{trained_on}-{epochs}ep`,
  `{setting}-moe-{read}-w{width}`,
  `{setting}-{loss}-{fwd,bwd}-{mixer}`. Run name `{arm}-trial{seed}`.
- Tags: `study=multiple-datasets`, `pass=3`, and on the channel arms `setting`
  (`target`, `source`), `loss`, `dir`, `mixer`.
- Params: `data.expert`, `data.joint` and `data.test` are the datasets of each
  stage; `data.expert_samples` and `data.joint_samples` are the caps;
  `expert{i}.dataset` is expert *i*'s dataset.
- The `experts-on-*` arms train the same experts four times. Identical
  `pool.digest` across them confirms the weights are the same.
- Fixed: `cnn3` experts, M = 3, 10 expert epochs, 30 joint epochs, batch 128,
  split seed 0, the testbed's default channel sizes.
- Run score (`testbed/results.py`): the median over the 5 trained epochs
  around the best joint_val epoch.
- Training is deterministic on the GPU (`torch.backends.cudnn.deterministic`).
