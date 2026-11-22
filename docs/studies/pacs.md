# Study: experts from PACS domains

**Question.** When each expert is trained on one PACS domain, does a channel
between them, trained on images of those same domains, classify the fourth
domain better than the same experts without communication?

## Background

In the two digit experiments ([first](digits-five.md),
[second](digits-five-v2.md)) no mixer in which the experts hear each other
was more accurate than the `self` mixer. This study changes three things. The
task is object recognition across four visual styles, which differ more than
the digit datasets did. The channel is trained on the experts' domains only,
so nothing is trained on the domain the ensemble is tested on. The expert is
a network with a residual stream of one width, which is the quantity the
study varies: a narrow stream limits what an expert can keep, and what one
expert has dropped only another can supply.

### The dataset

PACS has 9,991 images of 7 classes (dog, elephant, giraffe, guitar, horse,
house, person), each 3×227×227, in four domains.

| domain | dog | elephant | giraffe | guitar | horse | house | person | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| art painting | 379 | 255 | 285 | 184 | 201 | 295 | 449 | 2,048 |
| cartoon | 389 | 457 | 346 | 135 | 324 | 288 | 405 | 2,344 |
| photo | 189 | 202 | 182 | 186 | 199 | 280 | 432 | 1,670 |
| sketch | 772 | 740 | 753 | 608 | 816 | 80 | 160 | 3,929 |

Every image is resized to 3×64×64. PACS has no test set, so a fifth of each
domain, drawn per class, is set aside as one. The rest is split per class
into the three parts every study uses.

| domain | expert | joint | joint_val | test |
|---|---:|---:|---:|---:|
| art painting | 820 | 655 | 163 | 410 |
| cartoon | 938 | 750 | 187 | 469 |
| photo | 668 | 534 | 134 | 334 |
| sketch | 1,571 | 1,257 | 315 | 786 |

Each domain is the target in turn, with the other three as the experts'
domains. No model is pretrained: every weight is trained on PACS images
only.

### The expert

An expert is a convolutional network whose blocks all read from and add to
one stream of constant width `d`.

| stage | layers | output |
|---|---|---|
| `embed` | convolution, kernel 4, stride 4 | d×16×16 |
| `stage1` | 2 blocks | d×16×16 |
| `stage2` | 2 blocks | d×16×16 |
| `stage3` | 2 blocks | d×16×16 |
| `head` | group norm, mean over positions, linear | 7 |

A block computes an update (group norm, 3×3 convolution, GELU, 1×1
convolution) and adds it to the stream; nothing is applied after the sum.

| architecture | `d` | weights |
|---|---:|---:|
| `stream16` | 16 | 16,679 |
| `stream32` | 32 | 64,071 |
| `stream64` | 64 | 251,015 |
| `stream128` | 128 | 993,543 |

Experts are trained with a random crop (4 pixels of padding, cropped back to
64×64) and a random horizontal flip on every batch. Joint training is not
augmented.

### How a run is read

A run is read at its best epochs on the `joint_val` images of the target
domain, as in the digit experiments. The target's labels therefore choose
which epochs are reported, for every arm alike, and train nothing.

## First job: the experts alone

Before any channel is trained, this job measures whether experts trained on
PACS alone learn their own domain, how well they do on the others, and how
both depend on the stream width and the length of training.

| item | value |
|---|---|
| scheme | `per_dataset` |
| experts | 3 |
| trial seeds | 0, 1, 2, 3, 4 |
| study tag | `pacs` |
| pass tag | 1 |

| axis | values |
|---|---|
| target | art painting, cartoon, photo, sketch |
| architecture | `stream16`, `stream32`, `stream64`, `stream128` |
| expert epochs | 30, 100 |
| test set | art painting, cartoon, photo, sketch |

That is 128 arms and 640 runs.

    uv run python -m studies.pacs.run experts

## Second job: a classifier on the experts' streams

A channel can only help if, on a given image, the other experts hold
something that one expert lacks. This job measures that without a channel. A
classifier is trained on the final stream of one expert, and another on the
final streams of all three; if the second is more accurate, the information
is there for a channel to carry.

The experts are frozen and come from the first job's cache. Each expert is
run to the output of `stage3`, its stream is averaged over positions, and the
vectors of the experts being read are concatenated.

| step | shape at `d` = 64 |
|---|---|
| output of `stage3` | 64×16×16 |
| mean over positions | 64 |
| concatenated, one expert | 64 |
| concatenated, three experts | 192 |
| classifier | 7 |

The classifier is trained for 30 epochs on the `joint` images of the three
experts' domains, the images a channel would be trained on, and is read like
any other run.

| axis | values |
|---|---|
| target | art painting, cartoon, photo, sketch |
| architecture | `stream16`, `stream64` |
| expert epochs | 100 |
| test set | art painting, cartoon, photo, sketch |
| classifier | linear; MLP with one hidden layer of 256 and a GELU |
| experts read | expert 0, expert 1, expert 2, all three |

That is 32 arms; each trial of an arm makes 8 runs, one per classifier and
set of experts read, so 1,280 runs. The runs are tagged `head` and `reads`.

    uv run python -m studies.pacs.run readout

The classifier reads the end of each expert and discards position, while a
channel exchanges earlier and the experts can still act on what they receive.
The comparison therefore shows what three experts hold together at the end,
and is not a bound on what a channel can reach.

## Not in these jobs

The pooled model, the gate and the channel.
