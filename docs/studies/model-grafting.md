# Model grafting

Can frozen experts help one another when communication is the only way they
can? In a graft ([Grafting](../testbed/grafting.md)) one expert, the anchor,
alone sees the image and gives the answer; the obligates see nothing and take
part only through the channel. With no communication the obligates contribute
nothing, so the room communication has to fill is the whole gap between the
anchor alone and what the group knows, not the small gap left over bagging in
a communicating ensemble, where each expert's own add-on already did as well
as communicating.

The code is in `studies/model_grafting/`, built on `testbed/grafting.py`.

## Arms

| arm | mixer | the obligates' blocks from $a$ to $d$ |
|---|---|---|
| `graft_attention` | `attention` (CALM) | run |
| `graft_mean` | `mean` | run |
| `bypass` | `attention` | left out: the same channel, without them |

The anchor alone is every run's step 0, so it has no arm. The comparisons,
trial seed by trial seed:

| difference | what it measures |
|---|---|
| `graft_attention` − step 0 | what the whole graft adds to the anchor alone |
| `graft_attention` − `bypass` | what the obligates' frozen blocks add over the same trained channel |
| `graft_attention` − `graft_mean` | what choosing whom and where to listen adds over averaging the obligates position by position |

Deferred, at the user's request: untrained obligates (does the obligates'
training matter, or only their computation) and clone obligates (does a
different expert matter, or only more of the anchor's own blocks).

Decided by the user on 2026-11-27: Anchor and Obligate as the names; the side
branch; one stitch into every obligate, with no gate; the stitches start from
random weights; the class token goes through both stitches; CALM's attention
over every position of every obligate, with no "nobody" option, and `mean` as
the other mixer; the anchor is fixed by the trial seed (the group's first
expert); the middle third ($a$ = `block4`, $d$ = `block8`) as the default
points until a pilot places them; the bypass as the control.

## Fitted stitches

`--stitch-fit` (in the study's runner, `studies/model_grafting/stitch_fit.py`)
sets where the stitches start:

| `--stitch-fit` | the stitches |
|---|---|
| `random` (default) | start from random weights, as decided above |
| `fitted` | start from a least-squares fit, then train as usual |
| `fitted_frozen` | stay the fit; only the return trains |

The fit runs once, before training: the anchor and each obligate run the joint
split's images up to the read point, and each stitch becomes the one affine map
(the same at every cell) that best turns the anchor's stream there into that
obligate's own (Csiszárik et al. 2021 start stitches this way). It needs the
plain stitch, so the runner sets `--graft-stitch pointwise_linear`. This is the
one place the obligates see images; their answers are never read, and in
training and evaluation they still get only the stitched anchor stream. The
user approved this departure on 2026-11-27 for these arms only, after the
first overnight run found the random-start stitches leave the obligates at
chance (`stitched_accuracy` 1.3%).

## Running an arm

    uv run python -m studies.model_grafting.run --graft-arm graft_attention --trial-seeds 0 1 2
    uv run python -m studies.model_grafting.run --graft-arm bypass --arch vit \
        --graft-read block2 --graft-write block10 --trial-seeds 0 1 2

Every other flag goes to `testbed.run` as it is. Runs go to the MLflow
experiment `model-grafting`, tagged `study=model-grafting` and
`graft_arm=<arm>`, labelled `<arm>-<read>-<write>`. Defaults are the testbed's:
class-skewed pools of three ($\alpha = 1$), 10 joint epochs, Adam at
$10^{-3}$, batch 128, no augmentation.

## Pilot: where to read and write (2026-11-27)

Where should the channel sit? The pilot runs `graft_attention` and `bypass`
for both families at every pair of points $a < d$ among `embed`, `block2`,
`block4`, `block6`, `block8`, `block10` and `block12`: 21 pairs, which cover
both how deep the graft sits and how many of the obligates' blocks it uses
($d - a$, from 2 to 12). At $a$ = `embed` the stitch hands the obligates a
linear re-encoding of the image itself, so that column is the graft closest to
an ensemble. Trial seeds 0, 1, 2, run seed by seed, so that a cut-off still
leaves whole seeds; the runs are tagged `pilot=positions`.

Each position is read as `graft_attention` − step 0 and `graft_attention` −
`bypass`, on the testbed's number per run (the median test accuracy over the
5 epochs around the best joint_val epoch, `testbed/results.py`). Three trial
seeds give a rough answer only.
