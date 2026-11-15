# Study: the channel ladder

How much does each step up in a channel's complexity add, in both directions,
when the channel makes no spatial assumptions? The ladder starts from a
channel that is linear apart from the attention softmax, and adds one GELU at
a time. The parts are defined in [Channel](../testbed/channel.md). The study's
code is `studies/channel_ladder/`, and its decisions, with what is still open,
are in the [docket](../docket.md).

## Architecture

The study's architectural choices are in `studies/channel_ladder/__init__.py`.

### Levels

| level | encoder | decoder | adds |
|---|---|---|---|
| 0 | — | — | nothing: no exchange, so the ensemble is bagging |
| `1` | `linear_taper` | `linear_taper` | a linear message and a linear injection; the attention softmax is the only nonlinearity |
| `2-encoder` | `linear_taper_gelu` | `linear_taper` | a nonlinear message: a GELU in the encoder |
| `2-decoder` | `linear_taper` | `linear_taper_gelu` | a nonlinear injection: a GELU in the decoder |
| `3` | `linear_taper_gelu` | `linear_taper_gelu` | both: a GELU in the encoder and one in the decoder |

- **No level pools or broadcasts.** The linear taper parts read and write the
  whole feature map.
- **All levels have exactly the same weights** ([Channel](../testbed/channel.md)
  §3 and §5), so a difference between levels is the GELUs', not a difference
  in size. The two versions of level 2 show which side a nonlinearity helps on.
- **Both directions use the same levels.** A backward exchange runs one
  round; more rounds are a separate axis.
- **Level 0 needs no runs of its own.** Every run's step 0 is bagging.

### Mixers

Any level runs with any of the study's three mixers
([Channel](../testbed/channel.md) §2 and §4): `attention`, the channel's own,
and its two controls, `self` and `mean`. Each control keeps every other part
of the level, so a level minus its control is what communication adds. Phase 4
also runs `cosine_attention` ([Channel](../testbed/channel.md) §8), which isn't
one of the three: other studies take their arms from them.

### The MoE baseline

The soft-gated MoE with the `linear_taper_gelu` gate
([Soft-gated MoE](../testbed/moe.md)) weighs the experts' answers instead of
letting them communicate. Where it reads is an ablation, like the channel's
exchange points.

### Width

Every part reads or writes through $r = 8$: the linear taper encoders'
readings, the decoders' patterns, and the gate's readings of each expert.

## The experiment

Decided 2026-10-25 unless marked proposed.

**What counts as a result.** Two comparisons per level, each made trial seed
by trial seed and then averaged:

- the level minus its `self` control: what communication adds over each
  expert's own add-on;
- the level minus the MoE: whether communicating beats weighing the experts.

Anything else read off the runs is exploratory. The MoE isn't matched to the
channel in size; it is meant to be a sensible baseline, not an equal-weight one.

**Protocol.** 15 trial seeds (0–14, the runner's default) for every arm, 30
joint epochs, learning rate 1e-3, the joint loss. Each run is read as the
median test accuracy over the 5 epochs around its best joint_val epoch
(`testbed/results.py`). Every run records the git commit it ran from.

**Phase 1: where to attach.** Level 1 and its `self` control at every exchange
pair, and the MoE at every read point:

| direction | exchange pairs |
|---|---|
| forward | stage1 → stage2, stage2 → stage3, stage1 → stage3 |
| backward | stage2 → stage1, stage3 → stage2, stage3 → stage1 |

That is 12 channel arms and 3 MoE arms. Each direction gets its own pair; the
two needn't mirror each other. The rule, fixed before any phase 1 run (user,
2026-10-26):

- each direction's pair is the one with the largest mean of level 1 minus
  `self`;
- the MoE's read point is the one with its own highest mean, so the baseline
  is at its best.

After phase 1, check again whether 30 epochs is long enough.

Phase 1 ran on 2026-10-26, and its runs are the MLflow experiment
`channel-ladder-phase1`. It picked forward stage2 → stage3, backward
stage3 → stage1 and the MoE at stage3, and kept 30 epochs. Its results are in
the [docket](../docket.md).

**Phase 2: the ladder.** All four levels with `attention`, `self` and `mean`
at each direction's chosen pair, and the MoE at its chosen point. The level 1
and `self` arms there and the MoE were already run in phase 1, so 20 channel
arms are new. They are split into three phases, each its own experiment and
run in this order (user, 2026-10-28):

| phase | arms | question |
|---|---|---|
| 2a | forward stage2 → stage3: levels `2-encoder`, `2-decoder` and `3`, each with `attention` and `self` | both results for every level, forward |
| 2b | the same 6 arms, backward stage3 → stage1 | both results for every level, backward |
| 2c | `mean` at all four levels, at both pairs | exploratory: `mean` − `self` is what hearing the others adds, `attention` − `mean` what choosing whom to listen to adds |

Each phase's comparisons take level 1, its `self` control and the MoE from
`channel-ladder-phase1`.

**Phase 3: the width.** Phase 2's GELU levels added nothing at $r = 8$, so
by the r = 32 rule they are rerun wider, at $r = 16$, 24 and 32 (user,
2026-10-28), in that order, each its own phase: `3-r16`, `3-r24` and `3-r32`.
At each width, levels 1 and 3 with `attention`, `self` and `mean`, at both
chosen pairs: 12 arms per width. The MoE stayed phase 1's, at $r = 8$, as
planned; on 2026-11-02 it was added at each width too (`moe-stage3-r16`, `-r24`,
`-r32`), so level − MoE can also be read at equal $r$.

- Level 3 has both GELUs, so if it doesn't beat level 1 at some width, the
  two versions of level 2 have nothing to explain there. If it does, they
  run at that width.
- `self` widens with the level, so level − `self` stays a comparison at equal
  size. Level − MoE does not: at $r = 32$ the channel is up to 9× the gate.
- Arms are labelled with their width, e.g. `level3-self-stage3-stage1-r24`.

**Phase 4: the size of the queries and keys.** Forward attention is a fixed
wire at the linear encoders and softer at the GELU ones, and across phase 2's 8
attention arms softness follows message size. But message size only stands in
for the queries and keys, which no run has logged. Phase 4 takes size away with
`cosine_attention`, which reads only their directions
([Channel](../testbed/channel.md) §8), and logs their sizes and who each
receiver hears. Decided 2026-11-01, before any phase 4 run.

All arms are forward stage2 → stage3, at $r = 8$, with the protocol above, on
the lab machine. `self` and `mean` read no query or key, so they stay
phases 1, 2a and 2c's.

| arm | level | mixer | why |
|---|---|---|---|
| `level1-cosine_attention-stage2-stage3` | 1 | `cosine_attention` | level 1 with size taken away |
| `level2-encoder-cosine_attention-stage2-stage3` | `2-encoder` | `cosine_attention` | the GELU with size taken away |
| `level1-stage2-stage3` | 1 | `attention` | the reference for both of the above |
| `level2-decoder-stage2-stage3` | `2-decoder` | `attention` | a linear encoder with a GELU after it |
| `level2-encoder-stage2-stage3` | `2-encoder` | `attention` | the softer GELU arm |
| `level3-stage2-stage3` | 3 | `attention` | the other softer GELU arm |

The four `attention` arms repeat phases 1 and 2a, for two reasons. Those runs
didn't log queries, keys or who each receiver hears. And training again on
another GPU isn't identical, so a difference between arms should come from one
machine. The repeat of level 1 against phase 1, trial seed by trial seed, also
shows how much the machine moves the results.

The rule, fixed before any run. Each run is read by the study's rule, and
differences are paired by trial seed, with 95% ranges and the trial seeds
ahead. Softness is `attn_entropy/ex0` and `routing_mode_share/joint/ex0`, read
together.

1. **Is size what differs?** `msg_norm/query/ex0` and `msg_norm/key/ex0` across
   the four `attention` arms. If the linear encoders don't send larger queries
   and keys than the GELU ones, size isn't the explanation, whatever else
   happens.
2. **Does taking size away soften level 1?** Level 1 with `cosine_attention`
   minus level 1 with `attention`, in both softness readings.
3. **Does it reach the GELU arms?** It does if level 1 with `cosine_attention`
   has an entropy at or above the lower of `2-encoder` and 3, and a mode share
   at or below the higher of the two.
4. **Does accuracy hold?** Level 1 with `cosine_attention` minus level 1 with
   `attention`, in `test/accuracy`: it holds if the 95% range covers 0 and the
   mean is within ±0.1, the size 15 trial seeds can pin. Any effect near 0.1 is
   also read on `joint_val/accuracy` at epochs 21–30, since every trial seed
   shares one test set.
5. **Is anything left besides size?** `2-encoder` minus level 1, both with
   `cosine_attention`, in both softness readings. Size is gone from both arms,
   so this doesn't depend on the length the mixer rescales to, while 3 does.

If 1 to 4 all hold, the GELU's softer routing is scale, as the docket proposed.
5 says whether that is the whole story: a difference left there is the GELU
changing the directions as well. Who each receiver hears is read alongside,
as description.

Phase 4 ran on 2026-11-01, and its runs are the MLflow experiment
`channel-ladder-phase4`. 1 to 4 all held: level 1's queries and keys are 4–5×
the GELU arms', rescaling them took entropy from 0.006 to 0.82 and mode share
from 0.88 to 0.21, past the GELU arms, and test accuracy moved by +0.01 ± 0.10.
On 5, a small difference is left. Softer routing isn't better routing: without
size, training leaves routing about as sharp and as consistent as the untrained
channel's, the sender heard is no more often right, and the deficit to `self`
stays. The results are in the [docket](../docket.md).

**Phase 5: why queries and keys grow.** Phase 4 tied the fixed wire to the size
of the queries and keys, and left open why they grow. Its logs and a pilot on
one trial seed showed that the query and key heads barely change: what grows is
the readings all three heads share. Two things need explaining. Decided
2026-11-02, before any phase 5 run.

- **The jump.** In the first epoch the readings grow several times over in
  every arm, `self` included.
- **The later growth.** After epoch 1 sizes level off or fall, except in
  senders nobody hears under `attention` at the linear encoders.

Two ideas are on trial. Stage 2's features are all at least 0, with a large
average image, so a small change in the read weights can move every reading on
every image: that would be the jump. And two gradients reach the readings,
through the value and through the query and key. In the pilot the second alone
pushed them furthest, so the first may act as a brake, which a sender nobody
hears doesn't get.

All arms are level 1, forward stage2 → stage3, at $r = 8$, with the protocol
above, on the lab machine. `studies/channel_ladder/phase5.py` holds them. The
changes are the phase's own, not channel parts.

| arm | mixer | change | question |
|---|---|---|---|
| `level1-self-centred-stage2-stage3` | `self` | each expert's average stage 2 features on the joint split are subtracted before reading | is the average image behind the jump? |
| `level1-centred-stage2-stage3` | `attention` | the same | the same with routing, and does the wire go? |
| `level1-frozen-readings-stage2-stage3` | `attention` | the read layer never trains | is the growth needed? |
| `level1-routing-cut-stage2-stage3` | `attention` | the query and key heads read the readings without passing gradient back | what pushes unheard senders? |
| `level1-cosine_attention-value-cut-stage2-stage3` | `cosine_attention` | the same for the value head, expert 0 only | is the value gradient a brake? |
| `level1-cosine_attention-value-cut-no-bias-stage2-stage3` | `cosine_attention` | that, and no query or key head has a bias | is the bias routing's way of reaching the readings' size? |
| `level1-self-stage2-stage3` | `self` | none | reference |
| `level1-stage2-stage3` | `attention` | none | reference |
| `level1-cosine_attention-stage2-stage3` | `cosine_attention` | none | reference |

The references repeat earlier arms because reading size and weight size are new
measurements. `attention`'s repeat also checks that the new logging leaves
training alone: on the same machine it should match phase 4's run.

The rule, fixed before any run. Sizes are read at fixed epochs, per run, from
`msg_norm/reading/ex0` and `msg_norm/reading/expert{i}/ex0`. The jump is a run's
reading size at epoch 1 divided by its own at step 0. How much a sender is heard
is the mean over receivers of its `attn_heard` shares.

1. **Is the average image behind the jump?** Yes if either centred arm takes the
   jump below ×2 on at least 13 of 15 trial seeds. Absolute size is read too,
   since centring also shrinks step 0.
2. **Does the wire go with it?** At `level1-centred`, yes if `attn_entropy/ex0`
   is at least 0.5 by the study's rule, the level of the softer GELU arms.
3. **Is the growth needed?** Not if `level1-frozen-readings` minus the
   `attention` reference has a 95% range covering 0 and a mean within ±0.1 in
   `test/accuracy`, read by the study's rule and also on `joint_val/accuracy`
   at epochs 21–30. Its entropy should be at least 0.5 as well.
4. **Does routing push unheard senders?** In each run take the sender heard
   least at epoch 30. Yes if at `level1-routing-cut` its reading size at epoch
   30 over epoch 1 has a median of at most ×1.5, with the reference above that.
5. **Is the value gradient a brake?** At
   `level1-cosine_attention-value-cut`, yes if expert 0 ends epoch 30 at least
   2× the larger of its two partners on at least 13 of 15 trial seeds, while
   still heard, at a share of at least 5%.
6. **Is the bias the way in?** Yes if 5 holds and the same test fails at
   `level1-cosine_attention-value-cut-no-bias`. If the runaway survives without
   biases, routing's gradient is lengthening the readings some other way.

The jump is isolated if 1 holds, and the later growth if 4 and 5 both hold.

Phase 5 ran on 2026-11-02, and its runs are the MLflow experiment
`channel-ladder-phase5`. Neither is isolated. 1 failed: the jump stayed at ×5.3
and ×7.0, so the average image isn't behind it. 2 failed: centred `attention`
kept an entropy of 0.24, though its mode share fell from 0.88 to 0.25. 3 failed:
frozen readings made no wire but cost 0.21, so the read layer has to learn. 4
held, at ×0.85 against ×3.23, though the cut also stops the wire, so no sender
goes unheard. 5 failed narrowly, on 10 of 15 trial seeds. For 6, removing the
biases changed nothing, so they aren't the mechanism. The results are in the
[docket](../docket.md).

**Cost on the laptop**, one run at a time: about 35 s per forward run and 67 s
per backward run. Phase 1 took about 3 hours; 2a is about 1 hour, 2b about 2
and 2c about 2. If it drags, it moves to the lab machine, one run per GPU: phase
4's 90 forward runs took 14 minutes there, at about 19 s per run on each of two
GPUs. Runs made there come back into the laptop's `mlflow.db` through
`testbed/merge_runs.py`.

## Running an arm

`studies/channel_ladder/run.py` runs one arm over a list of trial seeds. An
arm is one of three kinds:

| kind | flags | tagged |
|---|---|---|
| a level | `--level L --exchange READ WRITE`, and `--mixer self` or `--mixer mean` for a control | `level` |
| the MoE baseline | `--moe READ`, where its gate reads | `baseline=moe` |
| bagging | `--bagging` | `baseline=bagging` |

```bash
uv run python -m studies.channel_ladder.run --phase 2 --level 2-encoder \
  --arm forward-2e --exchange stage2 stage3 --trial-seeds 0 1 2
uv run python -m studies.channel_ladder.run --phase 2 --moe stage3 \
  --arm moe-stage3 --trial-seeds 0 1 2
```

Every run belongs to a phase, which `--phase` names (`1`, `2a`, ...). Each phase
is its own MLflow experiment, `channel-ladder-phase<PHASE>`, so no one
experiment holds every arm the study has run. Comparing across phases, for
instance a phase 2 level against phase 1's MoE, means selecting both
experiments in the MLflow UI, or searching both in a script.

The runner hands `testbed.run` the study's choices: the level's parts or the
study's gate, the mixer (`attention` unless a control is named) and the
study's width. It also sets the study's default of 30 joint epochs. Every
other flag is `testbed.run`'s, and a flag given here overrides a default. It
refuses three things:

- flags the arm sets: `--encoder`, `--decoder`, `--gate`, `--gate-read`, or an
  `--exchange` for a baseline;
- `--experiment`, which the phase sets;
- more than one kind of arm at a time.

Every run is tagged in MLflow with `study` (`channel-ladder`), its `phase`,
and its `level` or `baseline`. These come on top of `testbed.run`'s own tags: `arm`,
`trial_seed`, `direction`, and `gate` for the MoE. Each run also logs its
parts by name, so it stays readable even if the levels are ever renumbered.
When grouping runs:

- group on the width and the mixer as well, which are params;
- runs started through `testbed.run` directly carry no study tags.

## Next levels

Not built. They come once the four levels have been read:

- a decoder that also reads the receiver's own features, so a message can
  scale what the receiver computes, not only add to it;
- more than one message per expert;
- backward exchanges with more than one round, where the open question of
  what a query means without recurrence comes in.
