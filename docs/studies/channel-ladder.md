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
of the level, so a level minus its control is what communication adds.

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
chosen pairs: 12 arms per width. The MoE stays phase 1's, at $r = 8$.

- Level 3 has both GELUs, so if it doesn't beat level 1 at some width, the
  two versions of level 2 have nothing to explain there. If it does, they
  run at that width.
- `self` widens with the level, so level − `self` stays a comparison at equal
  size. Level − MoE does not: at $r = 32$ the channel is up to 9× the gate.
- Arms are labelled with their width, e.g. `level3-self-stage3-stage1-r24`.

**Cost on the laptop**, one run at a time: about 35 s per forward run and 67 s
per backward run. Phase 1 took about 3 hours; 2a is about 1 hour, 2b about 2
and 2c about 2. If it drags, it moves to the lab machine, one run per GPU.

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
