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

## Next levels

Not built. They come once the four levels have been read:

- a decoder that also reads the receiver's own features, so a message can
  scale what the receiver computes, not only add to it;
- more than one message per expert;
- backward exchanges with more than one round, where the open question of
  what a query means without recurrence comes in.
