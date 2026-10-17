# Study 02 — Spatial Access at the Read and Write Ports

**Status.** `protocol — Part I complete, frozen 2026-10-04; runs begun` · opened 2026-09-23

Study 01 varied what the encoder *computes*. This study varies what the two
adapters are allowed to *see* and *say*.

Part I is the protocol and is frozen before the first run. Part II is the
record and is written afterwards. Nothing in Part I is edited once runs
begin — corrections go in §9.

---

# Part I — Protocol

## 1. Question & hypothesis

The channel destroys all spatial structure at the read port and can apply only
a spatially uniform correction at the write port (§3.1). Is either of those
limiting what communication achieves?

**Hypothesis.** Spatial access, i.e. the ability to read *where* information sits
and to write *where* a correction applies, improves the channel through two
mechanisms.

**Read side.** An encoder that can see spatial structure produces messages
carrying more information about the sender's state. The effects of this
improved semantic richness are expected to show up in message content and
possibly also in routing sharpness. This is because the encoder feeds query,
key and value together, so spatial access changes what is *said*, thereby
improving message content, as well as who is *listened to*, meaning a
potential improvement in message routing.

**Write side.** A spatially varying injection perturbs the frozen backbone
more usefully than a uniform channel bias does. A frozen expert's backbone
cannot modify itself to interpret the injected message more efficiently,
meaning that the quality of the injection, for which spatial access is a
prerequisite, could be a large determining factor in the expert's ability to
correct its classification.

It is important to note that the two mechanisms above may not be separable, as
messages carrying spatial information may need a decoder with spatial access
for that information to be usable.

## 2. Point of departure

This study inherits both its motivation and its measurement precision from
Study 01, which trained six arms over the same expert group, split and joint
seeds that §4.1 carries forward. The relevant findings are recorded here as they motivate some aspects of experimental design.
They were measured with each specialist's classification head trainable, which
§4.1 does not permit, so they are read here as priors on what this pipeline
produces rather than as values this study extends.

### 2.1 What Study 01 established

Study 01 varied the encoder's architecture along two axes: expressiveness, by
replacing three independent linear maps with a shared nonlinear trunk, and
channel capacity, by widening `key_dim` and `value_dim` from `k16/v64` to
`k64/v128`. A later pair of arms removed the trunk's LayerNorm to separate the
nonlinearity from the normalisation accompanying it.

| arm | encoder | widths | final test accuracy |
|---|---|---|---|
| `baseline` | `qkv` | k16/v64 | 65.87 |
| `wide` | `qkv` | k64/v128 | 65.81 |
| `mlp` | `mlp` | k16/v64 | 65.80 |
| `mlp_wide` | `mlp` | k64/v128 | 65.81 |
| `mlp_no_norm` | `mlp_no_ln` | k16/v64 | 65.94 |
| `mlp_no_norm_wide` | `mlp_no_ln` | k64/v128 | 65.98 |

Two conclusions were drawn, and a third follows from the accuracy table but was
not part of Study 01's own reading.

**Nonlinear encoders were adopted on the evidence of routing.** With three
independent linear maps, routing is degenerate. The figures below are means over
the final ten epochs across five seeds. The `wide` arm, the same linear encoder
at `k64/v128`, is included to show the effect of width alone. Differences are
expressed in standard errors of the difference (SE), the standard error being
that of a difference between two independently estimated arm means, so a value
of 2 is the threshold §4.3 adopts for calling a difference real. The final
column compares `mlp` with `baseline`.

| metric | `baseline` | `wide` | `mlp` | `mlp` against `baseline`, in SE |
|---|---|---|---|---|
| `routing_mode_share[receiver=joint]` | 0.934 | 1.000 | 0.402 | 3.9 |
| `attn_top1` | 0.988 | 1.000 | 0.832 | 4.5 |
| `attn_entropy` | 0.028 | 0.000 | 0.400 | 4.7 |
| `val_route_flip_rate` | 0.009 | 0.000 | 0.087 | 3.9 |
| `grad_norm[component=query_head]` | 0.029 | $1.3 \times 10^{-5}$ | 0.128 | 3.5 |
| `grad_norm[component=key_head]` | 0.029 | $1.0 \times 10^{-5}$ | 0.124 | 3.5 |
| `grad_norm[component=value_head]` | 0.472 | 0.558 | 0.257 | 18.1 |

A mode share approaching unity means a single wiring is used for every image, so
the channel is a constant skip connection and the question of who listens to whom
does not arise. The linear encoder sits at 0.934, with peak attention at 0.988
against the 0.333 that uniform attention over three senders would give, and a
validation route-flip rate of 0.009. Two of its five seeds are fully degenerate,
with a mode share of 1.000. Widening the encoder does not relieve this: at
`k64/v128` all four routing series are at their degenerate limits in every seed.
The trunk moves every one of these series: the modal wiring covers 40 percent of
the examples, and 8.7 percent of validation routes change from one epoch to the
next.

The last three rows give the mechanism. Under the linear encoder the query and
key heads receive little gradient, 0.029 each against the value head's 0.472,
and in the two degenerate seeds they receive approximately $10^{-6}$. At
`k64/v128` they are effectively dead in every seed, at approximately $10^{-5}$,
while the value head continues to receive 0.558. Degenerate routing was
therefore a trainability failure and not a capacity one: the heads that decide
who listens to whom were barely learning, and added width did not change this.
The shared trunk is what supplies them with gradient, raising both above 0.12.

**The trunk did not improve interpretation, and communication did not help the
ensemble.** The per-specialist correction and corruption rates Study 01 reported
moved in opposite directions on the two data sets. On training batches the trunk
widened the gap between corrected and corrupted answers from 0.034 to 0.048, a
difference of 7.8 standard errors. On validation the gap moved against the trunk
by 2.2 standard errors, corrupted answers rising while corrected answers barely
moved. Interpretation therefore improved only on the data the channel was
trained on. Recomputing its checkpoints
with the ensemble-level metrics this study introduces (§5.2) shows something
sharper. Net joint correction is **negative in every arm**: validation accuracy
after communication is 0.6622 against 0.6642 before it with linear encoders, and
0.6598 against 0.6662 with the trunk. The channel subtracted accuracy, and the
trunk subtracted more of it, with corrupted rising by 2.4 standard errors while
corrected did not move.

This is not visible in any arm-against-arm comparison, which can only see
differences between channels and not the sign of what a channel does, and it is
why §6 carries R8. Messages also became smaller rather than larger,
`msg_norm[part=value]` falling from 11.7 to 9.9, and final test loss rose from
1.415 to 1.431 while training loss fell, a mild overfitting signature. The
adoption of the trunk therefore rests on routing alone.

**Channel capacity had no effect**, which is why message width is not an axis of
this study (§3.3) and is instead held fixed (§4.1).

The third conclusion is that **no arm of Study 01 moved accuracy by a resolvable
margin.**
The six means in the accuracy table span 0.17 percentage points, and `mlp`
finished nominally below `baseline`. §2.2 shows that this spread is less than half the smallest
difference five seeds can distinguish from noise. Accuracy was therefore never
the quantity that settled Study 01, and §4.3 does not ask it to settle this
study either.

### 2.2 Measured precision of the pipeline

The six arms of §2.1, five seeds each, also measure what this pipeline can
detect. The figures below were extracted from Aim on 2026-10-01 and are the
basis of §4.3's decision rule.

| endpoint | pooled seed sd | SE of a two-arm difference | smallest resolvable difference (2 SE) |
|---|---|---|---|
| final-epoch test accuracy | 0.305 pts | 0.193 pts | 0.39 pts |
| final-epoch validation accuracy | 0.348 pts | 0.220 pts | 0.44 pts |
| best-epoch validation accuracy | 0.174 pts | 0.110 pts | 0.22 pts |

Three properties of these measurements bear on the design.

**Sharing a seed list across arms does not permit a paired comparison.** The mean
correlation of per-seed final test accuracy between arms is +0.11 over fifteen
arm pairs, with individual values from -0.90 to +0.89, which at five seeds is
indistinguishable from zero. The reason is structural: the backbones are frozen,
so a joint seed governs only adapter initialisation and training data order, and
an initialisation is not comparable between two architectures whose parameter
spaces differ. The difference of two arm means therefore carries the variance of
two independent means.

**Selecting the epoch on validation accuracy halves the seed standard
deviation,** from 0.348 to 0.174 points. The reduction is available to any study that records
accuracy per epoch rather than once at termination.

**The validation curve has no sharp optimum.** Averaged over the thirty
replicates, validation accuracy holds near 66.8 percent for the first twenty
epochs and then declines steadily, to 66.1 percent at the final epoch. The best
validation epoch has median 13 of 70, counting from 0, with a range of 0 to 36,
and final validation accuracy lies below the best in every replicate, by 1.2
points on average. Training therefore overfits gradually after an early plateau.
Within the plateau the position of the best epoch is set by noise, the best
epoch exceeding its four neighbours by 0.46 points on average, so selecting the
single best epoch would amount to selecting on noise.

## 3. Ports and arms

### 3.1 The two ports

Both defaults are unchanged from Study 01, on CIFAR ResNet-110. These shapes
are what §4.2's parameter counts and §5's diagnostics are computed from.
They are documented below as they are not derivable from the flags alone.

| port | flag | module | tensor | adapter sees / emits |
|---|---|---|---|---|
| READ | `--late layer3.0` | output of `layer3.0` | `(b, 64, 8, 8)` | reduced to `(b, 64)` by a spatial mean |
| WRITE | `--early layer1.2` | input to `layer1.2` | `(b, 16, 32, 32)` | emits `(b, 16)`, broadcast to all positions |

Wiring is `ensemble/train.py:694-695`; the injection is added in the pre-hook
at `ensemble/communicative/hooks.py:18`.

### 3.2 Ladders

The study consists of two ladders, one acting on the expert's encoder and the
other on its decoder. Both ladders share the same three steps: **no spatial
access**, **fixed spatial access**, and **conditioned spatial access**. The
encoder ladder carries a fourth level that serves only as a control.

**What the ladders hold fixed.** Every encoder arm is
`[spatial mapping] -> [MLP trunk] -> [QKV heads]`, with the trunk and
heads carried over from Study 01 unchanged. Every decoder arm is
`[MLP] -> [spatial mapping]`. Only the specified spatial mappings vary.
Every encoder level emits a 64-vector into an identical trunk, so the levels
differ only in how `(64, 8, 8)` is contracted. The trunk is identical across
every arm of the grid; the one arm permitted to widen it is a contingent control
that lies outside the grid, introduced in §4.2.

#### Encoder ladder

| level | name | front-end                                                                            | spatial access                                          | role |
|---|---|--------------------------------------------------------------------------------------|---------------------------------------------------------|---|
| E1 | `gap` | spatial mean over `(8, 8)`                                                           | *none*, channel means only                              | reference |
| E2 | `pool` | learned per-channel mask, input-independent: $f_c =  \sum_{h,w} a_c[h,w] · A_c[h,w]$ | *input-independent*, *where* it reads is the same for every input | treatment |
| E3 | `conv` | Conv3×3 stack, stride-2 -> flatten -> dense                                          | *input-conditioned*, with convolutional prior           | treatment |
| E4 | `flat` | AvgPool 2×2 -> flatten -> dense                                                      | *input-conditioned*, with no learned convolutional prior | control for E3 |

E3 and E4 both reach the trunk from the full tensor rather than from channel
means, and both apply a learned nonlinear stage before contracting space. Both
are therefore input-conditioned, and what separates them is the prior on that
stage. For E3 it is a full convolutional prior, through the application of several
convolutional layers; for E4 it is a locality prior alone, through the fixed
2×2 average pool.

#### Decoder ladder

| level | name | back-end                                                       | spatial access                                    | role |
|---|---|----------------------------------------------------------------|---------------------------------------------------|---|
| D1 | `broadcast` | $g_c$ added at every position                                  | *none*, uniform per-channel bias                  | reference |
| D2 | `mask` | $D_c[h,w] = g_c · s_c[h,w]$, $s$ learned and input-independent | *input-independent* shape, message-set amplitudes | treatment |
| D3 | `convt` | dense -> reshape -> transpose convs                            | *message-conditioned* shape, convolutional prior  | treatment |


#### Duality and outliers

E1/D1, E2/D2 and E3/D3 form architectural pairs. E4 has no partner: the
decoder that would have served as its counterpart, an unconstrained dense map
`Linear(value_dim -> C·H·W)`, was dropped for its unsustainable parameter
requirement.

This is also why E4 is crossed against one decoder rather than all three
(§3.3): the question it answers is about the encoder and has no decoder
counterpart to interact with.

#### Initialisation

E2 initialises $a_c = 1/HW$ and D2 initialises $s_c = 1$, so each begins
numerically identical to its reference level, E1 and D1 respectively. Crucially,
the existing zero-init discipline is preserved: the injection is still an exact
no-op at step 0.

#### Registry names

Encoders are `gap`, `pool`, `conv`, `flat`; decoders are `broadcast`, `mask`,
`convt`. Study 01's `mlp` encoder and `mlp` decoder are E1 and D1 renamed: the
MLP trunk is shared by every level of both ladders, so naming an arm after the
trunk's architecture is uninformative. The side effect is that Study 01 is no
longer reproducible at HEAD from its recorded flags; it remains reproducible at
its own commit.

### 3.3 Arm grid

An **arm** is an (encoder, decoder) pair trained end to end. An individual
level is not executable in isolation, since every run requires both adapters;
the two ladders of §3.2 therefore constitute the axes of the design rather
than the experiments themselves.

| | D1 `broadcast` | D2 `mask` | D3 `convt` |
|---|---|---|---|
| **E1** `gap` | ✓ *(Study 01 `mlp`, recomputed)* | ✓ | ✓ |
| **E2** `pool` | ✓ | ✓ | ✓ |
| **E3** `conv` | ✓ | ✓ | ✓ |
| **E4** `flat` | ✓ | — | — |

One further arm sits outside the grid and is executed only if its trigger fires.
It is defined in §4.2 and listed here so that the full set of arms the study may
run is visible in one place.

| contingent arm | differs from E1×D1 by | question it settles | trigger | cost |
|---|---|---|---|---|
| E1×D1, wide trunk | the MLP trunk widened to E3's trainable parameter count | whether a read-side gain is attributable to spatial access or to the parameters that carry it | a read-side effect clears §4.3's floor and is attributable to E3 | 5 runs |

The grid comprises ten arms, all of which are newly executed. E1×D1 shares
Study 01's `mlp` arm's manifest, split, joint seeds, epoch count, `k_rounds` and
`k16/v64` widths, and differs from it in one respect: the classification heads
are frozen here and were trainable there (§4.1). Three considerations motivate
the recomputation. The diagnostics of §5 postdate Study 01, so the
existing run records none of them, and recomputation yields a reference arm
whose diagnostic series are directly comparable with those of the remaining
nine. Several of those series also take values that are known in advance
at D1, since a broadcast injection is spatially constant and its residual
fraction is consequently zero whatever the arm has learned. Recomputation thus
exercises the new diagnostic code on cases whose output is predicted, and is
the study's only opportunity to distinguish a mis-wired measurement from a
substantive result. The third is that the single controlled difference from
Study 01's `mlp` arm makes the pair a measurement in its own right, of how much
of that study's accuracy came from head adaptation rather than from
communication. At five joint seeds the study therefore comprises 50
runs, or 55 should the contingent arm above be triggered.

The three treatment levels of each ladder are crossed completely rather than
sampled along the diagonal. Each row of the grid asks how much decoder spatial
access contributes *conditioned* on a given level of encoder spatial access,
which is a conditional effect rather than a main effect. Since §1 holds that
the two mechanisms may not be separable, these conditional effects are
precisely the quantities in question, and each row and column must therefore be
read in full; omitting the off-diagonal cells would render the non-separability
claim untestable.

Five contrasts carry the analysis, and §6 fixes the reading of each. The grid's
first column, E\*×D1, isolates the read-side effect with write-side access held
at zero, and its first row, E1×D\*, isolates the write-side effect with
read-side access held at zero; the latter determines whether a decoder possessing spatial
access can place a correction usefully given a message constructed without it.
E3×D3 against E1×D3 then qualifies whatever the first row reports, in that it
determines whether spatial access at the read port allows the encoder to supply
placement information that a spatially averaged message cannot. The decoder's
ability to exploit such information is crucial to this test, and accuracy is
accordingly read together with the effective rank of each arm's injection (§5).

The two remaining contrasts are the diagonal and the encoder control. The
diagonal, E1×D1 -> E2×D2 -> E3×D3, is the matched-access progression and
constitutes the most direct test of our hypothesis, while E4×D1 against E3×D1
separates a learned convolutional prior from a locality prior alone, on the
read side only.

Two aspects of the design are deliberately left uncrossed. E4 appears against
D1 alone: it has no decoder counterpart (§3.2), the question it addresses is
internal to the encoder ladder, and crossing it against D2 and D3 would add two
cells and ten runs in order to probe an interaction that §1 does not predict.
Message width is likewise held fixed, at `k16/v64` throughout, since Study 01
established that width has no immediate effect on validation accuracy; it is
therefore a held-fixed control (§4.1) rather than a factor of this design.

Arm identifiers follow the checkpoint `arch` string defined in
`ensemble/train.py`, namely `<encoder>_<decoder>_k16_v64_r1`; for example,
`conv_convt_k16_v64_r1`.

## 4. Controls

### 4.1 Held fixed

Every quantity below is identical across all arms. Those carried over from
Study 01 are held at Study 01's value so that the recomputed reference arm
(§3.3) remains comparable with it; the ports, the trunk and heads, and the
message widths are specified in §§3.1 to 3.3.

| quantity | value |
|---|---|
| expert group | `selection/manifests/encoder_study_group.json`, k = 3 |
| data split | `splits/three_way_seed42.pt`: 20,000 ensemble / 5,000 validation / official 10,000 test |
| joint seeds | 476981973, 621715752, 755818579, 1882162633, 2044397375 |
| specialists | n = 3, `k_rounds` = 1 |
| message widths | `key_dim` = 16, `value_dim` = 64 |
| ports | `--late layer3.0`, `--early layer1.2` |
| trunk and heads | the shared MLP trunk and the three QKV heads |
| decoder initialisation | final layer zero-initialised; injection an exact no-op at step 0 |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, no schedule |
| batch size | 128 |
| epochs | 70 |
| augmentation | random horizontal flip |
| determinism | `cudnn.deterministic` enabled |
| specialists | frozen entirely, classification heads included |

Nothing in a specialist trains. The classification head is frozen along with
the body, which leaves the encoder and decoder as the only parameters the loss
can reach, and makes any reduction in loss attributable to communication by
construction. A trainable head (as used in Study 01) would supply a second
route: a specialist could improve by refitting its readout to whatever
distribution the injection produced, which is a different mechanism from the
injection being more useful. 

Study 01 trained with the heads unfrozen, and the consequences are recorded in
§2 and §3.3.

The expert group is reused rather than reselected. Reselection would introduce a
second difference between this study and Study 01 and would invalidate the
reference arm's role as a bridge between them, for no gain.

Arms are additionally comparable as runs: all ten are executed from a single
commit, on one machine, with deterministic kernels enabled, so that a difference
between two arms cannot be attributed to the software or hardware under which
each was trained. 

### 4.2 Parameter budget

The two ladders confound spatial access with parameter count. E2 and D2 add one
mask of `C·H·W` entries; E3 and D3 add convolutional stacks whose cost is set by
the depths and widths chosen in §7. A gain anywhere in the grid is therefore
open to the objection that it was bought with parameters rather than with
spatial access.

This study does not match parameter counts across the grid, for two reasons.
Study 01 varied channel capacity directly and found no effect (§2.1), so a
capacity explanation for any effect observed here is already disfavoured.
Matching counts would also require widening the trunk of the low-access arms,
which §3.2 holds fixed, so the control cannot be imposed on the grid without
disturbing the one component every arm is supposed to share.

Two provisions follow. First, §7 sizes E4 to match E3's trainable parameter
count as closely as the two architectures permit, so that the E4×D1 against
E3×D1 contrast isolates the convolutional prior rather than doubling as a
capacity contrast. Second, a parameter-matched control is held in reserve: should
a read-side effect clear §4.3's floor and be attributable to E3, E1×D1 is rerun
with its trunk widened to E3's parameter count, at a cost of five runs. That arm
sits outside the grid of §3.3, its trigger is stated in §6, and it is the sole
sanctioned exception to §3.2's identical trunk. Study 01's capacity result makes
the trigger unlikely to fire, which is precisely why the control is contingent
rather than executed up front.

No equivalent control is available on the write side. The architecture that
would have supplied one, an unconstrained dense decoder, was dropped on cost
grounds (§3.2), and no remaining decoder has D3's parameter count without D3's
spatial access. A write-side gain is therefore attributable to spatial access
and to the parameters that accompany it only jointly. §13 records this as a
limitation.

### 4.3 Replication and the effect-size floor

Each arm is replicated across the five joint seeds of §4.1. Because that seed
list buys no paired-comparison advantage (§2.2), every cross-arm comparison uses
the unpaired statistic, and the anticipated floor is the one tabulated there.

**Decision rule.** A difference in accuracy between two arms is reported as real
only where it exceeds twice the standard error of that difference, the standard
error being computed from the pooled seed standard deviation of this study's own
arms. The figures of §2.2 fix the anticipated floor in advance; the realised
floor is recomputed from Study 02's runs and recorded in §11.

**Endpoint.** The accuracy endpoint is test accuracy at the epoch selected by
validation accuracy, which requires test accuracy to be tracked per epoch rather
than once at termination. Selection never consults the test set. Since the
validation curve has no sharp optimum (§2.2), the endpoint is the mean test
accuracy over a window of epochs centred on the best validation epoch rather
than the value at that epoch alone. That window is **five epochs**, the selected
epoch and two either side, clipped at the ends of training. Clipping is not
hypothetical: Study 01's best validation epoch fell as early as epoch 0, the
first epoch (§2.2), so a window of this width runs off the start of a run in
practice. The endpoint is recorded as `final_test_accuracy_endpoint`.

**Accuracy is a secondary endpoint.** The floor of roughly 0.4 percentage points
exceeds the entire spread of Study 01's six arms (§2.1). An effect of comparable
magnitude is not resolvable at five seeds, and lowering the floor to that scale
would require approximately twenty-six seeds per arm, or some 260 runs. Accuracy is accordingly reported together with its floor and
treated as corroborating evidence, while the diagnostics of §5 carry the primary
inference. §6's decision rules are written on that basis.

**Multiplicity.** The five contrasts enumerated in §3.3 are inferential and
pre-registered. Every other comparison the grid admits is exploratory and
reported as such. No formal multiplicity correction is applied, since at five
seeds the effect-size floor, and not the false-positive rate, is the binding
constraint on what this study can conclude.

## 5. Diagnostics

This section records which of the tracked
metrics bear on this study's question. It will also introduce six new metrics, and
determine which readings are prerequisites for others being interpretable at all.

### 5.1 Existing metrics

**Message routing:** §1 holds that an encoder with spatial access
changes both what is said and who is listened to; existing metrics allow us to gauge the latter:
`routing_mode_share`, both per-receiver and joint,
together with `attn_top1`, `attn_entropy` and `val_route_flip_rate`. §2.1
establishes that such metrics are the series this pipeline moves resolvably, by four
to five standard errors, under an architectural change that left accuracy
untouched. Because of this, the read side's comparative inference rests on these metrics
rather than on anything added for this study; the read-side instruments of §5.2,
the mask statistic and the message dimensionality series, serve as preconditions
to them. `self_attn_rate` will be read alongside these metrics, since
a receiver attending to its own message is not routing in any useful sense.

**Message content:** `msg_norm`, across its `query`, `key`, `value`,
`aggregated` and `injection` parts. The `injection` part exists to detect a
channel that has gone silent, which is a common outcome in multi-agent
communication with a learned channel and would invalidate every other reading
from that arm.

**Optimisation:** `grad_norm` and `update_ratio`, per component. These matter
more here than their routine diagnostic role suggests, for two reasons. §2.1
traced the Study 01 linear-encoder arms' degenerate routing to query and key heads that were receiving
almost no gradient, so a trainability failure is a live possibility. And the spatial mappings
this study introduces appear as the `reduce` and `expand` components. This is where E2's
pooling mask and D2's spatial mask live.

**Message interpretation:** the per-specialist correction, corruption and flip
rates that `answer_shift` used to carry have been removed, and are replaced by
the ensemble-level series of §5.2. They measured whether a member's own argmax
moved toward the label, which nothing optimises: the loss and the reported
accuracy are both on the prob-averaged readout, so a member flipping away from
the label is not intrinsically bad for the ensemble and one flipping toward it
is not intrinsically good, and averaging those flips over members gives a
quantity no part of the pipeline acts on. §2.1's finding that they were flat is
therefore not evidence that communication changes nothing; it is evidence about
a statistic that was never aligned with the objective.

`answer_shift_kl` survives, being the only reading of probability movement too
small to flip any argmax, and `routed_correct` survives unchanged. No prediction
attaches to either.

**Task performance:** `val_accuracy` and `test_accuracy`, both per epoch, the
latter being a change from Study 01 required by §4.3's endpoint.

### 5.2 Metrics added for this study

Eight metrics are added. Throughout, $D \in \mathbb{R}^{B \times C \times H \times W}$
denotes the injection tensor at the write port, with $C = 16$ and $H = W = 32$
(§3.1), and $a \in \mathbb{R}^{C \times H \times W}$ denotes E2's learned pooling mask
over the read port, with $H = W = 8$. For specialist $i$ of $n$, $V_i \in \mathbb{R}^{B \times d_v}$
denotes the value message emitted and $m_i \in \mathbb{R}^{B \times d_v}$ the aggregate
received, with $d_v = 64$ (§3.3).

| metric | quantity | value fixed by construction |
|---|---|---|
| `inj_resid_frac` | share of injection magnitude varying over space | exactly 0 at D1 |
| `inj_rank` | effective number of distinct spatial patterns | 0 at D1, exactly 1 at D2 |
| `inj_spread` | share of injection magnitude varying across examples | — |
| `pool_mask_dev` | departure of E2's mask from uniform | exactly 0 for E2 at initialisation |
| `msg_rank[part=value \| aggregated]` | effective directions of variation in a message emitted, and in one received | — |
| `msg_rank_ratio` | directions received relative to directions emitted | — |
| `joint_answer_shift[kind]` | correction and corruption of the ensemble's own answer | `pre_accuracy` constant across all arms, seeds and epochs |
| `marginal_answer_shift[specialist, kind]` | the same when only one member's update is substituted in | — |

#### Spatial residual fraction

Separate the injection into its per-channel spatial mean and what remains. For each example
and channel,

$$
\bar{D}_{b,c} = \frac{1}{HW}\sum_{h,w} D_{b,c,h,w},
\qquad
R_{b,c,h,w} = D_{b,c,h,w} - \bar{D}_{b,c} .
$$

Note that these two parts are orthogonal. Since $\sum_{h,w} R_{b,c,h,w} = 0$ for every
$(b,c)$ by construction, it follows that the inner product of the residual with the broadcast
mean vanishes:

$$
\langle R, \bar{D}\rangle
= \sum_{b,c} \bar{D}_{b,c} \sum_{h,w} R_{b,c,h,w}
= 0 .
$$

From this, since $D = \bar D + R$ with $\bar D$ broadcast over $(h, w)$, we can decompose the injection tensor's magnitude in quadrature, $\lVert D\rVert_F^2 = \lVert \bar{D}\rVert_F^2 + \lVert R\rVert_F^2$.
This means that the fraction formed from the residual and the injection magnitudes is bounded between $0$ and $1$, forming our first metric:

$$
\texttt{inj_resid_frac} = \frac{\lVert R\rVert_F}{\lVert D\rVert_F} \in [0, 1],
$$

with its square reading as the share of squared magnitude carried by the
spatially varying part.

At D1 the decoder emits $D_{b,c,h,w} = g_{b,c}$,
independent of position, so $\bar{D}_{b,c} = g_{b,c}$ and $R = 0$ exactly.


Note that this quantity is a property of the injection and not of the message: D2 attains a
high residual fraction while carrying no spatial information whatsoever.

The underlying intent of this measurement is to quantify how much spatial structure is present in a particular
injection tensor.

The ratio is undefined where $\lVert D\rVert_F = 0$, and this is not a remote edge case but the state of every arm at step 0, since the decoder's final layer is zero-initialised so that the injection begins an exact no-op (§3.2). The metric is reported as $0$ whenever $\lVert D\rVert_F$ falls below a tolerance of $10^{-12}$. Every arm therefore enters the series at $0$ and departs from it only as its decoder learns to write a non-zero perturbation. 

#### Batch spread

The same decomposition applied to the batch axis rather than the spatial axes.
With

$$
\tilde{D}_{c,h,w} = \frac{1}{B}\sum_b D_{b,c,h,w},
\qquad
S_{b,c,h,w} = D_{b,c,h,w} - \tilde{D}_{c,h,w},
$$

the orthogonality argument above applies unchanged, since
$\sum_b S_{b,c,h,w} = 0$, and

$$
\texttt{inj_spread} = \frac{\lVert S\rVert_F}{\lVert D\rVert_F} \in [0, 1] .
$$

This separates a working channel from a decoder emitting a large but
message-independent constant. Read with `inj_rank` it fixes which of
three regimes an arm occupies.

| `inj_rank` | `inj_spread` | interpretation |
|---|---|---|
| 1 | above zero | one fixed spatial pattern, amplitude set by the message: D2 by construction, and the mode into which D3 can collapse |
| 1 | at zero | one fixed pattern at fixed amplitude, the message ignored |
| above 1 | above zero | spatial pattern conditioned on the message |

#### Pooling mask departure

This metric is specific to the E2 encoder. E2 reads the port through $f_c = \sum_{h,w} a_c[h,w]\, A_{c}[h,w]$, initialised
at $a_c[h,w] = 1/HW$ so that the arm begins identical to E1 (§3.2). This metric aims to measure the departure from this initial form. Note that 
of particular importance is whether the mask becomes non-uniform; a mask
rescaled uniformly to $\alpha/HW$ still reads every position alike, and the
trunk downstream would then absorb the scale. This statistic is therefore the same
mean-removed form used above, applied to the mask parameters rather than an activation,

$$
\bar{a}_c = \frac{1}{HW}\sum_{h,w} a_c[h,w],
\qquad
\texttt{pool_mask_dev}
= \frac{1}{C}\sum_{c=1}^{C}
  \frac{\bigl\lVert a_c - \bar{a}_c \bigr\rVert_F}{\lVert a_c\rVert_F},
$$

which is invariant to the scale of $a_c$ and vanishes exactly when $a_c$ is
uniform, hence reads $0$ at initialisation.

E3 and E4 admit no equivalent statistic, since neither contracts space through a
single inspectable parameter.

#### Effective spatial rank

What the residual fraction cannot supply is how many *distinct* spatial patterns
the decoder produces. Flatten the residual of channel $c$ into a matrix whose
rows are examples:

$$
R^{(c)} \in \mathbb{R}^{B \times HW},
\qquad
R^{(c)}_{b,\,(h,w)} = R_{b,c,h,w},
$$

The quantity sought is the dimension of the span of those rows, which is
the subspace the decoder's output actually occupies. Ordinary rank will not serve
as that measure as it is extremely sensitive to small amounts of additive noise. In practice the rows will almost never be exactly dependent,
so the rank will essentially be $\min(B, HW)$, and a set of patterns amounting
to one pattern plus numerical noise is indistinguishable from $128$ genuinely
independent ones. What is needed is a measure that counts directions in
proportion to the magnitude they carry.

Form the Gram matrix over the batch dimension,
$G^{(c)} = R^{(c)} \bigl(R^{(c)}\bigr)^{\!\top} \in \mathbb{R}^{B \times B}$,
with eigenvalues $\lambda_1 \ge \dots \ge \lambda_B \ge 0$. The Gram matrix
discards the coordinates and retains only the geometry of the row set, its norms
and the angles between its members. Its eigenvalues are the squared singular values of $R^{(c)}$, so
$\lambda_i$ is the magnitude of the residual lying along the $i$-th principal
direction and the spectrum is the decoder's output energy itemised by
independent pattern.

The participation ratio

$$
\operatorname{PR}(c) = \frac{\bigl(\sum_i \lambda_i\bigr)^2}{\sum_i \lambda_i^2}
$$

is that magnitude-weighted count.  It
qequals $1$ exactly when $G^{(c)}$ has rank one, equals $r$ when $r$ eigenvalues
are equal and the rest vanish, and is bounded above by $\min(B, HW)$. The
weighting is what makes it usable: shares of $(0.9, 0.1)$ return $1.22$ rather
than $2$, so a dominant pattern accompanied by a weak one reads as very nearly a
single pattern. It is also invariant to the scale of $R^{(c)}$, and therefore
reports only how the decoder's output energy is distributed.

The metric averages it over channels,

$$
\texttt{inj_rank} = \frac{1}{C}\sum_{c=1}^{C} \operatorname{PR}(c) .
$$

No eigendecomposition is required. Because $G^{(c)}$ is symmetric positive
semidefinite, $\sum_i \lambda_i = \operatorname{tr} G^{(c)}$ and
$\sum_i \lambda_i^2 = \lVert G^{(c)}\rVert_F^2$, giving

$$
\operatorname{PR}(c)
= \frac{\bigl(\operatorname{tr} G^{(c)}\bigr)^2}{\lVert G^{(c)}\rVert_F^2}
= \frac{\lVert R^{(c)}\rVert_F^4}{\bigl\lVert R^{(c)} (R^{(c)})^{\!\top}\bigr\rVert_F^2},
$$

two Frobenius norms and one matrix product of cost $B^2 HW$.

An important note  is that taking the Gram over the batch dimension rather than over positions costs
nothing in information, since $R^{(c)} (R^{(c)})^{\top}$ and
$(R^{(c)})^{\top} R^{(c)}$ share their non-zero eigenvalues and hence their
participation ratio, while the batch-side matrix is the smaller of the two at
$B = 128$ against $HW = 1024$. One consequence of this choice is that $\min(B, HW) = B$ bounds the metric, so
the ceiling is set by the measuring batch and not by the architecture under test.
The readings are in this sense partly a property of the batch, which is a further
reason for §5.2's single fixed batch across every arm and epoch.

D2 gives the construction check. There
$R^{(c)}_{b,\,(h,w)} = g_{b,c}\bigl(s_c[h,w] - \bar{s}_c\bigr)$, so every row is a
scalar multiple of the single vector $s_c - \bar{s}_c$, the matrix has rank one,
and $\operatorname{PR}(c) = 1$ exactly. Where the residual vanishes identically
the ratio is $0/0$, so the metric is reported as $0$ whenever
`inj_resid_frac` falls below $10^{-6}$. D1 therefore reads $0$ and D2
reads $1$, both exactly.

The intent of this measurement is to establish the effective dimensionality of
the decoder's image: not the dimension its architecture permits, but the
dimension of what it was in fact observed to emit. D3 is built to place a
correction whose shape depends on the message, and this is the quantity that
distinguishes a decoder doing so from one that has settled on a single shape and
varies only its amplitude.

#### Effective Message Rank & Message Rank Ratio

The participation ratio defined above transfers to the messages themselves.
Writing $\operatorname{PR}(X)$ for the participation ratio of the Gram of a
matrix $X$, the only change is which mean is removed: messages carry no spatial
axis, so the mean subtracted is the batch mean and the Gram is a covariance matrix.

$$
\tilde{V}_i = V_i - \frac{1}{B}\sum_b (V_i)_b ,
\qquad
\texttt{msg_rank[part=value]} = \frac{1}{n}\sum_{i=1}^{n} \operatorname{PR}\bigl(\tilde{V}_i\bigr),
$$

and identically for `part=aggregated` with $m_i$ in place of $V_i$. Per-specialist series are tracked alongside the mean, following
`routing_mode_share`'s convention, so that one collapsed encoder among $n$ is
visible rather than diluted.

The `part=value` statistic counts the independent directions along which a sender's
message varies with the input, which is the number of degrees of freedom the
encoder exercises out of the $d_v$ available. The `part=aggregated` statistic is the same
quantity for what reaches a decoder, and it is governed by two processes: first, the senders' messages change with the input, and second,
the attention weights change with the input, so re-weighting contributes
variation of its own.

The ratio of the two,

$$
\texttt{msg_rank_ratio} = \frac{\texttt{msg_rank[part=aggregated]}}{\texttt{msg_rank[part=value]}},
$$

exceeds $1$ when what arrives spans more directions than any single sender
emits, and sits near $1$ when what arrives resembles one sender's message. Two
conditions are needed for the ratio to rise: the senders must occupy different
subspaces, and the aggregation must draw on more than one of them. The ratio
cannot say which is absent, and therefore must be read alongside
`routing_mode_share`, which settles whether attention is pinned.

The intent of this measurement is to establish whether an encoder with spatial
access exercises more of the channel than one without, and whether any
additional variation survives aggregation to reach a decoder.

#### Ensemble-level answer shift

The interpretation metrics this study reports are of the ensemble's answer, not
of each member's. Writing $\ell \in \mathbb{R}^{n \times 100 \times B}$ for the
per-specialist logits and

$$
A(\ell) = \log \frac{1}{n}\sum_{i=1}^{n} \operatorname{softmax}(\ell_i)
$$

for the prob-averaged readout. All metrics below rely on
$A(\ell^{\text{pre}})$ and
$A(\ell^{\text{post}})$, where $\text {pre}$ denotes the perception (non-communicative) pass
and $\text {post}$ denotes the last communication round.

`joint_answer_shift[kind=corrected]` is the share of examples the ensemble
answered wrongly before communication and correctly after;
`kind=corrupted` is the share it answered correctly before and wrongly after.
Their difference, reported as `kind=net`, is exactly the change in ensemble
accuracy that can be attributed to communication:

$$ 
\begin{eqnarray}
\texttt{net} &=& \texttt{corrected} - \texttt{corrupted} \\ 
             &=& \texttt{post_accuracy} - \texttt{pre_accuracy}.
\end{eqnarray}
$$

Because every specialist is frozen (§4.1), $\ell^{\text{pre}}$ depends only on
the input and the frozen weights. `kind=pre_accuracy` is therefore the same
number in every arm, every seed and every epoch, and the examples eligible for
correction are one fixed set across the whole grid. Note that this was not the case in Study 01 as its classification heads were trained in tandom with the communication channel.
As a result of this, its arms differ in pre-communication accuracy by about 0.2 percentage points.

`kind=agreement_pre` and `kind=agreement_post` are the average agreement rates across every possible pair of experts at differing
stages of communication. These metrics carry no prediction in this study; our choice of combiner means that there is no inherent value 
in concensus before or after communication. They have been retained because they are
act as a control on `net`. Agreement rising while `net` does not is consensus
without correctness.

`marginal_answer_shift` is intended to attribute the joint effect measured by the former metrics to 
specific members. For specialist $i$ it first substitutes that member's post-communication logits into an otherwise
pre-communication ensemble's logits. In other words, it generates  $\ell^{\text{pre}}$ with row $i$ replaced by
$\ell^{\text{post}}_i$, and asks what does
this member's update contribute to the joint decision on its own. The expectation is small values:
with three members and probability averaging, one shifted opinion must overcome
two unchanged ones to move the argmax.
#### Computation

The answer-shift series are computed from the pre- and post-communication
logits the forward pass already produces, so they cost no additional passes.
They are reported once per epoch for both the whole validation
set, and the accumulated batch-weighted across the epoch's training batches in the
manner of `train_accuracy`. Note that the training-side `pre_accuracy` drifts slightly between epochs where
the validation-side figure is constant: the training inputs are augmented with
random horizontal flips, so the frozen-specialist property holds of the weights
and not of the data.

The other six are computed on one fixed validation batch per epoch, the same batch at
every epoch and in every arm, so that the series are comparable both along
training and across the grid, and all are taken from the same forward pass.

## 6. Predictions & decision rules

Eight rules follow, grouped by what they establish. Each names the question it
settles, the clause of §1 it bears on, the series it reads, the contrast it
reads them over, and the threshold at which it decides. §12 reports the study
against them by number.

### 6.1 Rules governing every rule

- Every threshold is twice the standard error of the difference, with each
  series' standard error computed from this study's own seeds.
- **Measurement checks.** These readings are fixed by construction and asserted
  by `scripts/check_diagnostics.py`. A departure means the metric is reading the
  wrong tensor or reducing over the wrong axes, and voids its readings across
  the grid.
  - D1: `inj_resid_frac` and `inj_rank` of exactly 0 at every epoch.
  - D2: `inj_rank` of exactly 1 once its mask has left uniform, and 0 before
    then.
  - E2: `pool_mask_dev` of 0 at initialisation.
- **Usage checks.** An arm meeting any of the following has left unused the
  freedom its architecture grants. It is grouped with its reference arm in every
  comparison, and its rule is **untested** rather than refuted, which §11
  records.
  - A D3 arm at `inj_rank` of 1.
  - A decoder at `inj_spread` of approximately 0.
  - An E2 arm whose `pool_mask_dev` is still 0 after training.
  - An arm whose `reduce` or `expand` component receives no gradient.

### 6.2 Whether spatial access enriches the message

**R1 -  Message enrichment.**

- **Determines.** Whether spatial access at the read port increases the number
  of independent directions a sender's message occupies.
- **Bears on.** §1, read side: messages carrying more information about the
  sender's state.
- **Series.** `msg_rank[part=value]`, mean over specialists.
- **Contrast.** E2×D1 against E1×D1. These two share the trunk, the heads and
  the widths, differing only in the spatial mapping, and are therefore the only
  pair whose message dimensionality is attributable to spatial access. E3 and E4
  add front-end nonlinearity, which may raise effective dimensionality
  irrespective of message quality.
- **Confirmed if.** E2×D1 exceeds E1×D1 by more than two standard errors.
- **Refuted if.** The two lie within two standard errors of one another.
- **Untested if.** `pool_mask_dev` at E2 remains at 0 after training, the mask
  never having left its uniform initialisation.

A refutation here is stronger than a failure of significance. It would mean §1's read-side premise, that spatial
averaging at the read port is what limits message content, is wrong. That
premise is already in tension with the measurements, since a sender varies its
message along roughly seven of the 64 directions available to it, far short of
the channel width, of the 64-dimensional pooled input feeding the encoder, and
of the batch ceiling.

**R2 - Routing.**

- **Determines.** Whether spatial access makes routing depend on the input.
- **Bears on.** §1, read side: who is listened to.
- **Series.** `routing_mode_share` joint, `attn_entropy`, `attn_top1`,
  `val_route_flip_rate`.
- **Contrast.** The grid's first column, E1×D1 through E3×D1.
- **Confirmed if.** `routing_mode_share` falls by more than two standard errors
  from E1, with `attn_top1` remaining above the 0.333 that uniform attention
  over three senders would give, so that the fall is re-routing and not an
  argmax flipping between near-ties. `attn_entropy` and `val_route_flip_rate`
  are expected to rise, this being the signature §2.1 observed when the trunk
  was introduced.
- **Refuted if.** All four series lie within two standard errors across the
  column.
- **Untested if.** No condition beyond §6.1.


### 6.3 Whether spatial access improves the injection

**R3 - Spatially varying perturbation.**

- **Determines.** Whether a spatially varying injection perturbs a frozen
  backbone more usefully than a uniform channel bias.
- **Bears on.** §1, write side.
- **Series.** `final_test_accuracy_endpoint` (§4.3), gated on `inj_resid_frac` above 0,
  `inj_spread` above 0, and `inj_rank` above 1 at D3.
- **Contrast.** The grid's first row, E1×D1 through E1×D3, and the diagonal.
- **Confirmed if.** Accuracy at D2 or D3 exceeds D1 at the same encoder level by
  more than §4.3's floor.
- **Refuted if.** D2 and D3 both lie within the floor of D1 while their gating
  series pass. 
- **Untested if.** Any gating series fails.

**R4 - Message-conditioned placement.**

- **Determines.** Whether information sufficient to place a correction survives
  global average pooling at the read port.
- **Bears on.** §1, write side, and the design question of whether read-side
  spatial access is a prerequisite for write-side placement.
- **Series.** `inj_rank` at E1×D3, read against E3×D3.
- **Contrast.** E1×D3 against E3×D3.
- **Confirmed if.** `inj_rank` at E1×D3 exceeds 1 by more than two standard
  errors. Placement information then survives spatial averaging and must be
  carried in channel identity or channel magnitude.
- **Refuted if.** `inj_rank` at E1×D3 lies within two standard errors of 1. The
  decoder has collapsed to a fixed shape under amplitude modulation, which is
  D2's behaviour, and any accuracy E1×D3 gains over E1×D1 is not attributable to
  message-conditioned placement.
- **Untested if.** `inj_resid_frac` at E1×D3 is 0, no injection structure
  existing to rank.

Accuracy cannot separate these two outcomes, since both admit a gain over
E1×D1. `inj_rank` is the arbiter, which is why §3.3 reads this contrast
alongside it.

### 6.4 Whether the two mechanisms are separable

**R5 - Transmission.**

- **Determines.** Whether the additional directions R1 finds in senders'
  messages are still present in what a decoder receives, once the bus has
  combined those messages.
- **Bears on.** §1's closing note, that messages carrying spatial information
  may require a decoder with spatial access before that information is usable.
- **Series.** `msg_rank_ratio`, read alongside `routing_mode_share`, which
  settles whether a low ratio reflects pinned attention or redundant senders.
- **Contrast.** Along the encoder column at fixed decoder level.
- **Confirmed if.** `msg_rank_ratio` rises with encoder level by more than two
  standard errors, the additional content arriving differentiated.
- **Refuted if.** The ratio stays flat while R1 confirms. Content is then being
  produced and not transmitted, which locates the limitation in the bus rather
  than at either port.
- **Untested if.** R1 is itself untested or refuted, there being no additional
  content whose transmission could be assessed.

### 6.5 Whether any of it improves the ensemble

Two rules, asking different questions of the same endpoint: R6 whether an arm
beats the reference arm, R8 whether its channel beats not communicating at all.
They can disagree in either direction.

**R6 — Task performance.**

- **Determines.** Whether a mechanism shown to have operated also improved
  classification.
- **Bears on.** The hypothesis's claim that spatial access improves the channel,
  as distinct from its claims about how.
- **Series.** `final_test_accuracy_endpoint` (§4.3).
- **Contrast.** All five contrasts of §3.3.
- **Confirmed if.** An arm exceeds its reference by more than §4.3's floor.
- **Refuted if.** Every arm lies within the floor of E1×D1.
- **Untested if.** No condition beyond §6.1.

**R8 — Net joint correction.**

- **Determines.** Whether a channel improves the ensemble over not
  communicating, as distinct from improving it over another channel.
- **Bears on.** The hypothesis's premise rather than either of its mechanisms:
  §1 asks whether spatial access makes the channel better, and takes for
  granted that a channel is worth having.
- **Series.** `val_joint_answer_shift` at `kind=net`, read with its `corrected`
  and `corrupted` components and against `agreement_post`.
- **Contrast.** Each arm against itself. `kind=pre_accuracy` is the
  no-communication ensemble and is identical in every arm (§4.1), so this is
  the one rule in the section that needs no second cell.
- **Confirmed if.** `net` exceeds zero by more than two standard errors across
  seeds.
- **Refuted if.** `net` is within two standard errors of zero, or negative
  beyond them. A negative reading is the stronger outcome: the channel is
  actively harmful, and §1's question of whether spatial access improves it is
  then subordinate to whether it should exist.
- **Untested if.** No condition beyond §6.1.

### 6.6 Exploratory and contingent

**R7 — Prior.** Exploratory, not inferential.

- **Determines.** Whether a learned convolutional prior outperforms a fixed
  locality prior at the read port.
- **Bears on.** Nothing in §1, which concerns spatial access and not the prior
  through which it is obtained.
- **Series.** `final_test_accuracy_endpoint` (§4.3). `msg_rank[part=value]` is not interpretable
  across this pair, both arms adding front-end nonlinearity.
- **Contrast.** E4×D1 against E3×D1, their parameter counts matched by §7.
- **Reading.** E3 exceeding E4 beyond the floor attributes the difference to the
  learned convolutional prior rather than to locality. The two lying within the
  floor means a fixed local pool suffices and the convolutional stack is
  unnecessary expense.

## 7. Architectural parameters

Shapes are those of §3.1: `(64, 8, 8)` at the read port and `(16, 32, 32)` at
the write port. Every count below is for one specialist. Each of the three
carries its own encoder and its own decoder, so an arm's total is three times
the figure given, as §7.3 sets out.

Every encoder emits a 64-vector into a trunk shared by that encoder's query,
key and value heads, and not between specialists. Its cost is constant across
arms at 10,528 parameters for the trunk and the three heads, and is counted
once per specialist.

### 7.1 Encoder front-ends

| level | front-end | parameters per specialist |
|---|---|---|
| E1 `gap` | spatial mean over `(8, 8)` | 0 |
| E2 `pool` | learned mask over `(8, 8)` per channel, initialised at $1/HW$ | 4,096 |
| E3 `conv` | Conv3×3 64→32 stride 2, GELU, Conv3×3 32→32 stride 2, GELU, flatten to 128, Linear 128→64 | 35,968 |
| E4 `flat` | AvgPool 2×2, flatten to 1024, Linear 1024→32, GELU, Linear 32→64 | 34,912 |

E4's hidden width of 32 is chosen to match E3's parameter count. E4
is within 3 percent of E3's parameter count, with a ratio of 0.971.

E3's two strided convolutions reduce `(64, 8, 8)` to `(32, 2, 2)` inside the
convolutional stack, so the flatten reaches the trunk at 128 rather than the
4,096 a full flatten would present. E4's single 2×2 average pool reaches 1,024
and contracts through its hidden layer instead. 

### 7.2 Decoder back-ends

| level | back-end | parameters per specialist |
|---|---|---|
| D1 `broadcast` | Linear 64→64, GELU, Linear 64→16, broadcast over `(32, 32)` | 5,200 |
| D2 `mask` | D1's layers, with a learned `(16, 32, 32)` mask initialised at 1 | 21,584 |
| D3 `convt` | Linear 64→64, GELU, Linear 64→256, reshape to `(16, 4, 4)`, three ConvTranspose 4×4 stride 2 with GELU between | 33,136 |

D3 reshapes its dense output to `(16, 4, 4)` and upsamples from there, with three doublings
reaching the write port. The choice of starting tensor in the transpose-convolutional stack is in an attempt to keep the total ensemble parameter count reasonable. 

The final transpose convolution is zero-initialised, which is what preserves the injection as an exact no-op at step 0. D2 instead
zero-initialises the last Linear of its MLP and sets its mask to 1, since
zeroing both factors of a product would leave both permanently without gradient.

No decoder level matches another's parameter count, and §4.2 records the
consequence as a limitation rather than attempting to remedy it.

### 7.3 Per-arm parameter budget

Each specialist carries the constant 10,528 parameters of its trunk and heads
plus its own front-end and back-end. Note that the classification head is frozen
(§4.1) and so contributes nothing as opposed to Study 01. Arm totals are three times the
per-specialist figure.

| arm | per specialist | arm total | relative to E1×D1 |
|---|---|---|---|
| E1×D1 | 15,728 | 47,184 | 1.00 |
| E1×D2 | 32,112 | 96,336 | 2.04 |
| E1×D3 | 43,664 | 130,992 | 2.78 |
| E2×D1 | 19,824 | 59,472 | 1.26 |
| E2×D2 | 36,208 | 108,624 | 2.30 |
| E2×D3 | 47,760 | 143,280 | 3.04 |
| E3×D1 | 51,696 | 155,088 | 3.29 |
| E3×D2 | 68,080 | 204,240 | 4.33 |
| E3×D3 | 79,632 | 238,896 | 5.06 |
| E4×D1 | 50,640 | 151,920 | 3.22 |

The grid spans a factor of 5.06 in trainable parameters, which is the confound
§4.2 describes and declines to match away.
For scale, the largest arm trains 238,896 parameters against the roughly 1.7
million of a single frozen ResNet-110, so even E3×D3 adds about 14 per cent of
one backbone and under 5 per cent of the three.

The contingent control of §4.2, should §6.6 trigger it, widens E1×D1's trunk
until its per-specialist count reaches E3×D1's 51,696.

## 8. Execution

### 8.1 Commands

One arm at a time, the arm name carrying both halves of the architecture:

```
bash scripts/run_spatial_study.sh gap_broadcast
```

The whole sequence, detached so it survives the session that started it:

```
bash scripts/run_all_spatial_study.sh            # detach and return
bash scripts/run_all_spatial_study.sh --fg       # foreground
bash scripts/run_all_spatial_study.sh --dry-run  # print, touch nothing
```

Arms run in sequence and not in parallel: they share one GPU, and §4.1 requires
them trained under identical conditions. Both scripts forward anything after
the arm name to `replicate_joint.py`, so `--limit` and `--aggregate-only` work
through them.

The sequence is resumable. Per-seed state lives in
`scripts/replicate_state/spatial_<arm>.json`, and a rerun skips seeds already
recorded `done` while retrying anything `failed` or `interrupted`, so an
interrupted sequence is restarted by reissuing the same command. A failing arm
does not abort the ones after it.

Order is not required by the analysis, every arm being read against E1×D1, and
is chosen so that an interrupted sequence has still answered something:

| stage | arms | cumulative | what it settles |
|---|---|---|---|
| 1 | `gap_broadcast` | 1.5h | §6.1's measurement checks, on which everything else depends |
| 2 | `pool_mask`, `conv_convt` | 4.5h | the matched-access diagonal, §1's most direct test |
| 3 | `gap_mask`, `gap_convt` | 7.6h | R3 and R4 |
| 4 | `pool_broadcast`, `conv_broadcast` | 10.6h | R1, R2, R5, and all four corners of the interaction |
| 5 | `flat_broadcast` | 12.1h | R7 |
| 6 | `pool_convt`, `conv_mask` | 15.3h | R6 across all five contrasts, and the conditional effects of §3.3 |

Every mechanism rule therefore has an answer by stage 4, and the remaining
4.7 hours provide the conditional structure and the prior control.

### 8.2 Runtime

Measured on 2026-10-04 on an RTX 6000 Ada, 157 steps per epoch at batch 128.
These are measurements rather than estimates. They were taken while another
job was using the same GPU at roughly 85 per cent utilisation, so on a dedicated
GPU the times should be shorter
| arm | per epoch | per seed | per arm |
|---|---|---|---|
| `gap_broadcast` | 15.3s | 17.8 min | 1.49h |
| `pool_mask` | 15.3s | 17.9 min | 1.49h |
| `gap_mask` | 15.3s | 17.9 min | 1.49h |
| `flat_broadcast` | 15.5s | 18.1 min | 1.51h |
| `pool_broadcast` | 15.6s | 18.2 min | 1.52h |
| `conv_broadcast` | 15.6s | 18.2 min | 1.52h |
| `gap_convt` | 15.9s | 18.6 min | 1.55h |
| `pool_convt` | 15.9s | 18.6 min | 1.55h |
| `conv_convt` | 16.0s | 18.8 min | 1.56h |
| `conv_mask` | 16.1s | 18.8 min | 1.57h |
| **ten arms** | | | **15.3h** |

The contingent control of §4.2 adds five runs, about 1.5 hours, if §6.6
triggers it.

Architecture accounts for 5 per cent of that spread despite the 5.06-fold range
in trainable parameters (§7.3), because six frozen ResNet-110 forward passes per
step dominate anything the adapters do. Within an epoch of `conv_convt`:

| component | time | share |
|---|---|---|
| training steps, excluding the streaming validation forward | 7.9s | 48% |
| the per-step streaming validation forward | 5.0s | 31% |
| validation pass with answer-shift statistics | 1.2s | 7% |
| test pass | 2.2s | 13% |
| the spatial diagnostics of §5.2 | 0.05s | 0.3% |

Two of these are worth noting. The per-step validation forward, which supplies
`val_loss` at training-step frequency, is the largest discretionary cost in the
study at roughly 4.8 hours across the grid, and no rule in §6 reads that series;
it is retained because dropping it would be a departure from Study 01's
conditions, and §9 is where such a change would be recorded. And §5.2's six
diagnostics cost 50 milliseconds an epoch, two and a half minutes across all 50
runs, so the single-fixed-batch cadence was not a concession to cost.

### 8.3 Artifacts

| artifact | location |
|---|---|
| Aim experiment, per arm | `spatial_<arm>`, e.g. `spatial_conv_convt` |
| seed-aggregated summary | one run per experiment, tagged `summary`, carrying `<metric>_mean` and `<metric>_std` |
| per-seed state | `scripts/replicate_state/spatial_<arm>.json` |
| checkpoints | `ensemble/checkpoints/communicative_<three backbones>_<encoder>_<decoder>_k16_v64_r1_seed<seed>.pt` |
| arm log | `logs/spatial_study/<arm>.log` |
| sequence log | `logs/spatial_study/_sequence_<timestamp>.log` |

The checkpoint name carries the decoder as well as the encoder, which Study 01's
did not. Without it the arms of this grid that share an encoder would overwrite
one another seed for seed, since every arm also shares one expert group.

`bash scripts/spatial_progress.sh` summarises per-arm and per-seed state against
the logs in one screen. The Aim dashboard binds to `127.0.0.1:43800` under
`aim up`, reachable with `ssh -L 43800:localhost:43800`.

Two scripts check the architectures and the diagnostics against this protocol
rather than against themselves: `scripts/check_arch_params.py` asserts §7's
parameter counts and the initialisation guarantees of §3.2, and
`scripts/check_diagnostics.py` asserts the construction values §6.1 depends on.
Both are expected to pass before an arm is run and after any change to the
adapters.

---

# Part II — Record

## 9. Deviations from protocol

Dated log of anything that changed after §§1-8 were frozen — crashed seeds,
altered epoch counts, bugs found mid-run. Without this the freeze is
decorative.

_(none yet)_

## 10. Reading plan

### 10.1 Order of reading

The results are read in six parts. Every number reported is computed from Aim by one extraction
script and read from its output table, and is not transcribed by hand. The arm
value is the mean of five per-seed values, and its standard error is computed
as §6.1 specifies.

Each per-seed value is a mean over a window fixed here, before any result is
read, and placed independently of the series under test:

| rules | window |
|---|---|
| R1, R2, R4, R5, usage checks | epochs 60–69, what the arm learned to do |
| R3's gating series, R6, R7, R8 | §4.3's endpoint window, the model whose accuracy is reported |
| measurement checks | every epoch |

Curves are inspected throughout, but if a curve is still trending within its window it is still reported.
The trend will also be recorded as an observation.

**Part 0: measurement validity, all ten arms.** No later part can be read
until this one is complete.

- Run integrity: fifty runs and ten summary runs present, no non-finite values,
  and training curves free of divergence.
- §6.1's measurement checks.
- §6.1's usage checks, per arm and per seed. These determine which arms are
  grouped with their reference arm and which rules are recorded as untested, so
  they precede every contrast.
- E1×D1 against Study 01's `mlp` arm. The single controlled difference between
  the two is the freezing of the classification heads (§3.3), so the
  difference measures how much of Study 01's accuracy came from head
  adaptation.

**Part 1: the read side, E\*×D1.** Write-side access is held at zero.

- R1, E2×D1 against E1×D1 on `msg_rank[part=value]`.
- R2, routing across the column.
- R5 at D1, `msg_rank_ratio` along the column.
- R7, E4×D1 against E3×D1.

**Part 2: the write side, E1×D\*.** Read-side access is held at zero.

- R3 along the row.
- R4's `inj_rank` reading at E1×D3.

**Part 3: the diagonal and E3×D3 against E1×D3.**

- R3 along the diagonal.
- R4's contrast.
- The matched-access progression E1×D1, E2×D2, E3×D3, which is §1's most
  direct test.

**Part 4: the full 3×3 grid.**

- The conditional effects of §3.3, including the two off-diagonal arms
  E2×D3 and E3×D2, which enter the analysis here for the first time.
- R5 at D2 and D3.
- R6 across all five contrasts.

**Part 5: each arm against itself.**

- R8, `val_joint_answer_shift` at `kind=net` with its `corrected` and
  `corrupted` components and `agreement_post`, tabulated for all ten arms
  together.

### 10.2 Series read in every part

Each part reads the series its rules name. In addition, the following series
are read in every part, since none of them is named by a rule and an
unanticipated result is most likely to appear in one of them.

- `grad_norm` per component, in particular `reduce` and `expand`, and whether
  the query and key heads lose their gradient as they did in Study 01 (§2.1).
- `msg_norm` and injection magnitude.
- Routing per seed as well as its mean, Study 01's reference arm having
  contained two fully degenerate seeds whose mean concealed them.
- The validation curve: the gap between training and validation accuracy, the
  epoch of best validation accuracy, and how far the final epoch lies below it.
- Seed standard deviation per arm, read against the pooled value of §2.2, to
  establish whether any architecture is markedly less stable across seeds.
- Differences between specialists within an arm.
- Trainable parameter count against accuracy, which bears on whether the
  contingent arm of §4.2 is triggered.

### 10.3 Recording

Each part records two things:

- A table with one row per rule read, giving the contrast, the difference, its
  standard error and the verdict.
- A numbered list of observations falling outside the rules.

The six parts together make a few dozen comparisons at a two-standard-error
threshold, so one or two are expected to clear it by chance even in the
absence of any real effect. The rules of §6 were fixed before the results were
seen and are reported as tests. Every observation from §10.2 was not, and is
reported as exploratory: grounds for a follow-up study, not a finding of this
one.

## 11. Results

### 11.1 Part 0: measurement validity

All fifty runs completed seventy epochs with finite values, each summary run
aggregates the five joint seeds, and no run diverged. Every check of §6.1
passes, so no arm is grouped with its reference arm.

| check | reading |
|---|---|
| D1 `inj_resid_frac`, `inj_rank` | exactly 0 at every epoch |
| D2 `inj_rank` | exactly 1 at every logged epoch |
| E2 `pool_mask_dev` | 0 at initialisation (`check_diagnostics.py`), 0.92–0.99 after training |
| D3 `inj_rank` | 2.62, 3.34, 1.50 at E1, E2, E3 |
| `inj_spread` | 0.43–0.87 |
| `reduce`, `expand` gradient | zero at step 0 only |

Against Study 01's `mlp` arm, which differs only in trainable heads, E1×D1
reaches the same best validation accuracy (−0.07, −0.8 SE) but lies 0.33 below
it over epochs 60–69 (−2.4 SE): head adaptation slows the late decline rather
than raising the peak.

Observations:

1. Two seed runs carry a spurious `summary` tag in Aim; aggregation matches by
   seed and is unaffected.
2. Mean sender message rank is 1.6–2.2 in the E3 and E4 arms, against 4.3–6.9
   in the E1 arms and 6.7–11.9 in the E2 arms.
3. Sender message rank is lower in the D3 arm than in the D1 arm of rows E1
   (4.3 against 6.1) and E2 (6.7 against 11.9), but not of row E3 (1.7 against
   1.6). Each arm is trained end to end, so a difference between cells is a
   property of the pair and is not attributed to either adapter alone.

### 11.2 Part 1: the read side, E\*×D1

| rule | contrast | difference | verdict |
|---|---|---|---|
| R1 | `msg_rank[part=value]`, E2×D1 − E1×D1 | 11.94 − 6.15 = +5.80, +7.9 SE | confirmed |
| R2 | `routing_mode_share` joint, E2×D1 and E3×D1 − E1×D1 | −0.23 (−1.5 SE), −0.19 (−1.0 SE); other three series within 1.9 SE | refuted |
| R5 | `msg_rank_ratio`, E2×D1 and E3×D1 − E1×D1 | −0.13 (−0.9 SE), +0.05 (+0.3 SE) | refuted |
| R7 | endpoint, E3×D1 − E4×D1 | 66.89 − 66.68 = +0.22, floor 0.41 | within floor |

The realised floor is 0.41 points, from a pooled seed standard deviation of
0.32 over all ten arms.

Observations:

1. Sender message rank is still rising within the window in E1×D1 and E2×D1
   (+0.48 and +0.87 over epochs 60–69), so R1's difference is not yet at
   equilibrium.
2. In every D1 arm the seeds divide between degenerate routing (joint
   `routing_mode_share` 0.97–1.00) and input-dependent routing (0.15–0.33),
   which inflates R2's standard errors. Joint share is still falling within the
   window in every arm.
3. Aggregated message rank is 17.72 in E2×D1 against 9.62 in E1×D1 (+13.2 SE),
   and rises throughout training in all four arms.
4. `msg_rank_ratio` rises within the window in E3×D1 and E4×D1 (+0.19, +0.08)
   and is flat in E1×D1 and E2×D1.
5. At epoch 69 training accuracy exceeds validation accuracy by 10.7 points in
   E2×D1, against 1.6–4.7 in the other D1 arms. From epoch 20, E2×D1's
   training accuracy rises fastest (+6.9 points) and its validation accuracy
   falls fastest (−1.2 points); E4×D1's falls slowest (−0.4).
6. Gradient norms at the trunk are about seven times lower in E3×D1 and
   E4×D1 than in E1×D1 and E2×D1 (0.22–0.24 against 1.28–1.73). The heads and
   decoder show the same ordering with smaller ratios; at `reduce`, E3×D1
   (1.35) matches E2×D1 (1.32) and only E4×D1 is lower (0.94).
7. Seeds with two or more degenerate receivers number two in E1×D1, none in
   E2×D1, two in E3×D1 and three in E4×D1, and in every arm they are the seeds
   with the lowest query and key gradients. In one seed each of E3×D1 and
   E4×D1 all three receivers are degenerate and the query gradient is 0.006
   and 0.001. Gradient norms pool the three specialists, so the match is per
   seed, not per receiver.
8. Message norms are lower in E3×D1 and E4×D1 than in E1×D1 and E2×D1 at
   every point of the bus (value 6.1–6.8 against 9.1–10.2, aggregated 4.2–5.0
   against 8.2–10.3). Value and aggregated norms still rise within the window
   in E1×D1 and E2×D1 and are flat in E3×D1 and E4×D1; injection norm rises in
   all four.
9. Per-specialist message rank and marginal answer shift follow the same
   trajectory in every specialist; no arm is carried by one sender.

### 11.3 Part 2: the write side, E1×D\*

| rule | contrast | difference | verdict |
|---|---|---|---|
| R3 | endpoint, E1×D2 and E1×D3 − E1×D1 | −0.17 (−0.8 SE), −0.21 (−1.1 SE), floor 0.41; gates pass | refuted on the row |
| R4 | `inj_rank` at E1×D3 against 1 | 2.62 − 1 = +1.62, +16.4 SE | confirmed |

R3's diagonal and R4's contrast with E3×D3 are read in Part 3.

Observations:

1. The endpoint is selected before the injection has taken shape. Every
   E1×D3 seed's best validation epoch lies at 1–3, where its `inj_rank` is
   1.011 against 2.62 at the end of training; E1×D2's `inj_resid_frac` is 0.05
   over its endpoint windows against 0.29 at the end. R3's gates pass at two
   standard errors, but its accuracies describe injections that had barely
   departed from their reference arm's.
2. `inj_resid_frac` reaches 95 per cent of its final level by epochs 9–13 in
   E1×D3 but not until epochs 58–59 in E1×D2, which rises throughout training.
3. `inj_rank` at E1×D3 is still rising within the window (+0.25 over epochs
   60–69), so R4's reading is not at equilibrium.
4. Over epochs 60–69 training accuracy orders E1×D3 > E1×D2 > E1×D1 (74.1,
   72.4, 70.0) and validation accuracy the reverse (65.1, 65.3, 65.7); the
   gap at epoch 69 is 9.8, 7.7 and 4.7 points.
5. `inj_spread` rises through training in all three arms and orders
   E1×D1 > E1×D2 > E1×D3 (0.79, 0.67, 0.60).
6. `msg_norm[part=injection]` is not comparable across decoder levels: at D1
   it is the norm of the vector before broadcast to 32×32 positions, a
   factor of 32 below the injected map. Corrected, the row reads about 155,
   188 and 181.
7. Sender message rank falls early in training and then rises in all three
   arms, reaching its minimum later and lower in E1×D3 (arm mean 2.2 at
   epoch 15) than in E1×D1 (3.0 at epoch 4) and E1×D2 (2.8 at epoch 8). By
   epochs 60–69 it orders E1×D2 > E1×D1 > E1×D3 (6.89, 6.15, 4.34), as does
   aggregated rank; `msg_rank_ratio` is near constant.
8. Message norms are equal across the row (value 10.0–10.5) and rise through
   training in all three arms, so the row differs in the number of directions
   its messages occupy, not in their magnitude.
9. Joint `routing_mode_share` does not differ across the row (all pairwise
   differences within 1 SE) and declines modestly in every arm; attention
   statistics are near identical.
10. Gradient norms at the trunk, query and value heads order E1×D1 > E1×D2 >
    E1×D3 (trunk 1.73, 1.24, 0.79) and rise through training in all three.
    At the decoder MLP, E1×D2 lies below E1×D1 (0.97 against 1.26, −6.2 SE)
    and is flat while the other two rise; at `expand`, E1×D3 (0.86) is five
    times E1×D2 (0.16) and still rising.

### 11.4 Part 3: the diagonal, and E3×D3 against E1×D3

| rule | contrast | difference | verdict |
|---|---|---|---|
| R3 | endpoint, E2×D2 − E2×D1 and E3×D3 − E3×D1 | −0.27 (−1.0 SE), −0.21 (−1.1 SE), floor 0.41; gates pass | refuted on the diagonal |
| R4 | `inj_rank`, E3×D3 − E1×D3 | 1.50 − 2.62 = −1.12, −8.2 SE | E3×D3 lower |

With Part 2, R3 is refuted on the row and on the diagonal. Along the
matched-access progression, E2×D2 and E3×D3 lie −0.23 (−0.9 SE) and +0.03
(+0.2 SE) from E1×D1. E3×D3's `inj_rank` remains above 1 (+5.3 SE).

Observations:

1. E3×D3's gates pass only nominally. Over its endpoint windows `inj_spread`
   is 0.0006 and `inj_rank` 1.0000, clearing two standard errors because the
   seeds agree closely; in every seed the best validation epoch (0–3)
   precedes the first epoch at which `inj_spread` exceeds 0.01 (3–10). R3's
   E3×D3 accuracy therefore describes an injection that did not depend on
   the message. By epochs 60–69 the spread is 0.52 and the rank 1.50.
2. Validation accuracy over epochs 60–69 is matched within E2×D2 and E2×D1
   (−0.06, −0.3 SE) and within E3×D3 and E1×D1 (−0.04, −0.2 SE), and falls
   from epoch 20 in all six arms read here.
3. E3×D1's test accuracy does not fall over epochs 0–28 (+0.30), the only arm
   read here whose accuracy holds through early training.
4. Training accuracy over epochs 60–69 orders E2×D2 > E2×D1 > E1×D3 > E3×D3 >
   E1×D1 > E3×D1 (77.8, 75.1, 74.1, 70.9, 70.0, 68.4). E2×D2's
   training–validation gap at epoch 69 is 13.7 points, the largest in the
   grid read so far, and it is the only diagonal arm with no degenerate
   receiver.
5. `inj_rank` rises through training in both D3 arms and remains higher in
   E1×D3; both are still rising within the window (+0.25 and +0.11).

### 11.5 Part 4: the full 3×3 grid

| rule | contrast | difference | verdict |
|---|---|---|---|
| R5 | `msg_rank_ratio`, E2 and E3 − E1 at D2 and at D3 | −1.6 and +1.8 SE at D2; +0.3 and +0.1 SE at D3 | refuted |
| R6 | endpoint, every arm − E1×D1 | −0.23 to +0.24, largest +0.24 (+1.0 SE) at E3×D1; floor 0.41 | refuted |

Endpoint accuracy, in points:

| | D1 | D2 | D3 |
|---|---|---|---|
| E1 | 66.65 | 66.48 | 66.45 |
| E2 | 66.69 | 66.42 | 66.54 |
| E3 | 66.89 | 66.63 | 66.69 |

No conditional effect exceeds the floor. Of R6's five contrasts the largest
is E3×D3 − E1×D3, +0.24 (+1.9 SE).

Observations:

1. Every within-row difference from D1 is negative (−0.15 to −0.27) and
   every within-column difference of E3 from E1 is positive (+0.15 to
   +0.24), each within the floor.
2. Aggregated message rank is higher at E2 than at E1 in every column (+13.2,
   +12.8 and +6.1 SE), extending §11.2's observation 3 to D2 and D3.
3. The off-diagonal arms have the most degenerate routing in the grid: joint
   `routing_mode_share` is 0.93 in E2×D3 against 0.50 in E2×D1 (+4.1 SE), and
   0.91 in E3×D2 against 0.55 in E3×D1 (+2.4 SE).
4. In every D2 and D3 arm, spatial structure at the selected epoch is far
   below its final level: `inj_resid_frac` 0.03–0.16 against 0.26–0.29 at D2,
   and `inj_rank` 1.00–1.03 against 1.5–3.3 at D3. `inj_spread` is also near
   zero at the endpoint in E3×D3 and in three E3×D2 seeds.
5. Three orderings hold throughout the grid: sender message rank E2 > E1 > E3
   in every column; `inj_spread` D1 > D2 > D3 in every row and E2 > E1 > E3 in
   every column; and the training–validation gap at epoch 69 is largest in
   the E2 row (10.7–13.9 points) and smallest in the E3 row (2.7–6.0).

### 11.6 Part 5: each arm against itself

| rule | contrast | difference | verdict |
|---|---|---|---|
| R8 | `val_joint_answer_shift[kind=net]` against 0, each arm | −0.17 to +0.23 | refuted in every arm; negative beyond 2 SE in E1×D2, E3×D1, E3×D2 and E4×D1 |

No arm's `net` exceeds zero by two standard errors, the closest being E2×D3
(+0.23, t = 1.985). The four negative arms lie 2.6, 2.8, 2.8 and 3.5 standard
errors below zero, the outcome R8 names as the stronger: the channel is
harmful.

`net` in points, against the no-communication ensemble on each set:

| arm | validation, endpoint | validation, 60–69 | test, endpoint | test, 60–69 | training split, 60–69 |
|---|---|---|---|---|---|
| E1×D1 | −0.01 | −1.13 | +0.13 | −0.95 | +3.02 |
| E1×D2 | −0.14 | −1.52 | −0.04 | −1.31 | +5.47 |
| E1×D3 | +0.05 | −1.79 | −0.07 | −1.48 | +7.15 |
| E2×D1 | −0.07 | −2.02 | +0.17 | −1.61 | +8.16 |
| E2×D2 | −0.17 | −2.08 | −0.10 | −1.61 | +10.81 |
| E2×D3 | +0.23 | −1.95 | +0.02 | −1.46 | +10.72 |
| E3×D1 | −0.14 | −0.87 | +0.37 | −0.21 | +1.43 |
| E3×D2 | −0.13 | −1.05 | +0.11 | −1.02 | +4.30 |
| E3×D3 | +0.11 | −1.17 | +0.17 | −0.81 | +3.95 |
| E4×D1 | −0.10 | −0.61 | +0.16 | −0.06 | +0.75 |

The no-communication ensemble scores 66.86 on validation, 66.52 on test and
66.96 on the training split. Test `net` is test accuracy less the test
figure, recorded in each run as `baseline_avg_probs`; training-split `net` is
`joint_answer_shift[kind=net]`.

Observations:

1. Validation `pre_accuracy` is constant at 66.86 and `val_accuracy` equals
   `post_accuracy`, so selecting the endpoint by validation accuracy selects
   the epoch of largest validation `net`: R8's reading is biased upward.
2. The test set is not consulted in selection. On it, endpoint `net` lies
   between −0.10 and +0.37 in every arm; E3×D1 and E3×D3 exceed zero by 2.2
   standard errors, though E3×D1 reads −2.8 on validation. Seed standard
   errors omit the sampling error of a fixed evaluation set, about 0.25
   points for a paired `net` over 10,000 images when about 3 per cent change
   in each direction.
3. Over epochs 60–69 `net` is negative in every arm on both held-out sets,
   with `corrupted` exceeding `corrected`, and positive in every arm on the
   training split. Across the ten arms, training-split gain and test loss
   have a Spearman correlation of −0.95.
4. The training split's pre-accuracy matches validation's, so the backbones
   treat it as held-out data and the training-split gain is learned by the
   channel.
5. `agreement_post` is 54.6–56.6 against `agreement_pre` 67.8 in every arm:
   communication makes the specialists disagree more.

### 11.7 Exploratory: combiners without communication

To establish whether the channel's failure is specific to communication, the
three experts' outputs and their read-port (`layer3.0`) features were
computed once over the ensemble split, validation and test, and combiners were
fitted to the ensemble split with no communication. Regularisation strength
and, for minibatch fits, the stopping epoch were chosen on validation NLL;
test was not consulted. Feature-reading gates were trained with the channel's
optimiser and flip augmentation.

| combiner | parameters | test accuracy | test NLL |
|---|---|---|---|
| probability averaging (the no-communication ensemble) | 0 | 66.52 | 1.557 |
| one temperature shared by the three experts, T = 2.27 | 1 | 66.92 | 1.192 |
| temperature and fixed member weights | 4 | 67.32 | 1.184 |
| per-image gate on expert outputs, penalised | 19,459 | 67.27 | 1.185 |
| stacking (logistic regression on expert log-probabilities), penalised | 30,100 | 67.23 | 1.156 |
| per-image gate on pooled read-port features | 12,675 | 67.35 | 1.185 |
| per-image gate on spatial read-port features | 88,483 | 67.35 | 1.185 |

Observations:

1. The experts are overconfident: the NLL-optimal temperature is 2.27. With
   one fixed weight per expert, four parameters raise test accuracy by 0.80
   points, more than any arm of §11.5 achieves at its endpoint (0.43–0.90
   above each).
2. No per-image combiner improves on the four-parameter fit, whether it reads
   the experts' outputs or their read-port features. Unpenalised, each
   overfits as the channel does: the output gate reaches 72.3 on the
   ensemble split and 65.0 on test, stacking 83.0 and 61.2, and the spatial
   read-port gate 73.4 and 63.9 after 30 epochs.
3. On images where exactly one expert is right (2,493 in the ensemble split),
   no classifier predicts which one better than always choosing the best
   expert (49.4 per cent on validation), from outputs or from read-port
   features. Spatial read-port features are fitted perfectly on the ensemble
   split and score 36–42 per cent on held-out images.
4. Gate accuracy on test is flat from 2,500 to 20,000 training images
   (67.15–67.39), so the limitation is not the size of the split.
5. The three sets are alike: in each, all three experts are right on about
   49 per cent of images, none on about 24 per cent, and averaging is wrong
   while some expert is right on 8.8–9.5 per cent.

## 12. Analysis

Read against §6, explicitly naming which of R1 to R8 failed.

R1 and R4 are confirmed; R2, R3, R5, R6 and R8 are refuted; R7 finds E3×D1
and E4×D1 within the floor. The rules on message content and placement hold,
and every rule on accuracy fails.

- **What spatial access changes.** In every column, senders in the E2 arms
  occupy 1.5 to 1.9 times the directions of senders in the E1 arms, and the
  additional directions reach the decoders (R1; §11.5, observation 2).
  Placement information survives global average pooling (R4), and along the
  D3 column injection rank follows sender rank (3.34, 2.62 and 1.50 at E2,
  E1 and E3, against 6.65, 4.34 and 1.72). Messages in the E3 and E4 arms
  occupy 1.6–2.2 directions. Routing degeneracy is a per-seed outcome in
  every arm (R2).
- **What it does to accuracy.** No arm exceeds E1×D1 by the floor at the
  endpoint (R3, R6), but the endpoint precedes the spatial structure in every
  D2 and D3 arm (§11.5, observation 4). Over epochs 60–69, where that
  structure is developed, test accuracy is 0.36–0.81 points lower at D2 and
  D3 than at D1 in rows E1 and E3 (2.8–7.3 SE), level in row E2, and 0.50–0.66
  lower across the E2 row than at E1×D1. The refutations hold in either
  window.
- **Why it does not help.** At its best epoch the channel matches the
  no-communication ensemble on both held-out sets (R8; §11.6, observation 2).
  Thereafter it fits its training split and loses accuracy on held-out data
  (observation 3), most in the arms with the richest messages and injections:
  the E2 row, and D2 and D3 in row E1. E3×D1 and E4×D1 lead the grid over
  epochs 60–69 (+0.74 and +0.89 against E1×D1) by losing least: on test they
  lie at the no-communication ensemble (−0.21, −0.06), with the two smallest
  training-split gains in the grid and messages of about two directions.
  Study 01's `mlp` arm shows the same pattern, 0.45–1.09 points below the
  no-communication ensemble on validation over epochs 60–69.
- **Root cause.** The three experts share an architecture and training data
  and differ only by seed and snapshot, and which of them is right when they
  disagree cannot be predicted from their outputs or their read-port
  features (§11.7, observation 3). The only transferable gains are global, a
  temperature and fixed member weights, which the channel has no direct means
  to apply. Any per-image capacity, in the channel or in a gated mixture
  without communication, is therefore spent fitting the ensemble split, and
  more of it is spent where messages are richer. The group was selected for
  the gap between its oracle and averaged accuracy (76.4 against 66.9 on
  validation); that gap is real in hindsight but not learnable.
- **R5.** The rule's series does not answer its question. Whether the
  directions R1 adds reach the decoders is measured by
  `msg_rank[part=aggregated]`, which rises with `msg_rank[part=value]` from
  E1×D1 to E2×D1 (§11.2, observation 3); the ratio stays flat because both
  double. The ratio measures whether what arrives spans more than one
  sender's message, a different question. The verdict is reported as
  specified, but its stated reading, that content is produced and not
  transmitted, is contradicted by the aggregated rank.

## 13. Limitations

One expert group, one split, one port pair, n seeds, and whichever parameter
confounds survive into the comparisons actually made.

- One expert group, one split, one port pair and five seeds; differences
  below the 0.41-point floor are not resolvable.
- The endpoint precedes the spatial structure in every D2 and D3 arm, so R3
  and R6 test the treatments before they took effect. §12's end-of-training
  reading checks this but was not pre-registered.
- The endpoint is selected on the images R8 reads (§11.6, observation 1),
  and seed standard errors omit evaluation-set sampling error, about 0.25
  points for a paired `net` on test and 0.35 on validation.
- R5's series does not answer its question (§12).
- `msg_norm[part=injection]` reads D1 a factor of 32 low (§11.3, observation
  6); the logging is corrected for later studies.
- No pre-registered comparison exceeds the floor, so no conclusion rests on
  the parameter differences of §7.3. Of the end-of-training comparisons that
  do, E3×D1 and E4×D1 lead with 3.2–3.3 times E1×D1's parameters, but E2×D1,
  with 1.26 times, gains more on its training split than either (8.16
  against 1.43 and 0.75): the ordering does not follow parameter count.

## 14. Decisions & next steps

What this study changes about the project going forward.

- No level of spatial access is adopted. E1×D1 remains the reference arm: it
  is the least expensive (§7.3) and no arm exceeds it at the endpoint.
- The contingent arm of §4.2 is not triggered: E3×D1 exceeds E1×D1 by 0.24
  at the endpoint, below the floor.
- The binding constraint is the expert group, ahead of either port or the
  channel: exchangeable experts give per-image combination nothing to learn
  (§12). The approach is to be rethought in a future study, whose experts
  must disagree in ways that are predictable, for instance through different
  class subsets, augmentations or architectures, verified with §11.7's
  which-expert test before any channel is trained.
- The no-communication comparator is the calibrated, weighted average (67.32
  on test), not probability averaging; a channel's readout should carry the
  temperature and member weights itself rather than approximate them by
  perturbing the backbones.
- Channels are to be trained with effective regularisation and crop
  augmentation, with epoch selection on images other than those on which
  their effect is measured.
- If spatial access is revisited, the E2 arms are the candidates: the
  richest messages for the fewest added parameters (1.26 times at D1) and, at
  E2×D3, the richest placement.
