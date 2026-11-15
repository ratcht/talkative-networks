# v3 docket

## Expert generation: different expertise, same input

Every expert sees the full image. Diversity comes from what an expert has
learned, never from limiting what it sees. Seed-only copies of one model on one
dataset left no usable room for any combiner in the pilot. That is also the
setting where the ensembling literature finds plain averaging hard to beat.

1. **Class-skewed experts (ACTIVE).** Each expert trains on the shared 30k
   images, but its batches are drawn by its own class mix. The mix is rolled
   from a Dirichlet(α) over the 10 classes, using the expert's own seed.
   - α dial: large = equal mix (today's seed-only setup), ~1 = lopsided,
     ~0.1 = near-specialists. **Default α = 1.** In simulation, each
     expert's favorite class gets about 3× normal emphasis, two experts' mixes
     differ by about half, and a pair already covers about 86% of classes at
     half the normal rate or more. Keep α fixed when varying M (2–10).
   - Normal test set.
   - Strong specialists cap at about 10 distinct experts. At α ≈ 1 every
     roll differs, so groups of 10+ still work.

   *Precedents: Hinton et al. 2015 specialists; ZipIt; Dirichlet label skew
   from federated learning (Hsu et al. 2019).*
2. **Domain experts (DEFERRED until M > 10).** The same roll, but over a
   fixed set of domains (e.g. 12 rotations), with its own α and seed. It can
   be combined with class skew as a second dial. It turns the test set into
   rotated images, so choose it per study, not per M.
   *Precedents: Meta-DMoE; DomainBed's RotatedMNIST.*
3. **Heterogeneous architectures (DEFERRED).** For example MLP / LeNet-5 /
   3-stage CNN, all on the same data.
   *Precedents: Dead Weights, Live Signals; Mixture-of-Agents.*

Training recipe: a fixed 10 epochs. These are deliberately sub-optimal
experts; the val-loss stopping point is about 16 epochs for LeNet-5 and about
11 for the 3-stage CNN.

Baselines for every arm:
- bagging
- each expert's output layer re-fit on the joint data, then bagging
- learned weights on the experts' outputs. This is the soft-gated MoE's gate
  with nothing to read; not built.
- learned combiner on the experts' features (matched in size to the channel)
- **soft-gated MoE** (user asked for it, 2026-10-25; built, testbed/moe.py,
  docs/testbed/moe.md):
  - The gate reads every expert's whole map at one point through r = 8
    templates each, as the linear_taper encoder does. A GELU and a linear
    layer then score each expert, and the softmax weighs their answers.
  - It starts as bagging and changes no expert's own answer.
  - It is 13.9k weights reading stage3 and 37.7k reading stage2, for all
    three experts.
  - Run it through the study's runner (`--moe READ`), so it gets the same
    30 epochs as the levels.
  - DECISION (user, 2026-10-25): where the gate reads is an ablation, like
    the channel's exchange points.
  - Smoke run, trial seed 0: bagging 0.8849, MoE 0.8933 reading stage3 and
    0.8923 reading stage2. It trained steadily, and the largest weight
    averaged 0.89.
  - Not built: a gate that also reads the experts' answer probabilities.

## Communication architecture

- **Exchange:** read each expert at a read point → channel → write into each
  expert at a write point.
- **Forward** (write point strictly after the read point; decided
  2026-10-23): one pass.
  - Each exchange uses two points, and the next exchange may only read after
    the experts have processed the previous message.
  - So n points allow at most n/2 exchanges. On the 3-stage CNN that is one
    meaningful exchange; a second would write into the last linear layer.
  - K > 1 is a backward-only axis for now.
- **Backward** (write point before the read point): re-run the layers in
  between. K exchanges means K rounds.
- Only exchanges with expert computation between them count toward K. Two
  mixing steps back to back are just a bigger channel.
- Compare at equal K and equal channel size, and report compute (backward
  re-runs layers). K = 0 with a zero-initialized write is the plain ensemble.

Defaults:
- Encoders get the raw features at the read point (full spatial access).
  Each encoder sends one message per expert for now; the code allows more.
- Decoders get the received message and the receiving expert's features.
- **Complexity ladder** (docs/studies/channel-ladder.md, code in
  `studies/channel_ladder/`). No level pools or broadcasts. Level 1, the
  default, uses `linear_taper` encoders and decoders: each expert reads its whole map through r learned templates and
  writes a mix of r learned patterns. Level `2-encoder` adds a GELU in the
  encoder, `2-decoder` one in the decoder, and `3` both. The weights stay the
  same across levels. The study's runner tags every run with its level, and
  every run logs its parts by name.
  - DECISION (user, 2026-10-25): encoder and decoder are two linear layers
    through a narrow width, with **r = 8**. That is at most 0.4× an expert
    per expert on cnn3. Dense maps were 1.7–4.2×.
  - DECISION (user, 2026-10-25): part names say what each part is, not a
    generic class like `linear` or `mlp`, so a later, more specific variant
    gets a name of its own instead of forcing a rename. The names are
    `linear_taper` and `linear_taper_gelu` (user), and `mean` stays as it is.
  - DECISION (user, 2026-10-25): a `--mixer mean` control, where every
    expert hears the plain average of all the messages. It sits beside
    `self`: full − mean is what choosing whom to listen to adds, and
    mean − self what hearing the others adds.
  - DECISION (user, 2026-10-25): the decoder runs message → r → injection.
  - DECISION (user, 2026-10-25): all four combinations of the GELUs, as the
    levels 1, 2-encoder, 2-decoder and 3, rather than one order.
  - DECISION (user, 2026-10-25): the ladder stops at level 3 for now. If the
    GELU levels add nothing at r = 8, rerun them at r = 32 before concluding
    the nonlinearity doesn't help, since r is also the GELU's width.
  - Any mixer runs at any level. Which controls run at which levels is the
    study's design, not the ladder's (user, 2026-10-25); the proposal was
    `self` and `mean` at every level.
  - `pooled`, `spatial` and `broadcast` stay available as variants outside
    the ladder.
  - Phase 1 (docs/studies/channel-ladder.md), 2026-10-26. 225 runs: 15 arms
    × trial seeds 0–14, all from commit c28ccdf with a clean tree, every arm
    on the same experts per trial seed. Differences are paired by trial seed,
    in accuracy points, with 95% ranges.
    - Level 1 − `self` is negative at all six exchange pairs, from −0.04 to
      −0.10. Pooled over the pairs it is −0.07 ± 0.03, and level 1 is ahead
      on only 2 of 15 trial seeds. So attention hurts a little, consistently.
    - The rule picks forward **stage2 → stage3** (−0.040) and backward
      **stage3 → stage1** (−0.061). Both are ties with the runners-up,
      stage1 → stage2 (+0.003 ± 0.16 apart) and stage3 → stage2 (+0.006 ±
      0.08 apart). It picks the least harmful pair, not where communication
      helps.
    - The MoE's pick is **stage3**, at 89.83. It beats stage2 by 0.11 ± 0.11
      (12 of 15 seeds) and stage1 by 0.21 ± 0.12 (11 of 15).
    - Every level 1 arm beats the MoE at stage3 by 0.4–0.8, on 11–14 of 15
      trial seeds. `self` beats it by as much, so the edge is each expert's
      own add-on, not listening. That is v2's finding again: the fine-tuning
      carried the gain.
    - Exploratory: backward stage2 → stage1 beat forward stage1 → stage2 by
      +0.14 ± 0.12.
    - Run length: the mean curves are flat from about epoch 10, and the
      mean of epochs 26–30 is within 0.1 of epochs 16–20, either way. 48 of
      225 best epochs landed at 26–30, about what a flat curve gives by chance
      (1 in 5). 30 epochs stays.
  - Phase 2, 2026-10-28: 2a, 2b and 2c, run in sequence, each its own MLflow
    experiment. 300 runs, 20 arms × trial seeds 0–14, all finished, on the
    same experts per trial seed as phase 1.
    - Code: 2a, 2b and the first five arms of 2c record commit c28ccdf with
      `git_dirty=true`, because they were launched before the `--phase`
      change was committed. The uncommitted changes were the commit after
      it, c16e895, which touches only the runner's experiment and tags, its
      tests and the docs. The last three arms of 2c record c16e895, clean.
    - Read as in phase 1: differences paired by trial seed, in accuracy
      points, with 95% ranges and the trial seeds where the first arm is
      ahead.
    - **Level − `self`, forward stage2 → stage3:** −0.09 (2-encoder), −0.12
      (2-decoder) and −0.10 (3), on 3 of 15 trial seeds each. Pooled over
      the GELU levels it is −0.10 ± 0.06 (1 of 15). The same picking epochs
      by joint_val loss. Attention is slightly worse than `self`, as at level 1.
    - **Level − `self`, backward stage3 → stage1:** +0.05, +0.01 and −0.03;
      pooled +0.01 ± 0.08 (8 of 15). No difference.
    - **Level − MoE:** every level ahead, by 0.53–0.54 forward (11–12 of 15)
      and 0.72–0.78 backward (13–14 of 15), as at level 1.
    - The GELU levels add nothing: no GELU level differs from level 1 by more
      than 0.05 with attention, either direction. So the r = 32 rule applies.
    - Exploratory, the `mean` control (2c), pooled over the four levels:
      - forward, `mean` − `self` is +0.00 ± 0.04, and attention − `mean`
        −0.09 ± 0.07 (4 of 15);
      - backward, `mean` − `self` is **+0.12 ± 0.06 (13 of 15; 15 of 15
        picking by loss)**, and attention − `mean` is **−0.12 ± 0.05 (1 of
        15)**. Backward level 1 with `mean`, 90.80, is the best arm run so
        far.
      - So backward, hearing the others' plain average helps a little, and
        attention gives that back: choosing whom to listen to costs about as
        much as hearing the others adds.
    - Exploratory: backward beats forward at every level, by 0.18–0.25 (11–14
      of 15).
  - Phase 3, the width, 2026-10-28/29: levels 1 and 3 with `attention`,
    `self` and `mean` at both pairs, at r = 16, 24 and 32, each its own
    experiment (`channel-ladder-phase3-r16`, ...). 540 runs, all finished, all
    from commit 9ab5621 with a clean tree, on the same experts per trial seed
    as phases 1 and 2. Read beside r = 8 from phases 1 and 2, as before.
    - Stable at lr 1e-3 at every width: no NaN, and joint_val wobbles only a
      little more as r grows (backward, a median range of 0.72 points after
      epoch 5 at r = 8 and 0.88 at r = 32). Two forward level 1 runs (r = 16
      seed 2, r = 24 seed 12) dipped about 3 points for one epoch and
      recovered at the next.
      - Counting every single-epoch dip of more than 0.5 below bagging after
        epoch 5 gives ten runs, all forward level 1 with `attention` or
        `mean` at r ≥ 16, on trial seeds 2 and 12. None are `self` runs and
        none are level 3. Each is one epoch between 16 and 28, and the
        5-epoch median absorbs it.
    - **Level − `self` stays at or below zero at every width.** Forward,
      level 1 is −0.04, −0.08, −0.08 and −0.15 at r = 8, 16, 24 and 32 (1 of
      15 at r = 32), and level 3 is −0.08 to −0.11. Backward, both are within
      ±0.10 of zero at every width, under either way of picking epochs.
    - **The GELU doesn't help at any width either.** Level 3 − level 1 with
      attention is −0.03 to +0.10 forward and −0.05 to −0.12 backward. With
      `self`, backward, the GELU costs 0.13–0.16 at r ≥ 16 (3–5 of 15). The
      r = 32 rule has run, and the nonlinearity, as the ladder places it, adds
      nothing from r = 8 to 32.
    - **Width helps backward, communication or not.** Level 1 `self` gains
      0.09, 0.13 and 0.20 at r = 16, 24 and 32 over r = 8 (14 of 15 at
      r = 32), and level 1 with attention about 0.15 at each. Forward, width
      changes nothing.
    - Level − MoE grows backward with the width, from 0.76 to 0.93, but the
      channel grows with it (to 9× the gate), and `self` grows as much. It
      isn't communication.
      - Added 2026-11-02: the gate at r = 16, 24 and 32 (`moe-stage3-r16`,
        `-r24`, `-r32`, 45 runs in phase 3's experiments, laptop, from
        uncommitted code saved in `cache/channel_ladder_phase3_gate/`; the
        gate's own code is unchanged and every trial seed's `pool.digest`
        matches). Wider gates gain nothing: +0.05, +0.02 and +0.04 over r = 8
        (± 0.07–0.10), and +0.03 to +0.04 on test over all 30 epochs, while
        their training accuracy rises from 92.9 to 93.9. At equal r, channel −
        gate, levels 1 and 3 and all mixers pooled, is +0.50 to +0.59 forward
        and +0.81 to +0.90 backward (15 of 15 backward at r ≥ 16).
    - Exploratory, backward `mean` − `self` (levels 1 and 3 pooled) is +0.12
      at r = 8 and 16 under both rules. At r = 24 and 32 it is +0.02 and
      +0.01 picking epochs by joint_val accuracy, but +0.13 and +0.16 (12–13
      of 15) picking by joint_val loss. So the lead is weaker than phase 2
      suggested, and at the larger widths it depends on the rule.
      Attention − `mean` shrinks from −0.16 at r = 8 to about −0.05.
    - The best arms are backward level 1 with `mean` at r = 16 and 32
      (90.88) and with `self` at r = 32 (90.86).
  - Phase 4, the size of the queries and keys, 2026-11-01. 90 runs, 6 arms ×
    trial seeds 0–14, forward stage2 → stage3 at r = 8, all finished, no NaN,
    on the same experts per trial seed as phase 1 (checked by `pool.digest` on
    every run). Its rule, fixed before any run, is in
    docs/studies/channel-ladder.md.
    - Code and machine: the lab machine, one worker per GPU, 14 minutes.
      Commit 9ab5621 with `git_dirty=true`. The exact code is in
      `cache/channel_ladder_phase4/`: `code_head.txt`, `code_diff.patch` and
      `code_untracked.tgz`.
    - **Size is what differs.** Queries and keys are about 31 at level 1 and
      49 at `2-decoder`, against 6 at `2-encoder` and 7.5 at level 3. Each
      linear encoder is ahead of each GELU one on 15 of 15. Across the 60
      attention runs, the rank correlation of entropy with query × key is
      −0.95, the same as it was with message size.
    - **Taking size away ends the wire.** Level 1 with `cosine_attention`
      minus level 1 with `attention`: entropy +0.81 ± 0.02 (15 of 15), from
      0.006 to 0.82; mode share −0.67 ± 0.10 (0 of 15), from 0.88 to 0.21;
      `attn_top1` from 0.997 to 0.63.
    - **It reaches the GELU arms and passes them.** They read entropy 0.51 and
      0.39, and mode share 0.63 and 0.72. How far past depends on the length
      the mixer rescales to.
    - **Accuracy holds.** Test +0.01 ± 0.10 (5 of 15). At epochs 21–30, +0.04
      ± 0.11 on test and −0.05 ± 0.12 on joint_val; over all 30 epochs, +0.05
      ± 0.06 and +0.00 ± 0.08. The loss is a little worse on both, +0.007 ±
      0.006 on test and +0.007 ± 0.005 on joint_val.
    - **So by the rule, the GELU's softer routing is scale**, and scale is
      nearly all of it. With size gone from both, `2-encoder` − level 1 is
      +0.06 ± 0.03 in entropy (14 of 15), down from +0.51 ± 0.13, and −0.04 ±
      0.02 in `attn_top1`, down from −0.24 ± 0.07. Mode share flips, from
      −0.26 ± 0.18 to +0.14 ± 0.10 (13 of 15). This comparison doesn't depend
      on the length, so a little is left that isn't size.
    - Exploratory:
      - **Size accounts for how sharp each image's weights are.** Give every
        query and key a random direction and the logged size, with nothing
        fitted, and the entropy comes out roughly right: 0.87 for
        `cosine_attention` (seen 0.82 and 0.88), 0.48 for `2-encoder` (0.51),
        0.30 for level 3 (0.39), 0.02 for level 1 (0.006) and 0.01 for
        `2-decoder` (0.002). Entropy can't say who wins, though, so this says
        nothing about the routing's choices.
      - **Softer routing isn't better routing.** Without size, training leaves
        routing about where the untrained channel has it at step 0. Level 1
        goes from entropy 0.88 to 0.82, `attn_top1` 0.59 to 0.63 and mode
        share 0.26 to 0.21 (−0.05 ± 0.10), and `2-encoder` doesn't move. The
        routing isn't random: mode share is 0.20 against 0.04 for a fresh draw
        on every image, and `route_flip_rate` 0.18 against 0.67. Each receiver
        leans towards one sender, at 44%, but it already leaned at step 0, at
        48%, and only 16 of 45 receivers keep that favourite (15 by chance).
        In no arm is the sender heard more often right than at step 0:
        `routed_correct/all` is 84.3 to 85.6 (+1.3 ± 1.7) with
        `cosine_attention` and 84.5 to 84.1 (−0.4 ± 1.5) with `attention`.
      - **Starved gradient wasn't what kept routing from improving.** At epoch
        30, level 1's query and key get 9–11× the gradient with
        `cosine_attention` (0.09 against 0.01), with the result above.
      - **What softer routing does change.** Receivers hear more
        (`msg_rank_aggregated` 4.7 to 7.7, 15 of 15), and routes move more
        between epochs (`route_flip_rate` 0.04 to 0.22, 14 of 15).
      - **The deficit to `self` stays.** Over all 30 epochs on joint_val,
        `cosine_attention` − `self` is −0.11 ± 0.04 and `attention` − `self`
        −0.12 ± 0.06. On test they are −0.06 ± 0.04 and −0.11 ± 0.05. So the
        wire isn't what costs attention. `cosine_attention` − `mean` is −0.04 ±
        0.04 on test and −0.04 ± 0.03 on joint_val. `self` and `mean` here are
        the laptop's runs.
      - **What grows is the readings, not the heads.** In the first epoch,
        queries and keys grow about ninefold in every arm, while the steps
        logged for their heads allow at most 1.3×. Value grows with them, query
        ÷ value staying near 0.5, the ratio of untrained heads, and `self`'s
        value grows fivefold though its query and key heads get no gradient.
        After epoch 1 queries and keys stay at 15 with `cosine_attention` and
        reach 37 with `attention`; at `2-encoder` both mixers end near 5. That
        later growth is in senders nobody hears: at level 1 they go from 17 to
        65, while well-heard senders stay near 14 (the least heard is the
        larger on 15 of 15). What drives the readings is open. Rescaling
        queries and keys doesn't make routing
        independent of the readings' size: the heads have biases, so at step 0
        readings ×5 changes the top sender on 28% of image-receiver pairs,
        though only 1.5% after epoch 1 (one trial seed). Phase 5 found the
        biases aren't how the readings grow.
      - **Who each receiver hears.** At level 1, a receiver gives its three
        senders 94%, 6% and 0.3% with `attention`, and 44%, 32% and 24% with
        `cosine_attention`. All three receivers share a favourite on only 3 of
        15 trial seeds. At the linear encoders the favourite is almost never
        the sender with the largest key: 1 of 45 receivers at level 1 and 3 of
        45 at `2-decoder`, against 15 by chance, though keys differ about 6×
        between experts. Unexplained.
      - **The machine.** The four `attention` arms on the lab minus the same
        arms on the laptop are −0.01 ± 0.10, +0.04 ± 0.09, −0.07 ± 0.13 and
        −0.08 ± 0.10, so results from the two compare. Each pair starts from
        the same step 0 and parts within the first epoch. One trial seed's
        difference then has a spread of 0.16–0.23 with nothing changed but
        the GPU, about the same as level 1 between its two mixers (0.17). So
        most of a paired difference's noise is the training path, and only
        more trial seeds reduce it.
    - Proposed next, not run (2026-11-02): **do `attention`'s choices carry
      anything?** The report's finding 5 used only whether the sender heard is
      right, and the user pointed out that a partner can help without being
      right. An exploratory pass over this phase's `attn_heard` is in
      `cache/channel_ladder_phase4/analysis/complementarity/`, kept out of the
      findings until something firmer exists. Complementarity in features
      couldn't be measured there, and the lean towards partners whose answers
      complement the receiver is weak. Firmer tests, on exact lab replays at
      test time:
      - Give each image another image's routing weights: do choices per image
        matter? Forward `attention` is a fixed wire, so this bites only where
        routing is soft.
      - Swap each receiver's weights between its two partners: does its
        standing preference matter? This moves the decoder off what it
        trained on, so a drop alone wouldn't show the choice was informed.
  - Phase 5, why queries and keys grow, 2026-11-02. 135 runs, 9 arms × trial
    seeds 0–14, level 1, forward stage2 → stage3 at r = 8, all finished, no
    NaN, on the same experts per trial seed as phase 4. Its rule, fixed before
    any run, is in docs/studies/channel-ladder.md. By that rule **neither the
    jump nor the later growth is isolated**: of its six tests only 4 held.
    - Code and machine: the lab machine, one worker per GPU, 20 minutes. Commit
      9ab5621 with `git_dirty=true`. The exact code and the analysis are in
      `cache/channel_ladder_phase5/`.
    - The new logging leaves training alone: the `attention` and
      `cosine_attention` references match phase 4's test curves at every epoch
      on 15 of 15 trial seeds. So on one machine a run repeats exactly, and
      the rerun noise above is all between GPUs.
    - The heads don't grow. Under `self` their weights don't change at all,
      while the query goes from 1.75 to 9.3 in epoch 1.
    - **1 fails: the jump survives centring.** It stays at ×5.3 with `self` and
      ×7.0 with `attention`, below ×2 on 0 of 15, against ×6.0 and ×10 at the
      references. The share of the readings common to every image goes from
      0.75 to 0.02, and the part that differs between images jumps just as far
      either way (×5.4 with `self`). So the average image isn't behind it.
    - **2 fails: the wire weakens but stays sharp.** Centred `attention` has an
      entropy of 0.24 against the 0.5 asked (reference 0.006). See below for
      what did change.
    - **3 fails: the read layer has to learn.** Frozen readings are −0.21 ±
      0.15 on test (3 of 15) and −0.21 ± 0.13 on joint_val at epochs 21–30,
      though level with the reference at epoch 1 (+0.01 ± 0.34), and they make
      no wire (entropy 1.03). This tests whether the read layer must train,
      not whether readings must be large.
    - **4 holds, with a caveat.** With routing's gradient cut at the readings,
      each run's least-heard sender goes ×0.85 from epoch 1 to 30, against
      ×3.23 at the reference, with accuracy at +0.01 ± 0.08. But the cut also
      stops the wire (entropy 0.27, mode share 0.30), so that sender is heard
      14%, not 0%. It shows the cut stops the runaway, not that routing pushes
      unheard senders in particular.
    - **5 fails narrowly: the brake falls short.** With its value gradient cut,
      expert 0 ends at ×2 its larger partner or more, and still heard, on 10 of
      15, median ×2.24. Over the run the median is ×1.9, ×3.3, ×4.1, ×3.0 and
      ×2.2 at epochs 1, 5, 10, 20 and 30. It also stops being heard: 8%,
      against 33% at the reference, so this isn't the clean case of a sender
      that is heard but unbraked.
    - **6: the biases aren't the mechanism.** With no head biases routing can't
      depend on the size of the readings at all, and expert 0 runs away the
      same: ×2.18, 8 of 15, and within 15% of the arm with biases at every
      epoch read. The test's "yes" needed 5 to hold, so it has none.
    - Exploratory:
      - **Size sets the sharpness.** Over 300 (run, epoch) points of the four
        arms with plain `attention`, the rank correlation of entropy with
        query × key is −0.98.
      - **Who wins is a different matter.** Centring takes mode share from 0.88
        to 0.25 while the top weight stays at 0.90: sharp, but no longer the
        same wiring on every image. Across arms mode share stays near 0.25
        until queries are over 80% common to every image, then climbs (0.41,
        then 0.73 above 90%). But frozen readings and untrained channels are
        80% common at a mode share of 0.23–0.28, so a large common part looks
        necessary, not sufficient.
      - **The extra size isn't needed.** Readings end at 11 with routing's
        gradient cut, against 34, at the same accuracy.
      - **Routing's extra push is in the common part.** At epoch 1 that part
        grows ×12 with `attention` and ×6.5 with `self`, while the part that
        differs between images grows ×6.3 and ×5.4.
      - **Centring costs later on.** The loss at epochs 26–30 is worse by 0.05
        on test and 0.04 on joint_val with `attention`, and by 0.04 with
        `self`. Accuracy at epochs 21–30 is −0.20 ± 0.13 and −0.24 ± 0.18 with
        `attention`, and −0.02 ± 0.07 and −0.16 ± 0.08 with `self`.
    - Proposed next, not run: phase 5's fallback for the jump, `self` with SGD
      and with a larger `read` init. An untested candidate is that random
      templates start nearly at right angles to stage 2's features, so a small
      turn raises every reading several times over while the weights barely
      grow (×1.3 in epoch 1). Centring can't prevent that. The check is each
      template's cosine with the features at step 0 and at epoch 1.
- Freeze everything after the write point, including each expert's output
  layer.
- Zero-initialize the write.
- DECISION (user, 2026-10-25): train on the joint loss, the ensemble's
  averaged prediction (`--loss ensemble`, as in v2), for now. Each expert's own
  loss (`--loss per_expert`), alone or added to the joint loss, is a separate
  study. Each expert's own accuracy is logged either way (`expert_accuracy`).
- DECISION (user, 2026-10-25): the same learning rate for every arm, 1e-3,
  for as long as it stays stable. It is never tuned per arm, because then a
  comparison between arms would partly measure whose learning rate suited it
  better. If some arm turns out unstable at 1e-3, lower it for every arm.
  - Smoke run, trial seed 0 (2026-10-25): levels 1–3, plus level 1's `self`
    and `mean` controls, forward stage2 → stage3 and backward stage2 →
    stage1.
  - Every arm trained steadily at 1e-3. joint_val stayed at 0.90–0.91 and
    the injection grew smoothly.
  - Joint accuracy beat joint_val by about 1 point forward and up to about 3
    points backward (the `mean` control's last epoch).
  - Dense maps had been unstable at 1e-3 backward. At 3e-4 they trained
    steadily, but with a gap of about 4 points.
  - One trial seed, not a result.

Measurement (docs/testbed/metrics.md):
- Every metric is logged at step 0, which is bagging, and after every epoch,
  test included. Nothing selects on test.
- DECISION (user, 2026-10-25): each run is boiled down to one number per
  metric (testbed/results.py). Take the trained epoch with the highest
  joint_val accuracy and the 5 epochs around it, and use the median of the
  metric (test accuracy, say) over them. Arms are then compared by averaging
  over trial seeds. The plots in MLflow stay the way to look at runs; the
  number is only for comparing them.
  - Details settled when building it: joint_val only picks the epochs, and
    the median is of test. Near either end of the run the window shifts
    inward, so it always holds 5 trained epochs, and step 0, bagging, is never
    in it. The earliest epoch wins a tie.
  - Runs are 30 epochs, the channel ladder runner's default.
  - Check (trial seed 0, level 1 and its `self` control, both directions, 30
    epochs, 2026-10-25):
    - joint_val is flat from about epoch 5, at 0.905–0.915;
    - the best epoch landed at 13, 28, 14 and 18;
    - forward was still creeping up at epoch 30, so if more trial seeds show
      that too, the runs need to be longer;
    - in the 10-epoch smoke runs, 4 of 10 arms had their best epoch at the
      last one.
  - The joint split is overfit by epoch 30: joint beat joint_val by 2–3
    points forward and almost 5 backward, while joint_val stayed flat.
  - An arm then takes about 30 s forward and 65 s backward on the laptop.
- Trial seeds. Pilot, 2026-10-25: level 1 and its `self` control, forward
  stage2 → stage3 and backward stage2 → stage1, trial seeds 0–9, 30 epochs,
  each run read by the rule above. Spreads are per trial seed, in accuracy
  points:
  - Each arm's own number moves by 0.20–0.33 between trial seeds.
  - Paired differences move by 0.21 (forward, level 1 − self), 0.22
    (backward, level 1 − self) and 0.27 (level 1, forward − backward).
    Unpaired, the same differences would move by 0.37–0.43, so pairing cuts
    the seeds needed by about 3×.
  - Pinning an average difference to ±0.1 then takes 17, 20 and 28 trial
    seeds; ±0.2 takes 5–7, and ±0.05 takes 68–109.
  - Ten seeds give only a rough spread: the true value is 0.69–1.83× the
    estimate (95%), so n is a ballpark.
  - Aiming for ±0.1, since v2's communication effect was about 0.2, would
    take about 30 trial seeds, set by the noisiest pair.
  - DECISION (user, 2026-10-25): 15 trial seeds for now, to keep iteration
    fast. At the pilot's spreads, that pins level-vs-control differences to
    about ±0.11 and forward-vs-backward to about ±0.14.
  - Bagging moves by 2.4 points between trial seeds, since the experts'
    class skews differ, but every trained arm lands at 90.4–90.7. The gain
    over bagging, +2.7 forward and +2.9 backward, tracks how weak bagging was
    (correlation −0.99) and moves by 2.2. So compare trained arms with each
    other, not with bagging.
  - Pilot hints only: level 1 − self was −0.05 ± 0.15 forward and
    −0.09 ± 0.16 backward, and forward − backward was −0.21 ± 0.19.
  - Running: two runs logging to `mlflow.db` at once worked, with all 40
    finished. On the laptop's single GPU, though, two at once halved each
    one's speed, so run one per GPU.

First comparison (K = 1, one channel each, same size). To be settled by an
ablation during the study (user, 2026-10-25). Open between:
- (a) Same two points, opposite directions. Forward reads after stage 1 and
  writes after stage 2. Backward reads after stage 2, writes after stage 1,
  and re-runs stage 2.
- (b) Same message, different destination (recommended). Both read after
  stage 2. Forward writes after stage 3. Backward writes after stage 1 and
  re-runs stage 2.
- Input for the choice:
  - With the ladder's parts, a forward exchange and its backward mirror (the
    same two points, swapped) are always the same size. So (a) compares the
    directions at equal size and equal r.
  - Under (b), backward is 2.1× the size of forward at equal r. Equal size
    needs r = 16 forward against r = 8 backward, which lets the forward
    experts say twice as much (docs/testbed/channel.md §5).

Watch out: a forward exchange that writes into the output head is the
feature-combiner baseline. Each direction needs frozen processing after the
write point.

Extensions (deferred):
- Same-point "lateral" exchanges: one per point, CommNet / cross-stitch style.
- Overlapping forward channels.
- Deeper experts, to allow more forward exchanges.
- One-directional ablation: all experts send, one receives and answers.
  This mirrors Cache-to-Cache and Dead Weights, Live Signals.
