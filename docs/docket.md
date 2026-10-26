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
  - Run it through the study's runner, so it gets the same 30 epochs as the
    levels.
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
