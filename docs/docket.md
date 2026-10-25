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
- learned weights on the experts' outputs
- learned combiner on the experts' features (matched in size to the channel)

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
  The default `pooled` encoder averages inside itself; `spatial` weighs
  positions (with a learned position tag) before summing them.
- Decoders get the received message and the receiving expert's features. The
  default `broadcast` adds one value per channel everywhere; `spatial` lets
  each position combine its own features with the message.
- Freeze everything after the write point, including each expert's output
  layer.
- Zero-initialize the write.
- Open: train on the ensemble's averaged prediction (as in v2), or on each
  expert's own loss (`--loss`)? Each expert's own accuracy is logged either way
  (`expert_accuracy`).

Measurement (docs/testbed/metrics.md):
- Every metric is logged at step 0, which is bagging, and after every epoch,
  test included. Nothing selects on test.
- Open: which epoch a run is scored at, chosen from joint_val. The candidates
  are the best joint_val epoch, or a window around it as v2 used, since a flat
  validation curve makes a single best epoch noisy. To be decided in the
  results stage.

First comparison (K = 1, one channel each, same size). Open between:
- (a) Same two points, opposite directions. Forward reads after stage 1 and
  writes after stage 2. Backward reads after stage 2, writes after stage 1,
  and re-runs stage 2.
- (b) Same message, different destination (recommended). Both read after
  stage 2. Forward writes after stage 3. Backward writes after stage 1 and
  re-runs stage 2.

Watch out: a forward exchange that writes into the output head is the
feature-combiner baseline. Each direction needs frozen processing after the
write point.

Extensions (deferred):
- Same-point "lateral" exchanges: one per point, CommNet / cross-stitch style.
- Overlapping forward channels.
- Deeper experts, to allow more forward exchanges.
- One-directional ablation: all experts send, one receives and answers.
  This mirrors Cache-to-Cache and Dead Weights, Live Signals.
