# Communicative Ensembles of Frozen Networks

## 1. Summary

**Framing.** We recast ensembling as a cooperative multi-agent problem: each member of the ensemble is an agent that observes the same input, and the ensemble's prediction is aggregated from the agents' individual outputs. Without access to one another's internal state, this framing reduces to existing methods. Bagging aggregates independent predictions with fixed, uniform weights, while mixture-of-experts learns input-dependent weights through a central gate. Notably, coordination occurs only at end of the forward pass. This framing becomes substantive once agents can exchange information about their internal state and condition their decisions on what they receive, this same setting is addressed by learned communication in multi-agent reinforcement learning.

**Question.** Can agents that exchange messages through a learned channel outperform agents whose outputs are only aggregated?

**Method.** Each agent encodes late-layer, semantically rich features into a message. These messages are mixed across agents by an attention bus. Each agent injects its received message into its early layers and re-runs the intervening block. This repeats for K rounds before any logit is produced. Backbones are frozen; only the interface is trained.

**Status.** A full pipeline has been built on CIFAR-100 with a pool of 50 ResNet-110 snapshot experts. Initial experiments are under way.

**For discussion.** Architecture choices for the encoder/decoder and bus; whether the pool and task are the right testbed; whether freezing is a requirement of this method.

---

## 2. Claim and the frozen constraint

Standard ensembles combine members only at the output: logit averaging, probability averaging, majority vote, or a gate that selects one member. Members never see each other's internal state; each produces an answer alone.

**Claim:** members of an ensemble hold complementary information in their internal representations that output averaging cannot use, and a trained channel between frozen members can transfer enough of it to raise accuracy above averaging.

**Why frozen.** Backbones stay frozen; only the interface is trained:
- **Cheap composition:** pretrained networks are widely available; no retraining and no catastrophic forgetting.
- **Extensibility:** adding member N+1 costs one encoder/decoder pair plus a bus refit.
- **Clean measurement:** backbones cannot adapt, so any gain is attributable to the channel. Zeroing the messages recovers the non-communicating control exactly.

---

## 3. Related work and the gap

### Related work

- **Ensembling and MoE.** Combine at the output only. Snapshot ensembles (Huang et al. 2017) supply our expert pool. Deep mutual learning (Zhang et al. 2018) and codistillation (Anil et al. 2018) couple members at the loss during training; members are independent at inference.
- **Multi-agent communication.** RIAL/DIAL, CommNet, BiCNet, IC3Net, ATOC, TarMAC (Das et al. 2019). TarMAC's signature/query/value bus is the design used here. All train agents jointly from scratch, with a fixed agent set, on partial observations, from reward. None of those conditions hold for ensembling.
- **Model stitching** (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021). Two frozen models, one trained adapter. Pairwise, one-directional, sequential.
- **Dead Weights, Live Signals** (Armstrong et al. 2026). Frozen models in a feedforward DAG with trained adapters. Closest existing work. One-directional, single output node, no multi-round exchange.
- **Tools borrowed.** Linear probes (Alain & Bengio 2016); divided space-time attention (TimeSformer, Bertasius et al. 2021); attention over cached history (Transformer-XL, Dai et al. 2019); top-down feedback into early layers (Zamir et al. 2017).

### The gap

No ensembling method lets frozen pretrained members exchange internal representations bidirectionally over multiple rounds, with new members addable without retraining the rest.

Multi-agent communication has the exchange but not the frozen constraint. Stitching and DAG composition have the frozen constraint but not the exchange.

---

## 4. System

### Data and models

- CIFAR-100, CIFAR-native ResNets (depth 20–110), 32×32, no upscaling.
- Train set split class-stratified into backbone / ensemble / validation (25k / 20k / 5k). Official 10k test set untouched. All seeds derive from one master seed.

### Pipeline

1. **Expert pool:** 50 ResNet-110 snapshot experts (10 seeds × 5 snapshots) from cyclic-cosine-LR trajectories.
2. **Selection:** every k-subset of the pool scored on validation. Scores are cached, so group choice is an experimental variable.
3. **Joint training:** trains the interface over the frozen selected group.

### Architecture

- Each backbone tapped at an early and a late layer via forward hooks; backbone module unmodified.
- **Encoder:** reduces the late feature map, projects to query / key / value.
- **Bus:** single-head scaled dot-product attention over the member axis; each member attends over all members including itself.
- **Decoder:** MLP expanding the message to the early tap's shape, added as a residual. Final layer zero-initialized, so step 0 equals the control.
- **Readout:** probability average via logsumexp over per-member log-probs.
- Message geometry is swappable: pooled (one vector per member) or patch-wise (one message per spatial block, attention run per position).
- Recurrence is carried by the frozen backbone: round r's messages depend on round r−1's output.
- **Frozen:** backbone body, held in `eval()` so BatchNorm statistics cannot drift. **Trainable:** encoders, decoders, bus, and each backbone's final fc head (a deliberate relaxation).

### Baselines and ceilings

- **Baselines:** untrained logit averaging; untrained probability averaging; trained K=0 control (same interface, messages zeroed).
- **Ceilings**, computed from the selection scores: `bagging` (uniform weights), `soft_oracle` (per-example convex weights, solved by LP), `oracle` (per-example one-hot weights). `oracle − bagging` is the accuracy gap between an ideal router and plain averaging, and is the primary group selector.

**Success means:** communicative accuracy exceeds the K=0 control and both untrained averages, by more than seed variance.

### Diagnostics

- The trainer logs message norms, per-member answer shifts split into corrected and corrupted, flip rate, pairwise agreement (homogenization check), attention entropy and routing patterns, and per-component gradient norms. Tracked in Aim with replicate runs over seeds.
- Linear probes (`probing/`) ask three questions in order:
  1. do members hold complementary information (**existence**);
  2. does the channel transmit it (**delivery**);
  3. can the receiver use it (**usability**, answerable only end-to-end).

---

## 5. Results to date

*[To be filled in from the lab machine.]*

- **Accuracy:** communicative vs K=0 control vs logit average vs probability average —
- **Channel activity:** corrected / corrupted rates, flip rate, agreement before and after —
- **Probes:** existence gap, delivery gap —
- **Encoder ablation** —
- **Patch-wise vs pooled** —

---

## 6. Proposed work

Ordered by dependency. Each item names the question it answers.

1. **Baseline replicates.** K=0 vs K=1 at current defaults, replicated over seeds. Bootstrap CI on the existence-probe gap. *Is there an effect at K=1, and how much complementary information is there to transmit?*
2. **Encoder ablation** *(running)*. Four arms over one fixed group: linear Q/K/V; wider key and value; shared nonlinear trunk; same trunk without LayerNorm. *Is channel capacity limited by dimensionality or by expressiveness?*
3. **Patch-wise messages** *(running)*. Keep spatial structure instead of pooling. *Does pooling discard the information that needs to be sent?*
4. **Mixing-rule comparison** *(not started)*. Attention vs a learned fixed mixing matrix vs concatenation of all peer messages vs per-peer decoding with no mixing. *Does attention do anything a simpler rule does not?*
5. **Partial-unfreeze arm** *(not started)*. LoRA or adapters in the comm-zone, or unfreeze the last block. *Can a frozen receiver use a new input at all?* Separates delivery from usability.
6. **K-rounds sweep.** After 1–3. *Does a second round add anything once the first is working?*
7. **Temporal attention over message history** *(designed, not implemented)*. Caches each member's received messages and attends over them, factorized so communication and memory paths are separately ablatable. After 6. *Does removing the first-order Markov limit help?*
8. **Diversity scaling.** Select groups across the range of `oracle − bagging`. *Does benefit scale with the available gap?*
9. **Extensibility.** Train on N, freeze the interface, add member N+1, train only its interface, compare to full retraining. *Does the extensibility claim hold?*

---

## 7. Open questions

### Architecture

- **Encoder/decoder.** The encoder mean-pools the late grid to a vector and applies a linear map, so the message cannot carry more than the pooled features contain. Options: patch-wise messages; wider heads; a nonlinear trunk; cross-attention where the receiver queries the sender's full feature grid. Which of these is worth testing beyond the current ablation, and is the decoder (MLP to a residual at one early tap) the right injection mechanism?
- **Bus.** Attention was adopted from TarMAC because it is the best-developed mechanism for representation-level exchange, not because the settings match: here members see the same input, there is no bandwidth limit, few members, and a differentiable loss. Is attention the right tool at all, or would a learned fixed mixing matrix or concatenation do the same job? If attention, should it be patched (run per spatial position) or flattened (one attention over all patches of all members)?
- **Bus gradients.** Should gradients flow across members through the bus, with the bus learned jointly with all interfaces, or should each interface treat the bus as a fixed environment? Both are implementable; neither has been compared.
- **Freezing.** Freezing makes the measurement clean and the method cheap, and may also be what prevents the receiver from using the message: B<sub>i</sub> was trained on A<sub>i</sub>(x) and now receives A<sub>i</sub>(x) + D<sub>i</sub>(msg). A partial-unfreeze arm would separate delivery from usability, but weakens the frozen-composition claim. Is freezing a hypothesis under test or a requirement of the method?

### Testbed

- **Pool diversity.** Snapshot experts from one dataset may be too similar to hold complementary information. Are heterogeneous members (architecture, data, task) needed before the premise can be tested?
- **Task.** Output averaging is unusually strong for classification because the output is a low-dimensional simplex where independent errors cancel. This may be the setting least able to show a benefit. Settings where output averaging is ill-defined (segmentation, detection, sequence generation, differing output spaces) give representation-level combination more room.

### Scope

- **Language.** The mechanism is architecture-agnostic in principle: tap two layers, encode, mix, inject. LM ensembling baselines are weak (logit averaging needs a shared tokenizer), so there is more to improve on. But K re-runs conflict with KV caching, and messages would be per-token or pooled. Is this worth scoping now or later?
- **Stopping rule.** What result would justify continuing, and what would end the line? RL raises the ceiling but changes the problem, so it is not a fallback for a null result here.

---

## Appendix: changes from the earlier design

- ImageNet-100 with class-disjoint specialists → CIFAR-100 with a snapshot pool over the full label space. The earlier CIFAR objection (upscaling caps accuracy) applied only to ImageNet-pretrained backbones.
- Scope narrowed from "stage toward RL composition" to an ensembling method with RL as a possible extension.
- Baselines B1–B5 replaced by logit average, probability average, trained K=0 control, and the three computed ceilings.
- BiRNN/BiLSTM bus and Q=K self-attention replaced by the TarMAC bus.
- Backbone fc heads are trainable; earlier notes said fully frozen.
- RL-specific mitigations (DIAL-style noise, entropy-scheduled reward) and early PPO experiments dropped.
- Axial attention no longer cited as support for factorized attention; TimeSformer ranks it worst among time-aware schemes.
