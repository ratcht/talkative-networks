# Testbed metrics

This page defines every metric the v3 testbed logs: what it measures, how it is
computed, and how to read it. The code is in `testbed/metrics/`. Every MLflow
key is listed in `testbed/metrics/registry.py` with a one-line summary, and the
same list is the description of each MLflow experiment, so it can be read from
the MLflow UI.

Most of these metrics come from the v2 testbed. §9 lists what changed.

## 1. When metrics are measured

A run logs every metric at **step 0**, before any training, and again after
every epoch, at steps $1, \dots, E$. At step 0 the channel is untrained and its
decoders inject exactly zero, so the ensemble is plain bagging. Every effect of
communication is a change from step 0.

Each metric is measured on one or more of three splits:

| split | measured | contents |
|---|---|---|
| `train` | during each epoch, from step 1 | accumulated over the epoch's batches, with the channel as it was at each batch |
| `joint_val` | at step 0 and after each epoch | a full pass, including every exchange's metrics |
| `test` | at step 0 and after each epoch | a full pass |

Nothing selects on test. Which epoch a run is scored at is decided afterwards,
from joint_val. That rule is still open (see `docs/docket.md`), and logging
test at every epoch is what lets it be chosen later.

The training-dynamics metrics are averaged over the optimiser steps of each
epoch. Everything else is computed from passes the run makes anyway, so no
metric costs an extra forward pass.

## 2. Reading the keys

Every MLflow key is a path. It starts with one of three things:

- **The split**, for metrics about the ensemble's answers, which are measured
  on several splits: `joint_val/accuracy`, `test/joint_answer_shift/net`.
- **The metric**, for metrics about one exchange, which are measured on
  joint_val only. The key then ends with where the exchange ran:
  `msg_norm/value/ex0`.
- **The metric**, for training dynamics, which ends with the part of the
  channel it measures: `grad_norm/encoder.query/ex0`.

Where an exchange ran is called its **site**. A forward exchange is `ex0`,
`ex1`, …, in the order the exchanges were given. A backward exchange runs once
per round, and each round is its own site: `ex0/round0`, `ex0/round1`, …. So
the key alone tells the second forward exchange (`ex1`) from the second round
of a backward exchange (`ex0/round1`).

Everything is numbered from 0: exchanges, rounds, and experts (`expert0`,
`expert1`, … in group order, the same order as the run's `expert0`, `expert1`,
… parameters).

## 3. Notation

The ensemble has $M$ experts. A split has $N$ examples, with labels $y_n$.

**Answers.** $\ell_{n,i}$ is the logit vector expert $i$ gives example $n$, and
$p_{n,i} = \operatorname{softmax}(\ell_{n,i})$ its probabilities, with
$p_{n,i,c}$ the probability of class $c$. A superscript says when:
$\text{pre}$ before any communication, $\text{post}$ after the last exchange.

The experts are frozen, so $\ell^{\text{pre}}$ depends only on the input. It is
what bagging sees, and it is the same in every arm, every trial and every
epoch.

The ensemble averages its experts' probabilities and answers with the most
likely class:

$$
\bar p_n = \frac{1}{M}\sum_{i} p_{n,i},
\qquad
\hat y_n = \arg\max_c \, \bar p_{n,c} .
$$

$\hat y^{\text{pre}}_n$ and $\hat y^{\text{post}}_n$ are its answers before and
after communication. $\hat y_{n,i} = \arg\max_c \ell_{n,i,c}$ is expert $i$'s
own answer.

**Messages.** In an exchange, expert $i$ sends a value
$v_{n,i} \in \mathbb{R}^{d_v}$, where $d_v$ is the channel's `value_dim` (64 by
default). The mixer gives each receiver $r$ a weight $w_{n,rs}$ for every
sender $s$. The weights are non-negative and sum to 1 over $s$, and the
receiver gets the weighted average

$$
m_{n,r} = \sum_{s} w_{n,rs} \, v_{n,s} ,
$$

the **aggregated** message. The receiver's decoder turns $m_{n,r}$ into the
**injection** $D_{n,r}$, which is added to the receiver's features at the write
point and has their shape: $C \times H \times W$ for a feature map, with
$P = HW$ positions.

**Shorthand.** $\lVert\cdot\rVert$ is the Euclidean norm of a tensor, all its
entries taken together. $[\,\text{statement}\,]$ is 1 when the statement is
true and 0 when it is false, so its average is the share of cases where the
statement holds. $\operatorname{mean}_n$ averages over examples.

## 4. Summary

| metric | quantity | value fixed by construction |
|---|---|---|
| `accuracy` | share of examples the ensemble answers correctly | bagging's accuracy at step 0 |
| `loss` | the training objective | — |
| `expert_accuracy` | share of the experts' own answers that are correct | — |
| `joint_answer_shift` | answers communication corrected and corrupted | `pre_accuracy` constant across arms, trials and epochs; `net` = 0 at step 0 |
| `marginal_answer_shift` | the same, with only one expert's update applied | 0 at step 0 |
| `answer_shift_kl` | how far communication moved the experts' probabilities | 0 at step 0 |
| `routed_correct` | how often the expert a receiver listened to was right | — |
| `msg_norm` | size of what is sent, received and injected | `injection` = 0 at step 0 |
| `attn_entropy` | how evenly receivers spread their attention | between 0 and $\log M$ |
| `attn_top1` | each receiver's largest attention weight | between $1/M$ and 1 |
| `routing_mode_share` | share of examples routed the most common way | — |
| `self_attn_rate` | how often a receiver listens most to itself | — |
| `route_flip_rate` | how much routing changed since the previous step | absent at step 0 |
| `inj_spread` | share of the injection that differs between examples | 0 at step 0 |
| `inj_resid_frac` | share of the injection that differs between positions | 0 for the `broadcast` decoder |
| `inj_rank` | number of distinct spatial patterns injected | 0 for `broadcast` |
| `msg_rank_value`, `msg_rank_aggregated` | number of directions a sent or received message varies along | — |
| `msg_rank_ratio` | directions received relative to directions sent | — |
| `grad_norm` | size of the gradient reaching each part of the channel | — |
| `update_ratio` | how far one step moves each part, relative to its size | — |

## 5. Task performance

### `accuracy`

The share of examples the ensemble answers correctly, after communication:

$$
\texttt{accuracy} = \operatorname*{mean}_n \, [\, \hat y^{\text{post}}_n = y_n \,] .
$$

At step 0 the channel injects nothing, so this is bagging's accuracy. How far
it rises above step 0 is what communication added, which `joint_answer_shift`
splits into the answers it corrected and the answers it corrupted.

Arms are compared on this, read on test at an epoch chosen from joint_val.

### `loss`

The training objective, evaluated on the split. With `--loss ensemble`, the
default, it is the negative log-likelihood of the ensemble's averaged
probabilities:

$$
\texttt{loss} = -\operatorname*{mean}_n \, \log \bar p^{\,\text{post}}_{n,\,y_n} .
$$

With `--loss per_expert` it is the mean of the experts' own cross-entropies:

$$
\texttt{loss} = -\operatorname*{mean}_{n,i} \, \log p^{\text{post}}_{n,i,\,y_n} .
$$

Every split uses the run's own objective, so the train and joint_val losses
are directly comparable. Two arms trained on different objectives are not
comparable by loss; compare them by accuracy.

### `expert_accuracy`

The share of the experts' own answers that are correct, after communication:

$$
\texttt{expert_accuracy} = \operatorname*{mean}_{n,i} \, [\, \hat y^{\text{post}}_{n,i} = y_n \,] .
$$

`--loss ensemble` does not optimise this. An expert can get worse at answering
on its own while the ensemble's answer gets better, because what is trained
is the average. `--loss per_expert` does optimise it. Measured on joint_val
and test.

The intent of this measurement is to show whether communication improved the
experts themselves or only their combined answer.

## 6. Communication interpretation

These metrics compare the answers before and after communication, example by
example.

### `joint_answer_shift`

How communication changed the ensemble's answers. Every example falls into one
of four groups: right before and after, wrong before and after, wrong before
and right after (**corrected**), or right before and wrong after
(**corrupted**). The metric reports the shares:

$$
\texttt{pre_accuracy} = \operatorname*{mean}_n \, [\, \hat y^{\text{pre}}_n = y_n \,],
\qquad
\texttt{post_accuracy} = \operatorname*{mean}_n \, [\, \hat y^{\text{post}}_n = y_n \,],
$$

$$
\texttt{corrected} = \operatorname*{mean}_n \, [\, \hat y^{\text{pre}}_n \ne y_n \text{ and } \hat y^{\text{post}}_n = y_n \,],
\qquad
\texttt{corrupted} = \operatorname*{mean}_n \, [\, \hat y^{\text{pre}}_n = y_n \text{ and } \hat y^{\text{post}}_n \ne y_n \,].
$$

The two unchanged groups contribute equally to both accuracies, so the change
in accuracy is exactly the corrected share minus the corrupted share:

$$
\texttt{net} = \texttt{corrected} - \texttt{corrupted} = \texttt{post_accuracy} - \texttt{pre_accuracy} .
$$

`net` is the accuracy communication added. The two parts tell apart two
channels whose `net` is flat: one that changes nothing, and one that makes
many changes that cancel. Both parts are shares of all examples, so
`corrected` is at most $1 - \texttt{pre_accuracy}$ and `corrupted` at most
$\texttt{pre_accuracy}$.

`pre_accuracy` is bagging's accuracy. The experts are frozen, so it is the
same in every arm, every trial and every epoch, and the examples available to
correct are one fixed set.

The last two kinds measure how often experts agree, averaged over every pair
of experts:

$$
\texttt{agreement_pre} = \operatorname*{mean}_n \; \operatorname*{mean}_{i < j} \, [\, \hat y^{\text{pre}}_{n,i} = \hat y^{\text{pre}}_{n,j} \,] ,
$$

and `agreement_post` likewise after communication. The ensemble averages its
experts, so agreement has no value of its own. It is a control on `net`.
Agreement rising while `net` does not means the experts are converging on
each other's answers rather than on the right one: consensus without
correctness.

Logged on all three splits. On train it covers the epoch's batches as the
channel was at each batch, so it trails joint_val slightly; its
`pre_accuracy` is still constant, since the experts and the joint split never
change.

The intent of this measurement is to split communication's effect on accuracy
into the help and the harm that make it up.

### `marginal_answer_shift`

Which experts' updates carry the effect. For expert $i$, start from the
answers before communication and replace only expert $i$'s logits with its
logits after communication:

$$
\ell^{(i)}_{n,j} =
\begin{cases}
\ell^{\text{post}}_{n,i} & \text{if } j = i, \\
\ell^{\text{pre}}_{n,j} & \text{otherwise.}
\end{cases}
$$

The ensemble's answer on these logits, $\hat y^{(i)}_n$, is compared with
$\hat y^{\text{pre}}_n$ exactly as in `joint_answer_shift`, giving
`corrected`, `corrupted` and `net` for expert $i$ alone (keys
`joint_val/marginal_answer_shift/net/expert0`, …).

The marginal `net`s do not add up to the joint one, because the ensemble's
answer is not additive in its experts. The gap is informative: marginal
`net`s summing to less than the joint `net` means the updates cooperate,
moving the vote together where no single update could. Expect small values,
since one updated expert has to outvote $M - 1$ unchanged ones to change the
answer. Measured on joint_val.

The intent of this measurement is to find an expert whose update alone drags
the vote the wrong way, or the one whose update carries the gain. With class
skew, read it alongside each expert's class mix.

### `answer_shift_kl`

How far communication moved each expert's probabilities, averaged over
experts and examples. It is the KL divergence from the probabilities before
communication to those after:

$$
\texttt{answer_shift_kl} = \operatorname*{mean}_{n,i} \, \sum_c p^{\text{post}}_{n,i,c} \log \frac{p^{\text{post}}_{n,i,c}}{p^{\text{pre}}_{n,i,c}} .
$$

It is 0 only when nothing moved. Every metric above counts answers, and
misses changes in probability too small to change an answer; this one does
not. `net` near 0 with a clearly positive KL means communication is shifting
the experts' beliefs, just not far enough to change what they answer.
Measured on joint_val.

### `routed_correct`

Whether receivers listen to experts who are right. For receiver $r$ on example
$n$, take the sender it weights most,

$$
a_{n,r} = \arg\max_s \, w_{n,rs} ,
$$

and ask whether that sender was right before communication. The metric is the
share of (example, receiver) pairs in a subset $S$ where it was:

$$
\texttt{routed_correct} = \operatorname*{mean}_{(n,r) \in S} \, [\, \hat y^{\text{pre}}_{n,\,a_{n,r}} = y_n \,] .
$$

There are three subsets, each a key:

- `all`: every pair where the receiver listened most to another expert,
  $a_{n,r} \ne r$.
- `corrected`: those pairs where the receiver's own answer went from wrong to
  right, $\hat y^{\text{pre}}_{n,r} \ne y_n$ and $\hat y^{\text{post}}_{n,r} = y_n$.
- `corrupted`: those pairs where it went from right to wrong.

Pairs where a receiver listened most to itself are left out. A corrected
receiver was wrong before by definition, so counting its own message would
measure the definition, not the routing.

To read it, compare `all` with the experts' average accuracy before
communication (`expert_accuracy` at step 0), which is roughly what listening at
random would score. If
routing does the work, `corrected` sits above `all` (the receiver listened to
someone who knew the answer) and `corrupted` below it (the harm came from
listening to someone who was wrong). If all three are about equal, routing has
nothing to do with what communication changed.

Counts are pooled over all of joint_val, and a subset with no pairs is not
logged. Needs a mixer with weights.

## 7. Communication content

These metrics describe what one exchange sends, how it is weighted and what is
injected, independently of what the experts then do with it. They are logged
per site, on joint_val. The five on attention and routing read the mixer's
weights, and are logged only for a mixer that has them (`attention` does).

### `msg_norm`

The average size of what is sent, received and injected:

$$
\texttt{value} = \operatorname*{mean}_{n,i} \lVert v_{n,i} \rVert ,
\qquad
\texttt{aggregated} = \operatorname*{mean}_{n,r} \lVert m_{n,r} \rVert ,
\qquad
\texttt{injection} = \operatorname*{mean}_{n,r} \lVert D_{n,r} \rVert .
$$

The injection is measured at the size it is added: a `broadcast` injection
counts once at every position.

`injection` is exactly 0 at step 0. A channel whose injection stays near 0 has
gone silent, which is a common outcome for a learned channel, and then no
other reading of that run means anything.

The intent of this measurement is to find where messages vanish, if they do:
in sending (`value`), in mixing (`aggregated`) or in decoding (`injection`).

### `attn_entropy`

How evenly each receiver spreads its weight over the senders:

$$
\texttt{attn_entropy} = -\operatorname*{mean}_{n,r} \, \sum_s w_{n,rs} \log w_{n,rs} ,
$$

taking $0 \log 0 = 0$. It is 0 when every receiver puts all its weight on one
sender, and $\log M$ when it weights all senders equally. Low and falling
means receivers are settling on one sender each. High and flat near $\log M$
means they average everyone equally, which together with a flat `msg_norm`
suggests the attention is not choosing at all.

### `attn_top1`

Each receiver's largest weight, averaged:

$$
\texttt{attn_top1} = \operatorname*{mean}_{n,r} \, \max_s \, w_{n,rs} .
$$

It reads directly: $1/M$ is equal weights, 1 is listening to one sender only.
Unlike the entropy, its scale does not depend on $M$. Read the two together:
one dominant sender with a flat tail (high `attn_top1`, middling entropy) and
weight split evenly between two senders (both middling) look alike in entropy
alone.

### `routing_mode_share`

Whether routing depends on the input. Write example $n$'s routing as the tuple
of who each receiver listened to most, $a_n = (a_{n,r})_r$, with $a_{n,r}$ as
in `routed_correct`. The metric is the share of examples routed the most
common way:

$$
\texttt{joint} = \max_{a} \, \operatorname*{mean}_n \, [\, a_n = a \,] ,
\qquad
\texttt{expert}r = \max_{s} \, \operatorname*{mean}_n \, [\, a_{n,r} = s \,] .
$$

The `joint` key covers the whole tuple. The `expert0`, `expert1`, … keys cover
one receiver each.

`attn_entropy` and `attn_top1` average over examples, so a receiver that
always listens to expert 2 and one that listens to a different, image-specific
expert each time look the same to them. This metric separates the two. A share
of 1 means the same wiring for every image: the channel behaves like a fixed
connection, and the question of who listens to whom has no input-dependent
answer. A low share means routing changes with the input. Check `attn_top1`
too, though: a low share with a low `attn_top1` is the argmax flipping between
near-equal weights, which is noise, not routing.

With $M$ experts there are $M^M$ possible wirings, so for large groups `joint`
reads near 0 even when every receiver routes consistently. Read the
per-receiver keys then.

### `self_attn_rate`

How often a receiver listens most to its own message:

$$
\texttt{self_attn_rate} = \operatorname*{mean}_{n,r} \, [\, a_{n,r} = r \,] .
$$

The mixer includes each expert's own message, so that with $M = 2$ a receiver
still has a choice. A high rate means receivers mostly read themselves back,
and little is being communicated.

### `route_flip_rate`

How much routing changed since the previous step: the share of (example,
receiver) pairs whose $a_{n,r}$ differs from the previous evaluation's
$a'_{n,r}$,

$$
\texttt{route_flip_rate} = \operatorname*{mean}_{n,r} \, [\, a_{n,r} \ne a'_{n,r} \,] .
$$

The examples are the same joint_val examples in the same order every time. It
is not logged at step 0, which has nothing to compare with. If routing were
re-drawn at random, it would read $(M - 1)/M$.

The intent of this measurement is to tell whether routing was learned or fixed
by the initialisation. A `routing_mode_share` of 1 at the end of training fits
two stories with opposite remedies:

- Routing explored and then collapsed: high early, then decaying.
- Routing was set by the initial weights and never moved, because the
  attention stopped receiving gradient: near 0 from step 1.

The decoders start at zero, so early in training the attention gets no
learning signal, and the second story is a real possibility.

### Two tools

The remaining metrics rely on two constructions, set out once here.

**Splitting a tensor's size.** Take a tensor $D$ and its average $\bar D$ over
some of its axes, repeated back along those axes to $D$'s shape. The remainder
$R = D - \bar D$ sums to zero along the averaged axes, while $\bar D$ is
constant along them, so the two are orthogonal:
$\langle R, \bar D \rangle = 0$. Their squared sizes therefore add,

$$
\lVert D \rVert^2 = \lVert \bar D \rVert^2 + \lVert R \rVert^2 ,
$$

and $\lVert R \rVert / \lVert D \rVert$ lies between 0 and 1. It is the share
of $D$'s size that varies along the averaged axes, and its square is the
share of the squared size.

**Counting directions.** Given a set of vectors, the rows of a matrix $X$, how
many independent directions do they use? Ordinary rank does not answer this,
because real data is never exactly dependent: one pattern plus a little noise
has full rank. The **participation ratio** counts directions, each weighted by
its share of the size. With $\lambda_1, \lambda_2, \dots$ the eigenvalues of
$G = X^\top X$, which are the squared sizes of the vectors along each
principal direction,

$$
\operatorname{PR}(X) = \frac{\bigl(\sum_j \lambda_j\bigr)^2}{\sum_j \lambda_j^2}
= \frac{\bigl(\operatorname{tr} G\bigr)^2}{\lVert G \rVert^2} .
$$

The second form holds because $\sum_j \lambda_j = \operatorname{tr} G$ and
$\sum_j \lambda_j^2 = \lVert G \rVert^2$ for a symmetric matrix, so no
eigendecomposition is needed. Some values:

- 1 when all the vectors lie on one line;
- $k$ when they spread equally over $k$ directions;
- 1.22 for sizes split 0.9 and 0.1 between two directions: nearly one
  direction.

Rescaling $X$ does not change it. It is reported as 0 when $X$ is all zero.
$X^\top X$ and $X X^\top$ have the same nonzero eigenvalues, so either matrix
gives the same ratio. The code uses $X^\top X$, which has one row per column of
$X$ and so stays small however many examples there are.

### `inj_spread`

The share of the injection that differs between examples. For receiver $r$,
take its injection on every example, $D_r = (D_{n,r})_n$, and split it around
its average over examples, $\tilde D_r = \operatorname{mean}_n D_{n,r}$:

$$
\texttt{inj_spread} = \operatorname*{mean}_r \, \frac{\lVert D_r - \tilde D_r \rVert}{\lVert D_r \rVert} .
$$

At 0, the decoder adds the same thing to every image: a learned constant
shift, not a message. `msg_norm`'s `injection` cannot tell a constant shift
from a working channel, since both can be large; this metric can.

This matters with class skew. Each skewed expert leans toward its favourite
classes, and a constant shift per expert can correct that lean and raise
accuracy with no communication at all. A gain over bagging with
`inj_spread` near 0 is that, not communication.

Reported as 0 when the injection is zero (norm below $10^{-12}$), as at step 0.

### `inj_resid_frac`

The share of the injection that differs between positions. For each example
and feature channel, take the injection's average over positions,

$$
\bar D_{n,r,c} = \frac{1}{P} \sum_{h,w} D_{n,r,c,h,w} ,
\qquad
\texttt{inj_resid_frac} = \operatorname*{mean}_r \, \frac{\lVert D_r - \bar D_r \rVert}{\lVert D_r \rVert} .
$$

It is 0 when the injection is the same at every position. That is always the
case for the `broadcast` decoder, so there it checks the measurement rather
than the decoder.

It describes the injection, not the message: a fixed spatial mask whose
strength the message sets reads high while carrying no spatial information.
Read it with `inj_rank`. Logged only where the write point is a feature map,
and reported as 0 when the injection is zero.

### `inj_rank`

How many distinct spatial patterns the injection uses. For receiver $r$ and
feature channel $c$, put the spatial remainder of every example in the rows of
a matrix,

$$
R^{(r,c)} \in \mathbb{R}^{N \times P} ,
\qquad
R^{(r,c)}_{n,(h,w)} = D_{n,r,c,h,w} - \bar D_{n,r,c} ,
$$

and count its directions:

$$
\texttt{inj_rank} = \operatorname*{mean}_r \, \operatorname*{mean}_c \, \operatorname{PR}\bigl(R^{(r,c)}\bigr) .
$$

It is 1 for one fixed pattern whose strength varies, and above 1 when the
pattern's shape depends on the message. It is at most $P$, a property of the
write point rather than of the measurement. It is reported as 0 where
`inj_resid_frac` is below $10^{-6}$, since then there is no spatial pattern to
count, so `broadcast` reads 0. A feature channel with no spatial pattern
counts as 0 in the average over channels.

Together with `inj_spread` it places the decoder in one of these regimes:

| `inj_rank` | `inj_spread` | reading |
|---|---|---|
| 0 | 0 | the same shift for every image, at every position |
| 0 | above 0 | one value per feature channel, set by the message: `broadcast` working |
| 1 | 0 | one fixed spatial pattern, message ignored |
| 1 | above 0 | one fixed spatial pattern, its strength set by the message |
| above 1 | above 0 | a pattern whose shape depends on the message |

### `msg_rank_value` and `msg_rank_aggregated`

How many directions an expert's messages actually vary along. For expert $i$,
stack its sent values over joint_val into a matrix $V_i \in \mathbb{R}^{N \times d_v}$
and remove the average message before counting:

$$
\texttt{msg_rank_value}_i = \operatorname{PR}\bigl(V_i - \operatorname*{mean}_n v_{n,i}\bigr) .
$$

`msg_rank_aggregated` is the same for the aggregated message $m_{n,i}$ the
expert receives. Both are logged per expert and averaged (`mean`), so that one
dead encoder shows on its own instead of being diluted in the average. Both
are at most $d_v$.

Removing the average matters. A part shared by every message would otherwise
dominate the count and read about 1 for a channel in perfect health. The
count is of directions used, not of information: directions of noise count
the same as directions of signal.

The received message varies for two reasons: the senders' messages change
with the input, and so do the weights. Mixing can therefore add variation of
its own.

The intent of this measurement is to see how much of the channel's width the
encoders use, out of the $d_v$ available, and how much of that variation
reaches the decoders.

### `msg_rank_ratio`

Directions received relative to directions sent:

$$
\texttt{msg_rank_ratio} = \frac{\operatorname*{mean}_i \, \texttt{msg_rank_aggregated}_i}{\operatorname*{mean}_i \, \texttt{msg_rank_value}_i} .
$$

Above 1, what arrives spans more directions than a typical sender emits, so
the mixing combines senders. Near 1, what arrives looks like one sender's
message. Rising above 1 needs two things: senders that vary in different
directions, and mixing that draws on more than one of them. The ratio cannot
say which is missing; `routing_mode_share` and `attn_top1` show whether the
attention is pinned to one sender.

## 8. Training dynamics

These measure whether the channel's weights get gradient and move, separately
for each part of each channel. A part is named after the module it is in, e.g.
`encoder.query/ex0` is the query layer of exchange 0's encoders, pooled over
the experts. The parts are `encoder.query`, `encoder.key` and `encoder.value`,
`encoder.score` and `encoder.position` for the spatial encoder, `decoder.mlp`,
and `mixer` for a mixer with weights. `total` is every channel weight
together. A backward exchange's rounds share one channel, so it has one set.

Both are averaged over the optimiser steps of each epoch.

### `grad_norm`

The size of the gradient reaching a part $g$, with weights $\theta_g$:

$$
\texttt{grad_norm}_g = \bigl\lVert \nabla_{\theta_g} \mathcal{L} \bigr\rVert ,
$$

taken after the backward pass and before the optimiser step. The parts'
squared norms add up to the squared `total`.

The intent of this measurement is to catch vanishing or exploding gradients,
part by part. `encoder.query` and `encoder.key` get gradient only through the
attention weights, so they are the first to go quiet if the attention
saturates. v2 traced its degenerate routing to exactly that.

### `update_ratio`

How far one optimiser step moves a part, relative to the part's size:

$$
\texttt{update_ratio}_g = \frac{\lVert \theta^{\text{after}}_g - \theta^{\text{before}}_g \rVert}{\lVert \theta^{\text{before}}_g \rVert} .
$$

Adam rescales its steps, so a healthy-looking gradient can still move the
weights almost nothing, and this catches that. As a rough guide, about
$10^{-3}$ per step is healthy, below $10^{-6}$ is effectively frozen, and above
$10^{-1}$ is thrashing.

## 9. Changes from v2

- **One version per metric.** v2 logged most communication metrics twice: on
  each training batch, and over the full validation set. v3 keeps only the
  full pass over joint_val, which takes a fraction of a second. The exceptions
  are `joint_answer_shift`, also logged on train, and the training dynamics.
- **Exact over the whole split.** v2 computed `inj_*` and `msg_rank_*` on one
  fixed batch of 128. v3 computes them over all of joint_val. `inj_rank` now
  uses the matrix over positions, $R^\top R$, rather than over examples,
  $R R^\top$. These give the same ratio for the same rows, but this one stays
  small however many examples there are. Its ceiling is now $P$, set by the
  write point, instead of the batch size.
- **Per site.** Every exchange metric is logged per site, and each round of a
  backward exchange is its own site; v2 kept the last round only.
- **Test every epoch.** `test/*` is logged at every step, including
  `joint_answer_shift` on test. The bagging reference is step 0 and
  `pre_accuracy`, not a separate metric.
- **Parts named from the code.** `grad_norm` and `update_ratio` are split by
  module name, so v2's fixed list of components and its `other` bucket are
  gone.
- **Dropped.**
  - `final_*`, `final_test_accuracy_endpoint` and the replicate `*_mean`/`*_std`
    summaries: runs are scored from the per-step series by the results stage.
  - `val_loss` on a single streamed batch.
  - `layernorm_param_norm`: v3 has no LayerNorm.
  - `pool_mask_dev`: no v3 encoder has a fixed learned mask.
  - The per-batch copies of the routing and answer-shift metrics.
- **New in v3.** `expert_accuracy`, from the testbed skeleton.
