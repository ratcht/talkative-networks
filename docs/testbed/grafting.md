# Model grafting

This page defines a graft: a group of frozen experts in which one expert, the
**anchor**, alone sees the image and gives the answer, and the others, the
**obligates**, take part only through a channel. The code is in
`testbed/grafting.py` (`--graft` in `testbed.run`); the study that uses it is
[Model grafting](../studies/model-grafting.md). What every logged metric means
in general is in [Metrics](metrics.md); §6 says what changes for a graft.

In a communicating ensemble every expert sees the image and the answer is
the average of all of them, so each expert's knowledge reaches the answer with
no communication at all. In a graft the obligates' knowledge can reach the
answer only through communication: cut the channel and they contribute
nothing. The starting point is therefore the anchor alone, not bagging.

| term | meaning |
|---|---|
| anchor | the expert that sees the image and gives the answer: the group's first expert, the same in every arm of a trial seed |
| obligate | every other expert of the group; it never sees the image and its answer is never read |
| read point $a$ | where the anchor's stream is read, and where the obligates are written |
| write point $d$ | where the obligates are read, and where the anchor is written |
| outbound | the stitch from the anchor's stream at $a$ into each obligate |
| return | the attention from the anchor's stream at $d$ over every obligate's stream at $d$, added to the anchor's |
| bypass | the control: the same channel with the obligates' blocks from $a$ to $d$ left out |

This departs from the rule that every expert sees the same full input
([Docket](../docket.md)): the obligates see nothing, and act only through
what the anchor sends them.

## 1. Notation

The group has $M$ experts; expert 0 is the anchor, and the obligates are
$j = 1, \dots, M-1$. Every expert keeps one residual stream
([Experts](experts.md) §1): $x^{(i)}_0 = \mathrm{embed}_i(\text{image})$, then
$x^{(i)}_\ell = x^{(i)}_{\ell-1} + f^{(i)}_\ell(x^{(i)}_{\ell-1})$ for blocks
$\ell = 1, \dots, L$, and the answer is $\mathrm{head}_i(x^{(i)}_L)$. The
stream has $C$ channels at each of $P$ positions: $64 \times 256$ for the
ResNet $12\times64$, $96 \times 65$ for the ViT $12\times96$, whose class token
counts as a position here.

| symbol | flag | default | meaning |
|---|---|---|---|
| $a$ | `--graft-read` | `block4` | the read point |
| $d$ | `--graft-write` | `block8` | the write point, later than $a$ |
| $h$ | `--graft-heads` | $C/32$: 2 for the ResNet, 3 for the ViT | attention heads |
| | `--graft-mixer` | `attention` | `attention` or `mean` (§3) |
| | `--graft-stitch` | the family's own (§2) | the outbound stitch |
| | `--graft-bypass` | off | the control (§4) |

## 2. The pass

The anchor runs the image to its read point, and on to its write point, as it
always would:

$$
u = x^{(0)}_a , \qquad z = x^{(0)}_d .
$$

**Outbound.** Each obligate's stream at the read point is a stitch of the
anchor's, and the obligate runs its own blocks from there to the write point:

$$
x^{(j)}_a = S_j(u) , \qquad
r_j = x^{(j)}_d = x^{(j)}_a + \sum_{\ell = a+1}^{d} f^{(j)}_\ell\big(x^{(j)}_{\ell-1}\big) .
$$

As in model stitching (Bansal et al. 2021), the stitch's output is the whole
stream the obligate starts from: its embedding and its blocks before $a$ never
run, nor do its blocks after $d$ or its head. $S_j$ is the same affine map
from $C$ channels to $C$ channels at every position:

- `bn_pointwise_linear_bn`, the ResNet's default: batch norm, a $1\times1$
  convolution, batch norm, Bansal et al.'s stitch for convolutional networks.
  The norms only help optimisation; at evaluation the whole stitch is one
  affine map.
- `pointwise_linear`, the ViT's default: Bansal et al.'s "token-wise linear
  function", the class token included.

**Return.** Every position of the anchor's stream at the write point attends
over every position of every obligate's stream there (CALM, Bansal et al.
2024), and the result is added to the anchor's stream:

$$
x^{(0)}_d = z + \Delta , \qquad \text{answer} = \mathrm{head}_0\big(x^{(0)}_L\big),
$$

with the anchor's blocks after $d$ run on the new stream. §3 defines $\Delta$.

This is the side branch: both stitches join the same depth in the two
networks, which is what the stitching papers test, and the anchor runs all of
its own blocks, so the obligates' blocks $a+1, \dots, d$ run alongside the
anchor's own. Every image gets one forward pass; nothing runs twice.

## 3. The return

Write $z_p$ for the anchor's stream at position $p$ (a vector of $C$), and
$r_{j,p}$ for obligate $j$'s. Each obligate's stream first goes through its
own projection, CALM's $f_\text{proj}$, a linear map at every position:

$$
s_{j,p} = F_j\, r_{j,p} + c_j .
$$

### Mixer: `attention`

Multi-head attention with $h$ heads of width $C/h$. Every anchor position asks
with a query, and every position of every obligate answers with a key and a
value; one softmax runs over all $(M-1) P$ of them, so each anchor position
chooses both which obligate to listen to and where:

$$
q_p = W^Q z_p , \qquad k_{j,p'} = W^K s_{j,p'} , \qquad v_{j,p'} = W^V s_{j,p'} ,
$$

$$
w_{p,(j,p')} = \operatorname*{softmax}_{(j,p')} \frac{q_p \cdot k_{j,p'}}{\sqrt{C/h}} ,
\qquad
\Delta_p = W^O \sum_{j,p'} w_{p,(j,p')}\, v_{j,p'} ,
$$

per head, the heads concatenated before $W^O$; every map has a bias. $W^Q$,
$W^K$, $W^V$ and $W^O$ are shared by the obligates; each obligate's own
basis is handled by its projection. No position embedding is added, as in
CALM: the ViT's tokens carry their own, and the ResNet's grid cells carry
little, so the ResNet's attention routes mostly by content. There is no
"nobody" option: the weights always sum to 1, and a reply that isn't useful
has to be removed by the value and output maps.

### Mixer: `mean`

Each anchor position takes the plain average of the same position in every
obligate, so nothing is chosen:

$$
\Delta_p = W^O\, W^V \frac{1}{M-1} \sum_j s_{j,p} .
$$

It has no query or key. Averaging over every position instead would hand each
anchor position the same global average.

### The zero start

$W^O$ starts with every weight and bias at zero. The anchor's answer depends
on the obligates only through $\Delta$, so before training $\Delta = 0$ on
every image and the graft's answers are exactly the anchor alone's. That is
the step-0 evaluation, the first point of every MLflow curve. Training moves
$W^O$ off zero only where that lowers the loss. A zero weight still gets a
gradient, which depends on its input, not its value, so $W^O$ moves on the
first update, and everything before it, the stitches included, gets gradient
from the second (ControlNet's zero convolutions work the same way). Only $W^O$
starts at zero; the stitches and the other maps start from PyTorch's default
random weights, trained on the task loss alone. The loss is the cross-entropy
of the anchor's answer.

## 4. The bypass

`--graft-bypass` keeps every part and leaves out the obligates' blocks
$a+1, \dots, d$: the stitched streams go straight to the return,
$r_j = S_j(u)$. It has exactly the graft's weights, and none of the
obligates' own weights take part.

In the graft, each obligate's skip connection carries $S_j(u)$ through its
blocks to $d$, and a chain of linear maps is itself one linear map. So the
graft can do everything the bypass does, plus whatever the obligates' frozen
blocks add, and **graft − bypass**, trial seed by trial seed, measures those
blocks. It cannot tell their training apart from frozen extra computation;
an arm with untrained obligates would, and is left for later.

## 5. Size

Per group of three (two obligates), counted from the code
(`tests/test_grafting.py`):

| | ResNet $12\times64$ | ViT $12\times96$ |
|---|---:|---:|
| outbound stitches (2) | 8,832 | 18,624 |
| projections $F_j$ (2) | 8,320 | 18,624 |
| $W^Q, W^K, W^V, W^O$ | 16,640 | 37,248 |
| **channel, `attention`** | **33,792** | **74,496** |
| channel, `mean` (no $W^Q$, $W^K$) | 25,472 | 55,872 |
| relative to the anchor | 3.8% | 5.5% |
| relative to both obligates, whole | 1.9% | 2.7% |
| relative to the obligates' blocks 5–8 | 5.7% | 8.3% |

The head count doesn't change these. For comparison, the linear taper channel
at $r = 8$ is 790,608 (ResNet) and 303,696 (ViT) for a group of three
([Channel](channel.md) §5).

## 6. Metrics for a graft

Only the anchor answers, so the metrics that compare experts' answers are not
logged: `expert_accuracy`, `answer_grid`, `marginal_answer_shift`,
`routed_correct`, and `joint_answer_shift`'s agreement and top-$k$ overlap
kinds. The others read as follows. The exchange site is always `ex0`, and the
obligates are numbered as in the group, from 1, since the anchor is expert 0.

| metric | for a graft |
|---|---|
| `accuracy`, `loss` | the anchor's answer after grafting; at step 0, the anchor alone |
| `joint_answer_shift` | `pre_accuracy`, `post_accuracy`, `corrected`, `corrupted` and `net`, with "before" the anchor alone |
| `answer_shift_kl` | how far grafting moved the anchor's probabilities from the anchor alone's |
| `msg_norm`, `msg_spread` | parts `stitched` (each obligate's stream as the stitch makes it), `reply` (each obligate's stream at the write point), `injection` ($\Delta$; norm only), and `query` and `key` (the anchor's queries, the keys of every obligate position, before the split into heads; `attention` only). Per obligate under `expert{j}` for `stitched` and `reply` |
| `attn_heard` | `attn_heard/expert0/from{j}`: the anchor's attention on obligate $j$, summed over its positions, averaged over the anchor's positions, the heads and the images; the shares sum to 1 |
| `attn_entropy`, `attn_top1` | over the obligates, after summing each obligate's positions: entropy runs from 0 (one obligate) to $\log(M-1)$, $\log 2 \approx 0.69$ for two; the largest share runs from $1/(M-1)$, 0.5 for two, to 1 |
| `inj_spread`, `inj_resid_frac`, `inj_rank` | of $\Delta$, as for an ensemble's injection; the grid statistics leave the ViT's class token out |
| `grad_norm`, `update_ratio`, `weight_norm` | components `stitch`, `project`, `query`, `key`, `value` and `out`, all under `ex0` |

Not yet logged for a graft: which positions an anchor position attends to,
`qk_cos_gap`, `routing_mode_share` and `route_flip_rate`.

## Sources

- Bansal, Nakkiran, Barak 2021, *Revisiting Model Stitching to Compare Neural
  Representations* (arXiv 2106.07682): the stitches, matched depths, and
  independently trained networks stitching at every layer.
- Csiszárik et al. 2021, *Similarity and Matching of Neural Network
  Representations* (arXiv 2110.14633): the $1\times1$ stitch, and the
  least-squares start not used here, since it needs the obligates run on the
  images.
- Bansal et al. 2024, *LLM Augmented LLMs: Expanding Capabilities through
  Composition* (CALM, arXiv 2401.02412): the anchor's cross-attention over
  projected representations of frozen augmenting models, added to the
  anchor's stream.
- Zhang et al. 2023, *Adding Conditional Control to Text-to-Image Diffusion
  Models* (ControlNet, arXiv 2302.05543): zero-initialised layers that leave a
  frozen model unchanged at the start.
