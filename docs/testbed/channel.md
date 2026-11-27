# Communication channel

This page defines the channel an exchange trains: its parts, two controls,
and their sizes. Studies combine the parts into arms, such as the levels of
the [channel ladder](../studies/channel-ladder.md). The code is in
`testbed/communication/channel.py`. Where exchanges
read and write, and which comparisons are still open, is in `docs/docket.md`.
What the logged metrics mean is in [Metrics](metrics.md).

An exchange reads every expert's residual stream at a read point, mixes it
across experts, and writes back into every expert's stream at a write point,
by adding an injection to it. The points are the stream after each block, and
the channel reads and writes nowhere else ([Experts](experts.md) §1). A forward
exchange writes later than it reads, in one pass. A backward exchange writes
earlier than it reads, and the stages in between re-run after each write. The
channel has three parts, and the same parts serve both directions; only the
read and write points change.

| part | one per | gets | returns |
|---|---|---|---|
| encoder | expert | the expert's whole feature map at the read point | its message: a query, a key and a value |
| mixer | exchange | every expert's message | what each expert received |
| decoder | expert | what its expert received, and that expert's features at the write point | an injection shaped like those features, which is added to them |

## 1. Notation

The ensemble has $M$ experts, indexed by $i$ and $j$. (The metrics page calls a
receiver $r$; here $r$ is the width below.) Everything is per example, so the
example index is left out.

At the read point, expert $i$'s features are its residual stream, $C$ channels
at each of $P$ positions: a $C \times H \times W$ grid for a ResNet, $C \times 65$
tokens for a ViT ([Experts](experts.md) §1). The encoder sees all of it, every
channel at every position, as one vector $f_i \in \mathbb{R}^{n_{\text{read}}}$
with $n_{\text{read}} = CP$. The write point holds $n_{\text{write}}$ numbers
in the same way; in v4 it is the same stream, so $n_{\text{write}} = n_{\text{read}}$.

| symbol | config field | default | meaning |
|---|---|---|---|
| $r$ | `width` (`--width`) | 8 | the narrow middle of every linear taper encoder and decoder |
| $d_k$ | `key_dim` | 16 | size of a query and of a key |
| $d_v$ | `value_dim` | 64 | size of a value |

## 2. The linear taper channel

### Encoder: `linear_taper`

Each expert reads its whole feature map through $r$ learned templates, then
builds its query, key and value from those $r$ readings:

$$
z_i = A_i f_i + a_i \in \mathbb{R}^r ,
\qquad
q_i = W^q_i z_i + b^q_i ,
\quad
k_i = W^k_i z_i + b^k_i ,
\quad
v_i = W^v_i z_i + b^v_i .
$$

Each row of $A_i$ is a template: one weight for every channel at every
position, with no shape imposed on it. So nothing is pooled. The query, key
and value are all built from the same $r$ readings, so an expert's message
carries at most $r$ independent numbers, whatever $d_k$ and $d_v$ are.
Construction check: `msg_rank_value` is at most $r$.

### Mixer: `attention`

Single-head attention over every expert's message, its own included, as in
TarMAC. With $M = 2$, leaving its own out would give each receiver one sender
and a softmax stuck at 1.

$$
w_{ij} = \operatorname*{softmax}_j \frac{q_i \cdot k_j}{\sqrt{d_k}} ,
\qquad
m_i = \sum_j w_{ij} \, v_j .
$$

$m_i$ is what expert $i$ received. With the `linear_taper` encoder and
decoder, the softmax is the channel's only nonlinearity: with the weights
$w_{ij}$ held fixed, the whole channel is linear in the experts' features.

### Mixer: `rmsnorm_attention`

Attention with each query and key RMS-normed first (QK-norm):

$$
\hat q_i = \frac{q_i}{\operatorname{RMS}(q_i)} ,
\quad
\hat k_j = \frac{k_j}{\operatorname{RMS}(k_j)} ,
\qquad
w_{ij} = \operatorname*{softmax}_j \frac{\hat q_i \cdot \hat k_j}{\sqrt{d_k}} ,
\qquad
m_i = \sum_j w_{ij} \, v_j ,
$$

with $\operatorname{RMS}(x) = \sqrt{\tfrac{1}{d_k} \sum x^2}$. Every logit
lies in $[-\sqrt{d_k}, \sqrt{d_k}]$. The norm has no gain, so the mixer has no
weights.

### Decoder: `linear_taper`

Each expert turns what it received into $r$ numbers, the strengths of $r$
learned patterns, and injects that mix of patterns:

$$
u_i = N_i m_i + c_i \in \mathbb{R}^r ,
\qquad
D_i = P_i u_i .
$$

$D_i$ is reshaped to the write point's shape and added to the features there.
Each column of $P_i$ is a pattern the size of the whole write point, free over
channels and positions, so nothing is broadcast. $P_i$ has no bias, so the
injection is always a mix of the $r$ patterns. $P_i$ starts at zero, so an
untrained channel injects nothing and step 0 is bagging. Construction check:
`inj_rank` is at most $r$.

## 3. The GELU versions: `linear_taper_gelu`

The `linear_taper_gelu` encoder puts a GELU on the readings, and the
`linear_taper_gelu` decoder puts one on the strengths:

$$
z_i = \operatorname{GELU}(A_i f_i + a_i) \quad \text{(encoder)} ,
\qquad
u_i = \operatorname{GELU}(N_i m_i + c_i) \quad \text{(decoder)} .
$$

A GELU has no weights, so each `linear_taper_gelu` part has exactly the
weights of its `linear_taper` version. $r$ is then also the width the GELU
acts on. So a GELU part that gains nothing at a small $r$ may be too narrow
rather than not nonlinear enough, and a wider run tells the two apart.

## 4. The controls: `self` and `mean`

Each control replaces the attention mixer and keeps every other part:

$$
m_i = v_i \quad (\texttt{self}) ,
\qquad
m_i = \frac{1}{M} \sum_j v_j \quad (\texttt{mean}) .
$$

Under `self`, each expert receives only its own message. Under `mean`, every
expert receives the plain average of all the messages, its own included. In
both, the query and key go unused, so they get no gradient, and the weights
are fixed, so there is no routing for the routing metrics to read.

Each expert's channel is also an add-on of its own: a path from its read point
to its write point, trained on the joint split, which no expert has seen. That
can raise accuracy with no communication at all. The intent of the controls is
to take that apart, trial seed by trial seed:

| difference | what it measures |
|---|---|
| full channel − `self` | what communication adds over each expert's own add-on |
| full channel − `mean` | what choosing whom to listen to adds over hearing everyone equally |
| `mean` − `self` | what hearing the others adds, with no choosing |

## 5. Size

Per expert, counting the encoder's two layers, then the decoder's:

$$
\underbrace{(n_{\text{read}} + 1)\, r}_{A_i,\ a_i}
+ \underbrace{(r + 1)(2 d_k + d_v)}_{\text{query, key, value}}
+ \underbrace{(d_v + 1)\, r}_{N_i,\ c_i}
+ \underbrace{n_{\text{write}}\, r}_{P_i}
= r \,(n_{\text{read}} + n_{\text{write}} + 2 d_k + 2 d_v + 2) + 2 d_k + d_v .
$$

With the defaults that is $r\,(n_{\text{read}} + n_{\text{write}} + 162) + 96$.
It depends on the two points only through $n_{\text{read}} + n_{\text{write}}$,
so a forward exchange and its backward mirror, the same two points swapped,
are always the same size. The mixers have no weights, each `linear_taper_gelu`
part has its `linear_taper` version's weights, and the controls train a subset
of them. With $M$ experts the channel
is $M$ times this, against $M$ experts, so the ratios below hold for the group.

In v4 every point is the residual stream, so $n_{\text{read}} = n_{\text{write}}$
and every exchange of one family is the same size, wherever it reads and
writes ([Experts](experts.md) §1). At $r = 8$:

| experts | stream | $n$ | encoder | decoder | total | × expert |
|---|---|---:|---:|---:|---:|---:|
| ResNet $12\times64$ (896,164) | $64 \times 16 \times 16$ | 16,384 | 131,944 | 131,592 | 263,536 | 0.29× |
| ViT $12\times96$ (1,363,012) | $96 \times 65$ | 6,240 | 50,792 | 50,440 | 101,232 | 0.07× |

Per expert; a group's channel is $M$ times this, against $M$ experts. Across
widths, per expert:

| experts | $r = 4$ | $r = 8$ | $r = 16$ | $r = 32$ |
|---|---:|---:|---:|---:|
| ResNet $12\times64$ | 0.15× | 0.29× | 0.59× | 1.18× |
| ViT $12\times96$ | 0.04× | 0.07× | 0.15× | 0.30× |

On v3's cnn3 experts the same widths gave 0.19–0.40× at $r = 8$, depending on
the points. For comparison, a dense map, with its own weight for every entry in every
message number, would be several times an expert. The `pooled` encoder with
the `broadcast` decoder is far smaller.

## 6. What the width limits

- **How much an expert says.** A message carries at most $r$ independent
  numbers, and an injection is a mix of at most $r$ patterns.
- **Pooling and broadcasting are not special cases at small $r$.** Reproducing
  the `pooled` encoder exactly takes $r \ge C$ at the read point, and a
  broadcast injection takes $r \ge C$ at the write point: 64 for the ResNet
  $12\times64$, 96 for the ViT $12\times96$. Below that the channel can only
  learn a cut-down version of either.
- **A ceiling.** The `linear_taper` decoder can use at most $r = d_v = 64$, since what
  it reads has $d_v$ numbers, and at that width the channel is already close
  to dense in size.

## 7. Parts that make spatial assumptions

The linear taper parts avoid these assumptions. These parts stay available
for reference.

| name | part | what it assumes |
|---|---|---|
| `pooled` | encoder | each channel averaged over positions before anything is read (v2's default) |
| `spatial` | encoder | positions, tagged with a learned position embedding, weighed and summed into one summary |
| `broadcast` | decoder | one value per channel, the same at every position |
| `spatial` | decoder | each position combines its own features with the message |

## 8. Attention on directions alone: `rmsnorm_attention`

Under `attention` (§2), an encoder can sharpen the softmax just by sending
larger queries and keys. `rmsnorm_attention` (§2) takes that away. A vector's
RMS is its length divided by $\sqrt{d_k}$, so dividing by the RMS rescales every
query and key to length $\sqrt{d_k}$, and the same softmax then reads a cosine:

$$
\hat q_i = \frac{q_i}{\operatorname{RMS}(q_i)} = \sqrt{d_k} \, \frac{q_i}{\lVert q_i \rVert} ,
\qquad
\hat k_j = \frac{k_j}{\operatorname{RMS}(k_j)} = \sqrt{d_k} \, \frac{k_j}{\lVert k_j \rVert} ,
\qquad
w_{ij} = \operatorname*{softmax}_j \frac{\hat q_i \cdot \hat k_j}{\sqrt{d_k}}
       = \operatorname*{softmax}_j \, \sqrt{d_k} \, \cos(q_i, k_j) .
$$

So a receiver's weights depend only on the angles between its query and the
keys. $m_i = \sum_j w_{ij} v_j$ as before, and the values keep their sizes.

$\sqrt{d_k}$ is the length a query would have if each of its numbers had
variance 1, which is the case attention's division by $\sqrt{d_k}$ assumes.
The length has to be chosen with care, because it sets how far the weights can
go. With $M = 3$ and $d_k = 16$, when one key lines up with a receiver's query
and the other two point the opposite way:

| rescaled to | scores lie in | a receiver's largest weight can reach | lowest `attn_entropy` |
|---|---|---|---|
| $\sqrt{d_k}$ | $[-4, 4]$ | 0.9993 | 0.006 |
| 1 | $[-0.25, 0.25]$ | 0.45 | 1.07 |

At length 1, attention would be close to `mean` whatever the channel learned,
so softer routing would prove nothing. At $\sqrt{d_k}$ a receiver can still
settle on one sender, as sharply as `attention` does at level 1 (entropy about
0.01), but only by direction. The lengths are fixed: the norm has no gain,
since a learned scale would bring back what this mixer removes.

The mixer has no weights, like `attention`. So on the same trial seed the two
start from identical channels, and a difference between them is the
normalising. `msg_norm` reads the queries and keys as the encoders sent them,
before this rescaling ([Metrics](metrics.md)).
