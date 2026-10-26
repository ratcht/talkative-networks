# Communication channel

This page defines the channel an exchange trains: its parts, two controls,
and their sizes. Studies combine the parts into arms, such as the levels of
the [channel ladder](../studies/channel-ladder.md). The code is in
`testbed/communication/channel.py`. Where exchanges
read and write, and which comparisons are still open, is in `docs/docket.md`.
What the logged metrics mean is in [Metrics](metrics.md).

An exchange reads every expert's features at a read point, mixes them across
experts, and writes back into every expert at a write point. A forward
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

At the read point, expert $i$'s features are a $C \times H \times W$ map. The
encoder sees all of it, every channel at every position, as one vector
$f_i \in \mathbb{R}^{n_{\text{read}}}$ with $n_{\text{read}} = CHW$. The write
point holds $n_{\text{write}}$ numbers in the same way.

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

On cnn3 experts (Fashion-MNIST: stage1 is $16 \times 14 \times 14 = 3136$,
stage2 $32 \times 7 \times 7 = 1568$, stage3 $64 \times 3 \times 3 = 576$; one
expert has 98,442 weights), at $r = 8$:

| exchange | encoder | decoder | total | × expert |
|---|---:|---:|---:|---:|
| forward stage1 → stage2 | 25,960 | 13,064 | 39,024 | 0.40× |
| forward stage2 → stage3 | 13,416 | 5,128 | 18,544 | 0.19× |
| forward stage1 → stage3 | 25,960 | 5,128 | 31,088 | 0.32× |
| backward stage2 → stage1 | 13,416 | 25,608 | 39,024 | 0.40× |
| backward stage3 → stage2 | 5,480 | 13,064 | 18,544 | 0.19× |
| backward stage3 → stage1 | 5,480 | 25,608 | 31,088 | 0.32× |

The side that touches the larger map holds most of the weights: the encoder
when reading there, the decoder when writing there. Across widths, each row
holding for both directions:

| points | $r = 4$ | $r = 8$ | $r = 16$ | $r = 32$ | $r = 64$ |
|---|---:|---:|---:|---:|---:|
| stage1 and stage2 | 0.20× | 0.40× | 0.79× | 1.58× | 3.16× |
| stage2 and stage3 | 0.09× | 0.19× | 0.38× | 0.75× | 1.50× |
| stage1 and stage3 | 0.16× | 0.32× | 0.63× | 1.26× | 2.52× |

For comparison, a dense map, with its own weight for every entry in every
message number, would be 1.7–4.2× an expert on these exchanges. The `pooled`
encoder with the `broadcast` decoder is 0.08–0.13×.

## 6. What the width limits

- **How much an expert says.** A message carries at most $r$ independent
  numbers, and an injection is a mix of at most $r$ patterns.
- **Pooling and broadcasting are not special cases at small $r$.** Reproducing
  the `pooled` encoder exactly takes $r \ge C$ at the read point, and a
  broadcast injection takes $r \ge C$ at the write point: 16 at stage 1, 32 at
  stage 2, 64 at stage 3 on cnn3. Below that the channel can only learn a
  cut-down version of either.
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
