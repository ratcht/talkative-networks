# Soft-gated MoE

This page defines the soft-gated mixture of experts, the baseline that weighs
the frozen experts' answers instead of letting them communicate. The code is
in `testbed/moe.py`. What its logged metrics mean is in [Metrics](metrics.md).

Every expert runs unchanged. A gate reads every expert's features at one
point and gives each expert a weight for this input. The ensemble's answer is
the weighted average of the experts' probabilities. Only the gate trains.

## 1. Notation

The ensemble has $M$ experts, indexed by $i$ and $j$. Everything is per
example, so the example index is left out. At the gate's read point, expert
$i$'s features are a $C \times H \times W$ map, read whole as one vector
$f_i \in \mathbb{R}^{n_{\text{read}}}$ with $n_{\text{read}} = CHW$. Expert
$i$'s probabilities over the classes are $p_i$.

| symbol | flag | default | meaning |
|---|---|---|---|
| — | `--gate` | — | the gate's architecture; `linear_taper_gelu` is the only one |
| — | `--gate-read` | the experts' last point | where the gate reads |
| $r$ | `--width` | 8 | how many readings the gate takes of each expert |

## 2. The gate: `linear_taper_gelu`

The gate reads each expert's whole feature map through $r$ learned templates,
as the `linear_taper` encoder does ([Channel](channel.md) §2), so nothing is
pooled. A GELU and one linear layer over every expert's readings then give
each expert a score, and the softmax of the scores gives the weights:

$$
z_i = A_i f_i + a_i \in \mathbb{R}^r ,
\qquad
s = W \operatorname{GELU}\bigl([z_1; \dots; z_M]\bigr) + b \in \mathbb{R}^M ,
\qquad
g = \operatorname{softmax}(s) .
$$

The ensemble's probabilities are the experts' weighted by the gate:

$$
\bar p = \sum_i g_i \, p_i .
$$

Because the gate sees every expert's readings at once, it can weigh one
expert against the others, for example trusting expert 2 on the inputs where
its features look like the classes it saw most. $W$ and $b$ start at zero, so
every $g_i$ starts at $1/M$ and step 0 is bagging, as with an untrained
channel. Construction checks: the weights sum to 1 for every example, and the
answer the metrics rebuild from the experts' logits and $g$ is the model's.

## 3. Size

The gate has $M (n_{\text{read}} + 1)\, r$ weights in its readings and
$(M r + 1)\, M$ in its scores. On cnn3 experts with $M = 3$ and $r = 8$:

| `--gate-read` | $n_{\text{read}}$ | whole gate | per expert |
|---|---:|---:|---:|
| `stage1` | 3,136 | 75,363 | 25,121 |
| `stage2` | 1,568 | 37,731 | 12,577 |
| `stage3` | 576 | 13,923 | 4,641 |

For comparison, a level-1 channel of the channel ladder is 18,544 to 39,024
weights per expert, depending on its points ([Channel](channel.md) §5).

## 4. What it can and can't change

The gate only chooses how much to trust each expert on each input. It never
changes an expert's own answer, so it can't correct an expert's own lean, such
as the class skew it was trained with. That is what each expert's own add-on
can do in a channel's `self` control, so the two baselines answer different
questions: whether weighing the experts is enough, and whether fixing each one
is.

Because no expert's answer changes, these metrics stay at their step-0 values
throughout a run: `expert_accuracy`, the agreement in `joint_answer_shift`
(`agreement_post` equals `agreement_pre`), `marginal_answer_shift` (0) and
`answer_shift_kl` (0). The exchange metrics are absent, since nothing is
exchanged. `grad_norm` and `update_ratio` report the gate as `gate.read` and
`gate.score`.

## 5. Running it

```bash
uv run python -m testbed.run --arm moe --gate linear_taper_gelu --gate-read stage3 \
  --trial-seeds 0 1 2
```

`--gate` replaces `--exchange`, and the two can't be combined. `--loss
per_expert` is refused, since no expert's answer depends on the gate. MLflow
tags the runs with `gate`, and logs the gate's architecture, read point and
width as params. In a study, run it through the study's own runner so it gets
the study's defaults, as the [channel ladder](../studies/channel-ladder.md)
does.

## 6. Other gates

Not built:

- a gate that also reads each expert's answer probabilities, which carry its
  confidence cheaply;
- fixed learned weights on the experts' answers, the same for every input:
  this gate with nothing to read.
