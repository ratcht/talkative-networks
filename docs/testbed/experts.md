# Experts

This page defines v4's experts: the residual stream every channel reads and
writes, the two expert families, the recipe each trains with, and the data.
The code is in `testbed/experts/` and `testbed/data.py`. How a pool of
experts comes to differ (seeds, class skew and the rest) is unchanged from
v3, in `testbed/experts/schemes.py`.

## 1. The residual stream

Every v4 expert keeps one tensor, the residual stream, from its first block
to its last. An embedding stage starts the stream, and every block adds its
output to it:

$$
x_0 = \mathrm{embed}(\text{image}), \qquad
x_\ell = x_{\ell-1} + f_\ell(x_{\ell-1}) \quad (\ell = 1, \dots, L), \qquad
\text{answer} = \mathrm{head}(x_L).
$$

So the stream has one shape at every point. The points are named after the
stage that produced them: `embed` is $x_0$, `block3` is $x_3$, and so on to
`block12`. **A channel reads and writes only there, in the stream.** Its
encoders read $x_\ell$, and its decoders' injections are added to $x_\ell$, so
the channel uses the same skip connection as the blocks.
`build_ensemble` checks this: it refuses experts whose points don't all share
one shape (`Expert.stream_shape`).

The stream is handed over channels first, `(B, C, *positions)`, in both
families, so a channel part that reads positions treats a ViT's token like a
ResNet's grid cell:

| family | stream at every point | positions |
|---|---|---|
| ResNet $L \times w$ | $w \times 16 \times 16$ | the 256 grid cells |
| ViT $L \times d$ | $d \times 65$ | the class token first, then the 64 patches |

The shape need only be uniform within one family's experts; a ResNet's
stream and a ViT's are never mixed in one group.

## 2. The two families

| | `resnet` | `vit` |
|---|---|---|
| Embedding | $3\times3$ convolution with stride side/16 | $4\times4$ patches (side/8) embedded linearly, a class token, learned positions |
| Block | $x + \mathrm{conv}(\mathrm{ReLU}(\mathrm{BN}(\mathrm{conv}(\mathrm{ReLU}(\mathrm{BN}(x))))))$, $3\times3$ convolutions | $x + \mathrm{attn}(\mathrm{LN}(x))$, then $x + \mathrm{MLP}(\mathrm{LN}(x))$; MLP $4\times$, heads of width 32 |
| Head | BN, ReLU, mean over the grid, linear | layer norm on the class token, linear |
| Default size | $12 \times 64$: 896,164 parameters | $12 \times 96$: 1,363,012 parameters |

On CIFAR-100. The ResNet is an *isotropic* pre-activation ResNet (He et al.
2016): every block has the same width and grid. A standard ResNet's
downsampling stages would change the stream's shape, so it has none. The ViT
is the standard pre-norm vision transformer (Dosovitskiy et al. 2021). Both
sizes are the pair chosen by the v4 task-model selection (report in
`docs/reports/v4-traditional-pairs/`). Other sizes register with
`testbed.experts.register`, e.g.
`register("resnet_8x32", resnet, width=32, depth=8)`.

## 3. Recipes

Each family trains with its usual recipe, `architectures.RECIPES`. A recipe
is a `Budget` (`testbed/experts/recipes.py`), and the pool's cache key
includes it, so changing any part of a recipe trains new experts.

| | `resnet` | `vit` |
|---|---|---|
| Optimiser | SGD, Nesterov momentum 0.9 | AdamW (0.9, 0.999) |
| Learning rate, batch | 0.1, 128 | $10^{-3}$, 256 |
| Weight decay | $5\cdot10^{-4}$ on every parameter | 0.05, none on norms, biases, positions or the class token |
| Schedule | cosine to zero | 5% linear warm-up, then cosine to zero |
| Other | | label smoothing 0.1; gradient norm clipped at 1 |
| Epochs | 100 | 200 |
| Shared | random crop with padding side/8 (black), horizontal flip; bfloat16 autocast | |

The schedule, per optimiser step $s$ of $S$ in the whole run, with
$S_w = \lfloor \text{warmup} \cdot S \rfloor$ warm-up steps:

$$
\eta_s =
\begin{cases}
\eta \, \dfrac{s+1}{S_w} & s < S_w \\[1ex]
\dfrac{\eta}{2} \left(1 + \cos \dfrac{\pi (s - S_w)}{S - S_w}\right) & s \ge S_w .
\end{cases}
$$

The augmentation draws each image's crop offset and flip from the expert's
own `"augment"` seed stream, separate from its batch order, so turning it on
or off never changes which images an expert sees when. `Budget()` with no
arguments is v3's recipe: Adam at a constant $10^{-3}$, no augmentation, no
autocast. `--expert-epochs` changes only the epochs of the architecture's
recipe.

**Training time**, measured by the v4 selection's own scripts on the lab
machine with four models training at once (two per GPU): about 23 minutes per
45 ResNet experts and about 62 minutes per 45 ViT experts, each $\pm 30\%$.
This code has not been timed yet. It trains the same models with the same
recipes, but differs in ways that can change speed: it keeps the last partial
batch of an epoch, and it leaves out the selection's `channels_last` layout,
`cudnn.benchmark` and TF32 matrix multiplies. It also normalises per channel,
not by one scalar. Pools are trained once and cached, so the time is paid
once per scheme and trial seed.

## 4. Data

`--dataset cifar100` is the default; `cifar10`, `fashion` and `mnist` remain.
The split is v3's: per class, 50% expert, 40% joint, 10% joint_val, and the
official test set apart. Images are scaled to $[0, 1]$ and normalised per
channel by the expert split's mean and standard deviation; `Data.fill` is a
black pixel in those terms, which the crop pads with.

Joint training (the channel, or the MoE gate) uses no augmentation, as in v3.

## 5. Running

```bash
uv run python -m testbed.run --arm bagging --trial-seeds 0 1 2
uv run python -m testbed.run --arm forward --exchange block4 block8 --trial-seeds 0 1 2
uv run python -m testbed.run --arch vit --arm forward --exchange block4 block8 --trial-seeds 0 1 2
```

The expert scheme stays a flag (`--scheme`), with v3's default, class skew at
$\alpha = 1$. The v4 selection measured only seed-only experts
(`--scheme seed_only`); class skew over CIFAR-100's 100 classes is unmeasured.
