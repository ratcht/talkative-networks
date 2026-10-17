# 04 Second attention block across patches (plan)

## Question

Does a second attention block that mixes across patches help? And how do the decoder (linear vs MLP) and the injection (additive vs replace) change the result?

## Words used below

| Word | Meaning |
|---|---|
| member | one frozen backbone in the ensemble. We use 3. |
| read point | where the hidden state is taken from: output of `layer3.0`, shape 64 channels × 8 × 8. |
| write point | where the message goes back in: input of `layer1.2`, shape 16 channels × 32 × 32. |
| patch | one piece of the read-point grid, set by `--patch`. |
| round | one full forward pass of every backbone. K = number of communication rounds. |

## Architecture (K = 1, so 2 passes)

Members only meet in block 1. Every other step is per member.

```mermaid
flowchart TD
  img["image, same for every member"]

  subgraph P0["pass 0: no message yet"]
    F1["member 1: image → layer3.0<br/>nothing injected"]
    F2["member 2: image → layer3.0<br/>nothing injected"]
    F3["member 3: image → layer3.0<br/>nothing injected"]
  end

  img --> F1 & F2 & F3

  F1 -->|"read: layer3.0 output<br/>(64, 8, 8)"| S1["member 1: reshape by --patch<br/>(P, d)"]
  F2 -->|read| S2["member 2: reshape<br/>(P, d)"]
  F3 -->|read| S3["member 3: reshape<br/>(P, d)"]

  S1 & S2 & S3 --> BL1["block 1: attention across members<br/>each patch of each receiver attends to the same patch of all 3 members, self included<br/>(P, n, d)"]

  BL1 --> C1["member 1: block 2<br/>attention across its own P patches<br/>(on/off arm)"]
  BL1 --> C2["member 2: block 2"]
  BL1 --> C3["member 3: block 2"]

  C1 --> D1["member 1: decoder<br/>linear or MLP<br/>(16, 32, 32)"]
  C2 --> D2["member 2: decoder"]
  C3 --> D3["member 3: decoder"]

  subgraph P1["pass 1"]
    E1["member 1: image → up to layer1.2"] --> W1["member 1: write at input of layer1.2<br/>additive or replace"] --> G1["member 1: layer1.2 → logits"]
    E2["member 2: image → up to layer1.2"] --> W2["member 2: write at input of layer1.2"] --> G2["member 2: layer1.2 → logits"]
    E3["member 3: image → up to layer1.2"] --> W3["member 3: write at input of layer1.2"] --> G3["member 3: layer1.2 → logits"]
  end

  img --> E1 & E2 & E3
  D1 -->|pass-0 message| W1
  D2 -->|pass-0 message| W2
  D3 -->|pass-0 message| W3

  G1 & G2 & G3 --> AVG["average softmax probabilities<br/>final prediction"]
```

- The message written in a pass is always the one built in the **previous** pass. Pass 0 has nothing to write.
- With K > 1, each pass also reads at `layer3.0` again (now affected by the message) and builds the message for the next pass. The last pass only produces logits.
- When block 2 is off, block 1's output goes straight to the decoders.

Final prediction: average of the 3 members' softmax probabilities from the last pass (same as now).

## Shapes, step by step

| Symbol | Meaning |
|---|---|
| b | batch size |
| n | number of members (3) |
| P | number of patches per member, set by `--patch` |
| d | size of each patch's vector, set by `--patch` |
| d_k | query/key size (`--key-dim`, 16) |

| `--patch` | What happens to the 64 × 8 × 8 state | P | d |
|---|---|---|---|
| 0 | average over the whole grid | 1 | 64 |
| 1 | every grid cell is a patch | 64 | 64 |
| 2 | 2 × 2 blocks, each flattened | 16 | 256 |

Block 1, across members:

| Step | Operation | Shape |
|---|---|---|
| 1 | stack members' reshaped states | h: (b, P, n, d) |
| 2 | Q = h·W_q, K = h·W_k, V = h·W_v | Q, K: (b, P, n, d_k), V: (b, P, n, d) |
| 3 | A = softmax over the n members of Q·Kᵀ / √d_k | A: (b, P, n, n) |
| 4 | m = A·V | m: (b, P, n, d) |

Block 2, across patches (input is block 1's output m):

| Step | Operation | Shape |
|---|---|---|
| 5 | Q₂ = m·W_q2, K₂ = m·W_k2, V₂ = m·W_v2 | Q₂, K₂: (b, n, P, d_k), V₂: (b, n, P, d) |
| 6 | A₂ = softmax over the receiver's P patches of Q₂·K₂ᵀ / √d_k | A₂: (b, n, P, P) |
| 7 | m′ = A₂·V₂ | m′: (b, P, n, d) |

Then:

| Step | Operation | Shape |
|---|---|---|
| 8 | decoder of receiver i maps its message to the write-point shape | (b, 16, 32, 32) |
| 9 | additive: `x + decoder output`. replace: `decoder output` instead of x | |

## Init

| Weight | Init |
|---|---|
| W_q, W_k (both blocks) | standard (PyTorch `nn.Linear`) |
| W_v (both blocks) | identity, shape d × d, learned. At step 0, V = the hidden state itself. |
| decoder, additive arms | last layer zero, so step 0 = the plain ensemble |
| decoder, replace arms | standard, since zero would wipe the receiver's state |

Decoders: linear = one `nn.Linear`. MLP = the current `Linear → GELU → Linear`.

## Arms

| patch | block 2 | decoder | injection | arms |
|---|---|---|---|---|
| 0 | off | linear, MLP | additive | 2 |
| 1 | on, off | linear, MLP | additive, replace | 8 |
| 2 | on, off | linear, MLP | additive, replace | 8 |

18 arms × 3 seeds = 54 runs.

- Patch 0 has no block 2: with one patch there is nothing to mix across.
- Patch 0 has no replace: its message is one 64-vector with no spatial layout, so it cannot replace a 16 × 32 × 32 state without either copying 16 values to every pixel or a ~1M-parameter decoder per member.

## Training setup (from experiment 03)

- Experts: `selection/manifests/encoder_study_group.json`
- Encoder `qkv_eye`, `key_dim 16`, taps `layer1.2` / `layer3.0`, K = 1
- Backbones frozen, `--freeze-fc`, `--patience 10`, 70 epochs max
- AdamW lr 1e-3, weight decay 1e-4
- Seeds: 476981973, 621715752, 1882162633

## Baseline

K = 0 (no communication) = untrained probability averaging: test 0.6652.

## Code changes

- `bus.py`: `EyeQKVEncoder` (`--encoder qkv_eye`, W_v initialized to identity, needs `--value-dim` = d). `PatchAttention` for block 2. `LinearDecoder` (`--decoder linear`).
- `hooks.py`: one `hook_inject` with `INJECTIONS` (`--inject {add,film,replace}`). Each mode sets how the message combines with the input, the decoder output width, and the decoder init. FiLM moved here from `--decoder film`.
- `train.py`: flag `--patch-attn`.
- `train.py` logging: block 1 attention metrics as now. Block 2 attention has shape (b, n, P, P), so only its entropy is tracked. Block 2 gets its own `grad_norm` component (`patch_attn`).
- `experiments/04_patch_attention/`: README, `run.sh`, `commands.md`.
