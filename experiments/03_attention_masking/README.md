# 03 Attention masking

## Question

- Is the channel communicating, or are receivers mostly reading their own message back?
- Additive or multiplicative (FiLM) injection, compared under the same setup?
- Can receivers use messages better if the blocks right after the injection point are trainable?

## Changes

- `--attn {all,others,self}`: mask on the bus attention.
  - `all`: each receiver attends to every member, itself included (previous behaviour).
  - `others`: self is masked out. Any gain must come from other members.
  - `self`: only self. No information from other members, so this is a self-loop control.
- `--unfreeze layer1.2 layer1.3 layer1.4`: the 3 blocks that first process the injected message become trainable (conv weights and BatchNorm scale/shift, 42,048 parameters total). BatchNorm running stats stay fixed.

## Setup

- Experts: `selection/manifests/encoder_study_group.json`
- Encoder `qkv`, `key_dim 16`, `value_dim 64`, taps `layer1.2` / `layer3.0`, patch 0, K = 1
- `--freeze-fc`, `--patience 10`, no member loss
- 70 epochs max, AdamW lr 1e-3, weight decay 1e-4
- First 3 joint seeds: 476981973, 621715752, 1882162633
- Trainable parameters: `mlp` 34,320, `film` 37,440, plus 42,048 with `--unfreeze`

## Arms

| Arm | Decoder | Attention | Unfreeze | K |
|---|---|---|---|---|
| `mask_mlp_all` | mlp | all | | 1 |
| `mask_mlp_others` | mlp | others | | 1 |
| `mask_mlp_self` | mlp | self | | 1 |
| `mask_film_all` | film | all | | 1 |
| `mask_film_others` | film | others | | 1 |
| `mask_film_self` | film | self | | 1 |
| `mask_mlp_all_unfreeze` | mlp | all | yes | 1 |
| `mask_mlp_others_unfreeze` | mlp | others | yes | 1 |
| `mask_film_all_unfreeze` | film | all | yes | 1 |
| `mask_film_others_unfreeze` | film | others | yes | 1 |
| `mask_k0_unfreeze` | | | yes | 0 |

## Baselines

- Arms without unfreeze: untrained averaging, val 0.6686, test 0.6652. With `fc` frozen this equals the K = 0 control.
- Arms with unfreeze: `mask_k0_unfreeze`. The unfrozen blocks can improve accuracy without messages, so the communication effect is `(arm) - (k0_unfreeze)`.

## How to read it

| Result | Meaning |
|---|---|
| `others` above baseline, `self` near baseline | members use information from each other |
| `self` close to `all`, `others` near baseline | the gain in `all` is a self-loop, not communication |
| unfreeze arms above `k0_unfreeze`, frozen arms near baseline | frozen receivers cannot use messages, trainable ones can |
| `film` above `mlp` within the same attention mode | multiplicative injection is better |

## Results

Not yet run.
