#!/usr/bin/env bash
# Patch attention study, one fixed expert group, first 3 joint seeds. Run one
# arm at a time, named p{patch}_{decoder}_{inject}[_pattn], e.g.:
#   bash experiments/04_patch_attention/run.sh p2_linear_replace_pattn
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_DIR=experiments/04_patch_attention

SPLIT_FILE=splits/three_way_seed42.pt
MANIFEST=selection/manifests/encoder_study_group.json
EPOCHS=70
PATIENCE=10
SEEDS=3
LATE_CH=64  # channels at the layer3.0 output

arm="${1:-}"
if [[ ! "$arm" =~ ^p([012])_(linear|mlp)_(add|replace)(_pattn)?$ ]]; then
  echo "usage: $0 p{0|1|2}_{linear|mlp}_{add|replace}[_pattn]" >&2
  exit 1
fi
patch="${BASH_REMATCH[1]}"
flags=(--patch "$patch" --decoder "${BASH_REMATCH[2]}" --inject "${BASH_REMATCH[3]}")
[[ -n "${BASH_REMATCH[4]}" ]] && flags+=(--patch-attn)

# qkv_eye needs value size = size of one patch
value_dim=$(( LATE_CH * (patch > 0 ? patch * patch : 1) ))

uv run python scripts/replicate_joint.py \
  --split-file "$SPLIT_FILE" \
  --experiment "pa_$arm" \
  --limit "$SEEDS" \
  --state-file "$EXP_DIR/state/pa_$arm.json" \
  -- \
  communicative \
  --manifest "$MANIFEST" \
  --encoder qkv_eye \
  --value-dim "$value_dim" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --freeze-fc \
  --output-dir "$EXP_DIR/checkpoints" \
  "${flags[@]}"
