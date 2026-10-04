#!/usr/bin/env bash
# Attention masking study, one fixed expert group, first 3 joint seeds. Run one
# arm at a time, e.g.:
#   bash experiments/03_attention_masking/run.sh film_others
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_DIR=experiments/03_attention_masking

SPLIT_FILE=splits/three_way_seed42.pt
MANIFEST=selection/manifests/encoder_study_group.json
EPOCHS=70
PATIENCE=10
SEEDS=3
UNFREEZE=(--unfreeze layer1.2 layer1.3 layer1.4)

arm="${1:-}"
case "$arm" in
  mlp_all)               flags=(--decoder mlp --attn all) ;;
  mlp_others)            flags=(--decoder mlp --attn others) ;;
  mlp_self)              flags=(--decoder mlp --attn self) ;;
  film_all)              flags=(--decoder film --attn all) ;;
  film_others)           flags=(--decoder film --attn others) ;;
  film_self)             flags=(--decoder film --attn self) ;;
  mlp_all_unfreeze)      flags=(--decoder mlp --attn all "${UNFREEZE[@]}") ;;
  mlp_others_unfreeze)   flags=(--decoder mlp --attn others "${UNFREEZE[@]}") ;;
  film_all_unfreeze)     flags=(--decoder film --attn all "${UNFREEZE[@]}") ;;
  film_others_unfreeze)  flags=(--decoder film --attn others "${UNFREEZE[@]}") ;;
  k0_unfreeze)           flags=(--k-rounds 0 "${UNFREEZE[@]}") ;;
  *)
    echo "usage: $0 {mlp|film}_{all|others|self} | {mlp|film}_{all|others}_unfreeze | k0_unfreeze" >&2
    exit 1
    ;;
esac

uv run python scripts/replicate_joint.py \
  --split-file "$SPLIT_FILE" \
  --experiment "mask_$arm" \
  --limit "$SEEDS" \
  --state-file "$EXP_DIR/state/mask_$arm.json" \
  -- \
  communicative \
  --manifest "$MANIFEST" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --freeze-fc \
  --output-dir "$EXP_DIR/checkpoints" \
  "${flags[@]}"
