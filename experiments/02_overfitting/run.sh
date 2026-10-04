#!/usr/bin/env bash
# Overfitting ablation on film_p0 (film decoder, pooled, k_rounds 1), one
# fixed expert group, first 3 joint seeds. film_p0 is the "none" arm. Run one
# arm at a time:
#   bash experiments/02_overfitting/run.sh freeze_fc
#   bash experiments/02_overfitting/run.sh early_stop
#   bash experiments/02_overfitting/run.sh member_loss
#   bash experiments/02_overfitting/run.sh all
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_DIR=experiments/02_overfitting

SPLIT_FILE=splits/three_way_seed42.pt
MANIFEST=selection/manifests/encoder_study_group.json
EPOCHS=70
K_ROUNDS=1
SEEDS=3

arm="${1:-}"
case "$arm" in
  freeze_fc)   flags=(--freeze-fc) ;;
  early_stop)  flags=(--patience 10) ;;
  member_loss) flags=(--member-loss 1) ;;
  all)         flags=(--freeze-fc --patience 10 --member-loss 1) ;;
  *)
    echo "usage: $0 {freeze_fc|early_stop|member_loss|all}" >&2
    exit 1
    ;;
esac

uv run python scripts/replicate_joint.py \
  --split-file "$SPLIT_FILE" \
  --experiment "reg_$arm" \
  --limit "$SEEDS" \
  --state-file "$EXP_DIR/state/reg_$arm.json" \
  -- \
  communicative \
  --manifest "$MANIFEST" \
  --decoder film \
  --epochs "$EPOCHS" \
  --k-rounds "$K_ROUNDS" \
  --output-dir "$EXP_DIR/checkpoints" \
  "${flags[@]}"
