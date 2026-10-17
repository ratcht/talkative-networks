#!/usr/bin/env bash
# Encoder architecture study — one arm at a time, over one fixed expert group
# (selection/manifests/encoder_study_group.json), replicated across the split
# file's joint_seeds:
#   bash scripts/run_encoder_study.sh baseline
#   bash scripts/run_encoder_study.sh mlp_no_norm
#
# To run every arm back to back and detached, use run_all_encoder_studies.sh —
# it calls this script once per arm.
#
#   arm               | encoder   | key | value | isolates
#   baseline          | qkv       |  16 |    64 | reference
#   wide              | qkv       |  64 |   128 | capacity
#   mlp               | mlp       |  16 |    64 | expressiveness
#   mlp_wide          | mlp       |  64 |   128 | both
#   mlp_no_norm       | mlp_no_ln |  16 |    64 | expressiveness minus LayerNorm
#   mlp_no_norm_wide  | mlp_no_ln |  64 |   128 | both minus LayerNorm
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_FILE=splits/three_way_seed42.pt
MANIFEST=selection/manifests/encoder_study_group.json
EPOCHS=70
K_ROUNDS=1

arm="${1:-}"
shift || true
# anything after the arm name is forwarded to replicate_joint.py's own argument
# block (before the `--`), so --dry-run / --limit / --aggregate-only work here
EXTRA=("$@")
case "$arm" in
  baseline)
    experiment=encoder_baseline
    encoder=qkv; key_dim=16; value_dim=64
    ;;
  wide)
    experiment=encoder_wide
    encoder=qkv; key_dim=64; value_dim=128
    ;;
  mlp)
    experiment=encoder_mlp
    encoder=mlp; key_dim=16; value_dim=64
    ;;
  mlp_wide)
    experiment=encoder_mlp_wide
    encoder=mlp; key_dim=64; value_dim=128
    ;;
  # mlp minus the trunk's LayerNorm — separates the nonlinearity's contribution
  # from the normalization that rides along with it in the mlp arms
  mlp_no_norm)
    experiment=encoder_mlp_no_norm
    encoder=mlp_no_ln; key_dim=16; value_dim=64
    ;;
  mlp_no_norm_wide)
    experiment=encoder_mlp_no_norm_wide
    encoder=mlp_no_ln; key_dim=64; value_dim=128
    ;;
  *)
    echo "usage: $0 {baseline|wide|mlp|mlp_wide|mlp_no_norm|mlp_no_norm_wide}" >&2
    exit 1
    ;;
esac

uv run python scripts/replicate_joint.py \
  --split-file "$SPLIT_FILE" \
  --experiment "$experiment" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -- \
  communicative \
  --manifest "$MANIFEST" \
  --encoder "$encoder" \
  --key-dim "$key_dim" \
  --value-dim "$value_dim" \
  --epochs "$EPOCHS" \
  --k-rounds "$K_ROUNDS"
