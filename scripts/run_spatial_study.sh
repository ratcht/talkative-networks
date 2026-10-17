#!/usr/bin/env bash
# Study 02 — spatial access at the read and write ports. One arm at a time,
# over the same fixed expert group as Study 01
# (selection/manifests/encoder_study_group.json), replicated across the split
# file's joint_seeds:
#
#   bash scripts/run_spatial_study.sh gap_broadcast
#   bash scripts/run_spatial_study.sh conv_convt
#
# An arm is an (encoder, decoder) pair; neither adapter is runnable alone. The
# grid is the three treatment levels of each ladder crossed completely, plus
# the E4 control against D1 only — ten arms, 50 runs at five seeds. See
# docs/studies/02-spatial-ports.md §3.3 for why the off-diagonal cells are
# included and §4 for what is held fixed.
#
#   encoder | spatial access at the read port
#   --------+--------------------------------------------------
#   gap     | none, channel means only                      (E1)
#   pool    | input-independent learned mask                 (E2)
#   conv    | input-conditioned, convolutional prior         (E3)
#   flat    | input-conditioned, locality prior only         (E4)
#
#   decoder | spatial access at the write port
#   --------+--------------------------------------------------
#   broadcast | none, uniform per-channel bias               (D1)
#   mask      | input-independent shape, message amplitude   (D2)
#   convt     | message-conditioned shape                    (D3)
#
# Anything after the arm name is forwarded to replicate_joint.py's own argument
# block, so --dry-run / --limit / --aggregate-only work here.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_FILE=splits/three_way_seed42.pt
MANIFEST=selection/manifests/encoder_study_group.json
EPOCHS=70
K_ROUNDS=1
KEY_DIM=16     # §4.1: width is held fixed, Study 01 having found no effect
VALUE_DIM=64

ARMS=(
  gap_broadcast gap_mask gap_convt
  pool_broadcast pool_mask pool_convt
  conv_broadcast conv_mask conv_convt
  flat_broadcast
)

arm="${1:-}"
shift || true
EXTRA=("$@")

found=0
for a in "${ARMS[@]}"; do
  [[ "$a" == "$arm" ]] && found=1 && break
done
if [[ "$found" -ne 1 ]]; then
  echo "usage: $0 {$(IFS='|'; echo "${ARMS[*]}")} [replicate_joint flags...]" >&2
  exit 1
fi

# the arm name *is* the architecture: encoder before the first underscore,
# decoder after it. Keeping one name rather than two flags is what makes the
# experiment name, the checkpoint name and the arm label the same string.
encoder="${arm%%_*}"
decoder="${arm#*_}"

echo "[spatial] arm ${arm}: encoder=${encoder} decoder=${decoder} k${KEY_DIM}/v${VALUE_DIM}"

uv run python scripts/replicate_joint.py \
  --split-file "$SPLIT_FILE" \
  --experiment "spatial_${arm}" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -- \
  communicative \
  --manifest "$MANIFEST" \
  --encoder "$encoder" \
  --decoder "$decoder" \
  --key-dim "$KEY_DIM" \
  --value-dim "$VALUE_DIM" \
  --epochs "$EPOCHS" \
  --k-rounds "$K_ROUNDS"
