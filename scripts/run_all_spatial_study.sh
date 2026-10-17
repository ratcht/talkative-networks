#!/usr/bin/env bash
# Run every arm of Study 02 back to back, detached, so the sequence survives
# the SSH session that started it.
#
#   bash scripts/run_all_spatial_study.sh          # detach and return
#   bash scripts/run_all_spatial_study.sh --fg     # run in the foreground
#   bash scripts/run_all_spatial_study.sh --dry-run
#
# Detaching is the default and is done with setsid, which puts the whole
# sequence in its own session and process group. That matters beyond surviving
# a hangup: during Study 01 a sequence was killed at the terminal, the parent
# replicate_joint.py died, and one orphaned ensemble/train.py child ran to
# completion — leaving its seed recorded as "running" forever while it quietly
# finished and wrote a checkpoint. Owning the process group means a later
# `kill -- -<pgid>` takes the children down with the parent.
#
# Arms run in sequence, not in parallel: they share one GPU, and the point of
# the study is comparing arms trained under identical conditions (§4.1).
#
# Resumable. replicate_joint.py keeps per-seed state in
# scripts/replicate_state/<experiment>.json and skips seeds already `done`, so
# re-running after an interruption picks up where it stopped.
#
# Order is not required by the analysis — every arm is read against gap x
# broadcast, so §3.3's contrasts do not depend on it. It is chosen so that an
# interrupted sequence has still answered something: the reference arm first,
# since its construction values validate the measurement and nothing else is
# interpretable until they hold (§6.1); then the matched-access diagonal, the
# most direct test of §1; then the two single-port columns; then the control
# and the remaining off-diagonal cells.

set -uo pipefail   # deliberately not -e: one failing arm should not silently
                   # abandon the nine that could still run
# absolute before the cd: the detach below re-execs this file by path, and a
# relative $0 stops resolving once the working directory changes
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

ARMS=(
  gap_broadcast                   # E1xD1 — reference, and the harness check
  pool_mask conv_convt            # the diagonal
  gap_mask gap_convt              # write side, read access held at zero
  pool_broadcast conv_broadcast   # read side, write access held at zero
  flat_broadcast                  # E4 — convolutional prior against locality
  pool_convt conv_mask            # the remaining off-diagonal cells
)

LOG_DIR="logs/spatial_study"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/_sequence_$STAMP.log"

DRY_RUN=""
FOREGROUND=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --fg|--foreground) FOREGROUND=1 ;;
    *) echo "usage: $0 [--fg] [--dry-run]" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

# Refuse to run twice over the same state. Two sequences share
# scripts/replicate_state/*.json and one Aim experiment per arm, so a second
# one marks seeds "running" that the first owns and both write duplicate runs.
LOCK="$LOG_DIR/.sequence.lock"
holder="$(cat "$LOCK/pid" 2>/dev/null || true)"
if [[ -z "$DRY_RUN" && -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
  echo "[spatial-study] a sequence is already running (pid $holder)." >&2
  echo "[spatial-study] Two sequences share scripts/replicate_state/*.json and one Aim" >&2
  echo "[spatial-study] experiment per arm, so the second would mark seeds \"running\"" >&2
  echo "[spatial-study] that the first owns and both would write duplicate runs." >&2
  echo "[spatial-study] Stop it with: kill -- -\$(ps -o pgid= -p $holder | tr -d ' ')" >&2
  exit 1
fi

# Re-exec detached unless already detached or explicitly foregrounded. The
# guard variable is what stops the re-exec from recursing.
if [[ -z "$FOREGROUND" && -z "$DRY_RUN" && -z "${SPATIAL_STUDY_DETACHED:-}" ]]; then
  export SPATIAL_STUDY_DETACHED=1
  setsid nohup bash "$SELF" --fg ${DRY_RUN:+--dry-run} > "$MASTER_LOG" 2>&1 < /dev/null &
  disown
  sleep 2
  pgid=$(cat "$LOCK/pgid" 2>/dev/null || true)
  cat <<EOF
[spatial-study] detached — safe to close this SSH session.

  arms      : ${#ARMS[@]} arms, 5 seeds each — 50 runs
  log       : $REPO_ROOT/$MASTER_LOG
  per-arm   : $REPO_ROOT/$LOG_DIR/<arm>.log
  pgid      : ${pgid:-unknown}

  follow    : tail -f $MASTER_LOG        (Ctrl-C stops watching, not training)
  progress  : cat scripts/replicate_state/spatial_*.json
  stop all  : kill -- -${pgid:-PGID}
EOF
  exit 0
fi

# the long-running process is the one that holds the lock, and its EXIT trap
# is the only thing that releases it
if [[ -z "$DRY_RUN" ]]; then
  rm -rf "$LOCK"
  mkdir -p "$LOCK"
  echo "$$" > "$LOCK/pid"
  ps -o pgid= -p $$ | tr -d ' ' > "$LOCK/pgid"
  trap 'rm -rf "$LOCK"' EXIT
fi

echo "[spatial-study] start $(date -Is)"
echo "[spatial-study] repo: $REPO_ROOT"
echo "[spatial-study] arms: ${ARMS[*]}"
[[ -n "$DRY_RUN" ]] && echo "[spatial-study] DRY RUN — no training will run"

declare -A RESULT
overall_start=$SECONDS

for arm in "${ARMS[@]}"; do
  arm_log="$LOG_DIR/${arm}.log"
  echo
  echo "======================================================================"
  echo "[spatial-study] arm '$arm'  $(date -Is)"
  echo "[spatial-study] log: $arm_log"
  echo "======================================================================"

  start=$SECONDS
  if [[ -n "$DRY_RUN" ]]; then
    # --dry-run is forwarded to replicate_joint.py, which prints the exact
    # per-seed commands without touching state or the GPU
    bash scripts/run_spatial_study.sh "$arm" --dry-run 2>&1 | tee -a "$arm_log"
    rc=${PIPESTATUS[0]}
  else
    bash scripts/run_spatial_study.sh "$arm" > "$arm_log" 2>&1
    rc=$?
  fi
  elapsed=$(( SECONDS - start ))

  if [[ $rc -eq 0 ]]; then
    RESULT[$arm]="ok"
    echo "[spatial-study] arm '$arm' finished in $(( elapsed / 60 )) min"
  else
    RESULT[$arm]="FAILED (rc=$rc)"
    echo "[spatial-study] arm '$arm' FAILED rc=$rc after $(( elapsed / 60 )) min"
    echo "[spatial-study] continuing to the next arm; see $arm_log"
  fi
done

echo
echo "======================================================================"
echo "[spatial-study] all arms attempted  $(date -Is)"
echo "[spatial-study] total $(( (SECONDS - overall_start) / 60 )) min"
echo "======================================================================"
failed=0
for arm in "${ARMS[@]}"; do
  printf '  %-18s %s\n' "$arm" "${RESULT[$arm]}"
  [[ "${RESULT[$arm]}" == ok ]] || failed=1
done

if [[ $failed -eq 1 ]]; then
  echo
  echo "[spatial-study] at least one arm failed. Re-running this script retries"
  echo "[spatial-study] only the unfinished seeds — completed ones are skipped."
  exit 1
fi
echo
echo "[spatial-study] every arm complete. Aggregates are in Aim, one summary run"
echo "[spatial-study] per experiment: ${ARMS[*]/#/spatial_}"
