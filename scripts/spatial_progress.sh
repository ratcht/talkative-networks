#!/usr/bin/env bash
# One-screen status for Study 02: which arms are done, which seed is running,
# and the last epoch line from the arm in flight.
cd "$(dirname "$0")/.."
ARMS=(gap_broadcast pool_mask conv_convt gap_mask gap_convt
      pool_broadcast conv_broadcast flat_broadcast pool_convt conv_mask)

printf '%-16s %-28s %s\n' ARM SEEDS LAST
for arm in "${ARMS[@]}"; do
  state="scripts/replicate_state/spatial_${arm}.json"
  if [[ ! -f "$state" ]]; then
    printf '%-16s %-28s %s\n' "$arm" "-" "not started"
    continue
  fi
  summary=$(python3 - "$state" <<'PY'
import json, sys
from collections import Counter
s = json.load(open(sys.argv[1]))
seeds = s.get("seeds", s.get("runs", {}))
if isinstance(seeds, dict):
    c = Counter(v.get("status", v) if isinstance(v, dict) else v for v in seeds.values())
else:
    c = Counter()
print(" ".join(f"{k}={v}" for k, v in sorted(c.items())) or "-")
PY
)
  last=$(tr '\r' '\n' < "logs/spatial_study/${arm}.log" 2>/dev/null \
         | grep -E "^epoch |^\[[0-9]+/5\]" | tail -1)
  printf '%-16s %-28s %s\n' "$arm" "$summary" "${last:0:70}"
done

echo
running=$(pgrep -fc "ensemble/train.py.*spatial_" || true)
echo "training processes: ${running:-0}"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
