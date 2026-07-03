#!/bin/bash
# Run Arm B (or the Sonnet subsample) shards concurrently.
# Usage: run_arm_b.sh [b|sonnet]
set -u
KIND="${1:-b}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCEN="$ROOT/experiments/llm-decision-market/scenarios"
OUT="$ROOT/results/llm-decision-market"
mkdir -p "$OUT"

if [ "$KIND" = "b" ]; then
  PATTERN="arm_b_shard"
else
  PATTERN="arm_b_sonnet_shard"
fi

pids=()
for scenario in "$SCEN/$PATTERN"*.json; do
  name="$(basename "$scenario" .json)"
  "$ROOT/.venv/bin/python" -m proposal_poker.simulate \
    --scenario "$scenario" \
    --extensions-dir "$ROOT/experiments/llm-decision-market/submissions" \
    --output "$OUT/${name}_report.json" \
    > "$OUT/${name}.log" 2>&1 &
  pids+=($!)
  echo "launched $name pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all shards finished (fail=$fail)"
exit $fail
