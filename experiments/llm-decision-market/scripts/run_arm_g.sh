#!/bin/bash
# Run the 4 Arm G (aligned liar, bounty 40) shards as fully detached
# processes so they survive the launching session. PIDs are written to
# results/llm-decision-market/arm_g_pids.txt; each shard writes its report
# only on completion, its raw LLM calls incrementally to
# experiments/llm-decision-market/logs/calls_gN_*.jsonl.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCEN="$ROOT/experiments/llm-decision-market/scenarios_v1"
OUT="$ROOT/results/llm-decision-market"
mkdir -p "$OUT"
: > "$OUT/arm_g_pids.txt"

for i in 0 1 2 3; do
  nohup "$ROOT/.venv/bin/python" -m proposal_poker.simulate \
    --scenario "$SCEN/arm_g_shard${i}.json" \
    --extensions-dir "$ROOT/experiments/llm-decision-market/submissions" \
    --output "$OUT/v1_arm_g_shard${i}_report.json" \
    > "$OUT/v1_arm_g_shard${i}.log" 2>&1 < /dev/null &
  echo "g_shard${i} $!" >> "$OUT/arm_g_pids.txt"
done
disown -a 2>/dev/null || true
cat "$OUT/arm_g_pids.txt"
