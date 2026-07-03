#!/bin/bash
# Run the 8 Arm F shards (4 lo-bounty + 4 hi-bounty) as fully detached
# processes so they survive the launching session. PIDs are written to
# results/llm-decision-market/arm_f_pids.txt; each shard writes its report
# only on completion, its raw LLM calls incrementally to
# experiments/llm-decision-market/logs/calls_f{lo,hi}N_*.jsonl.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCEN="$ROOT/experiments/llm-decision-market/scenarios_v1"
OUT="$ROOT/results/llm-decision-market"
mkdir -p "$OUT"
: > "$OUT/arm_f_pids.txt"

for tag in lo hi; do
  for i in 0 1 2 3; do
    nohup "$ROOT/.venv/bin/python" -m proposal_poker.simulate \
      --scenario "$SCEN/arm_f_${tag}_shard${i}.json" \
      --extensions-dir "$ROOT/experiments/llm-decision-market/submissions" \
      --output "$OUT/v1_arm_f_${tag}_shard${i}_report.json" \
      > "$OUT/v1_arm_f_${tag}_shard${i}.log" 2>&1 < /dev/null &
    echo "f_${tag}_shard${i} $!" >> "$OUT/arm_f_pids.txt"
  done
done
disown -a 2>/dev/null || true
cat "$OUT/arm_f_pids.txt"
