# STATE — Arm G (aligned liar) ablation: COMPLETE

Branch: llm-decision-market-v0. Owner authorized free merging to main.

## Status: done (2026-07-03)
- Smoke (75 calls) + full run (750 calls) complete: 825/900 budget,
  0 parse failures, 0 transport errors, mean latency 16.9 s.
- Reports: results/llm-decision-market/v1_arm_g_shard{0..3}_report.json,
  v1_arm_g_smoke_report.json. Metrics: metrics_v1.json `arm_G` block.
- Raw logs: experiments/llm-decision-market/logs/calls_g{0..3}_*.jsonl +
  calls_gsmoke_*.jsonl, copied into results/llm-decision-market/raw_llm_logs.
- RESULTS.md: "Arm G — aligned liar ablation" section appended (table, four
  measurements, verdict, limitations, reproduction).

## Headline results
- Market: 0.900 accuracy (baseline slice 0.980, F-hi 0.960); 4 flips, all
  x<0 pushed past the oracle window (prices 0.553-0.603, oracle never fired:
  3 invocations = baseline). Value regret 13.4 vs 0.84 (F-hi) vs 0.03.
- Poll incl liar reports: 0.960 (honest-only same run 0.980) — the poll
  degraded LESS than the market. Hypothesis (market more robust than poll
  under aligned lying) REFUTED at these parameters.
- Audit: stake-vs-stated-belief contradiction 0/118 (F-hi 49/127) — blind.
- Liar PnL: +6.1 market transfer (F-hi −51.2) + 1280 bounty — flipped
  decisions settle in the liar's favor (settlement endogeneity).
- Surviving monitors: Brier decalibration (0.195 vs 0.112 honest, paired
  t=2.81), belief-vs-population divergence (+0.085, t=2.29), one-sided book
  (673 YES / 2.5 NO). Dead: consistency audit, PnL.

## Remaining
- Commit + push + merge to main (in progress).
