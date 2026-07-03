# STATE — Arm G (aligned liar) ablation

Branch: llm-decision-market-v0. Owner authorized free merging to main.

## Goal
Arm G = Arm F hi-bounty (bounty 40, covert, proposals 0-49, v1 env, seed 777)
except the bribed seat's prompt instructs it to ALIGN its stated beliefs with
its stakes (lie with words AND money), defeating the stake-vs-belief audit.
Measure: (1) decision accuracy delta, (2) poll accuracy incl. the liar's
stated beliefs (key comparison: does the poll degrade more than the market?),
(3) stake-belief audit detection G vs F, (4) residual signatures
(belief-vs-population divergence, PnL, Brier decalibration of liar reports).
Budget ≤900 calls (750 main + 75 smoke).

## Plan / progress
- [x] Orient: RESULTS.md, analyze_v1.py, plugins, run_arm_f.sh read.
- [x] Plugin: submissions/agents/llm_market_liar.py (extends bribed agent,
      adds aligned-report instruction; agent_type llm_market_liar in logs).
- [x] Scenarios: make_scenarios_v1.py --arm-g 40.0 → arm_g_shard{0..3}.json
      + arm_g_smoke.json (5 proposals, 5 agents, 3 rounds).
- [x] Smoke validated on proposals 0-3 (53 calls, 0 parse failures, 0 errors;
      liar states 0.58 on signal -1.325 while staking YES — lies with words;
      one round-0 slip: stated 0.38 with YES stake, i.e. compliance imperfect
      but strong; network flaky: one call took 725 s through retries).
      Smoke tail (pid 4141) allowed to finish concurrently with full run.
- [~] Full run IN PROGRESS: run_arm_g.sh launched (~18:25 local), pids
      23168-23171 (arm_g_pids.txt), reports v1_arm_g_shard{0..3}_report.json
      appear on completion, logs calls_g{0..3}_*.jsonl (15 lines/proposal;
      shards carry 13/13/13/11 proposals).
- [x] Analysis: analyze_v1.py extended with arm_G block (decision metrics vs
      F-hi + baseline slice; poll-on-slice for baseline/F-hi/G; audit
      detection; divergence/PnL/Brier residual monitors). Regression-ran on
      existing data: metrics_v1.json unchanged, arm_G no-ops until reports
      exist.
- [ ] RESULTS.md "Arm G — aligned liar ablation" section + verdict.
- [ ] Copy raw logs into results/llm-decision-market/raw_llm_logs (v0
      convention: every prompt/response committed).
- [ ] Commit, push, merge to main.

## How to resume after interruption
- Check detached runs: `cat results/llm-decision-market/arm_g_pids.txt`,
  `ls -la results/llm-decision-market/v1_arm_g_*` ; shard reports appear only
  on completion; progress = line counts of
  experiments/llm-decision-market/logs/calls_g*_*.jsonl (15 lines/proposal).
- Network is flaky: agent retries transport errors 3x with 60 s waits. If a
  shard dies, relaunch just that shard with the same command from
  run_arm_g.sh (env draws are deterministic).
- Baselines for comparison: metrics_v1.json arm_F.hi block; baseline slice =
  v1 Arm B proposals 0-49 (accuracy 0.980).
