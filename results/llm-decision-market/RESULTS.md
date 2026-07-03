# LLM Agents in Decision Markets — Experiment v0

First controlled comparison of LLM agents in a *decision* market (approve/reject
proposals with participation costs, stake fees, and a costly verification
oracle), against a rational-agent market, a poll of the same LLMs, a
centralized LLM manager, and trivial baselines. Substrate: the Proposal Poker
engine (`mechanism-design/proposal-evaluation/MODEL.md`), branch
`llm-decision-market-v0`.

## Setup

- Environment: x ~ N(0,1) hidden quality, y ~ LogNormal(0,2) public importance,
  5 agents with wealth-dependent signal precision (tau_j = 2 W_j), deadweight
  participation cost phi·W·sqrt(y) and 1% stake fee, futarchy oracle
  (z = x + N(0, 0.1), cost C = 50) triggered when the market margin < 10%.
- Mechanism: built-in `binary_staking_market`, 3 acting rounds,
  winner_subsidy = 10 (without a subsidy the market is strictly zero-sum minus
  deadweight costs and rational agents never trade — verified empirically:
  zero participation across 150 proposals).
- 150 proposals, seed 777. A new engine option (`deterministic_env` +
  `proposal_offset`) gives every arm and every shard *identical* draws of
  (x, y, wealths, signals), so arms differ only in the decision process.
- LLM calls go through the local `claude` CLI, thinking budget capped at 1024
  tokens (uncapped thinking tripled latency without changing behavior in
  spot checks). Strict-JSON prompts; every call logged to JSONL
  (`raw_llm_logs/`). Arm B round-0 prompts contain *no* market state, so the
  logged first-round beliefs double as a clean poll (Arm C) of the same models
  on the same information.

Agent wealths drawn for this population: {1.87, 3.08, 5.63, 22.26, 51.87},
i.e. signal noise std from 0.52 (poorest) down to 0.098 (richest).

## Headline table

| Arm | Decision process | Acc. vs 1(x>0) | Value Σxy·D | Regret (of 368.24) | Mech. profit | LLM calls | Wall time |
|---|---|---|---|---|---|---|---|
| A | rational Bayesian agents + market | 0.947 | 367.99 | 0.25 | −1600 | 0 | ~4 s |
| B | **LLM (Haiku 4.5) agents + market** | **0.953** | **368.01** | **0.23** | −1550 | 2250 | ~101 min |
| C | poll: mean of B's pre-market beliefs > 0.5 | 0.953 | 367.95 | 0.29 | — | 0 extra | — |
| D | LLM manager, sees all 5 signals | 0.953 | 368.01 | 0.23 | — | 150 | ~19 min |
| E1 | random 50/50 | 0.613 | 191.50 | 176.74 | — | 0 | — |
| E2 | always approve | 0.500 | 228.15 | 140.09 | — | 0 | — |
| E3 | always reject | 0.500 | 0.00 | 368.24 | — | 0 | — |
| E4 | best-informed agent's raw signal | 0.953 | 368.01 | 0.23 | — | 0 | — |

Oracle-optimal Value* = 368.24. n = 150; the standard error on an accuracy of
0.95 is ±1.8 pp, so arms A–D and E4 are statistically indistinguishable.

## Sonnet subsample (first 30 proposals, matched environment)

| Arm (n = 30) | Acc. | Regret (of 126.5) | Notes |
|---|---|---|---|
| A rational market | 0.933 | 0.064 | |
| B Haiku market | 0.933 | 0.064 | mistakes: proposals 9, 27 (x = −0.03, −0.02) |
| B **Sonnet market** | 0.933 | 0.061 | *identical* mistakes: proposals 9, 27 |
| C Haiku poll | 0.967 | 0.031 | |
| C Sonnet poll | 0.967 | 0.026 | |
| E4 best signal | 0.933 | 0.061 | |

Galanis's "smarter agents aggregate better" does **not** show up in decisions
here — the ceiling binds both tiers, and Sonnet's market made exactly the
same two near-zero-|x| mistakes as Haiku's. Where the model tier *does* show
up is in trading quality: Sonnet stake-sizing correlations are similar
(stake vs precision +0.79, vs |signal error| −0.50), it also shows no
within-market belief updating, but its per-wealth utility profile on these
30 proposals is {+1.2, +2.7, +1.1, −1.6, +3.3} (poorest→richest) versus
Haiku's {−2.5, +3.1, +1.8, +1.1, +3.1} on the same 30 — the poorest,
noisiest Sonnet agent no longer loses, consistent with more conservative
low-conviction sizing. Smarter models lose less money, not make better
decisions (on 30 proposals this is suggestive, not significant).
Sonnet calls were ~4x slower (mean 90 s vs 21 s per call at the same
1024-token thinking cap); 450 calls, 0 parse failures, 2 transport errors
(retried into abstain).

## Key findings

**1. The market added nothing over a poll — because the environment
saturated.** Arms B (market), C (poll), D (manager) and E4 (single best
signal) all decide identically on 149/150 proposals and all sit at the
accuracy ceiling set by the richest agent's nearly-noiseless signal (std
0.098). Every mistake any arm made was on a proposal with |x| < 0.06, i.e.
essentially worthless decisions: importance-weighted regret is ≤ 0.08% of
Value* in all four arms. The null "market ≈ poll" is real but is a statement
about this environment's difficulty, not yet about mechanisms: to
discriminate aggregation methods the environment needs poorer individual
signals (lower phi/alpha), fewer wealthy outliers, or adversarial/correlated
noise. This mirrors Galanis's prediction-market setting, where differences
only emerge when no single trader is sufficient.

**2. Haiku agents are strikingly calibrated and act consistently with their
stated beliefs.** Round-0 belief calibration (n = 750 agent-proposals): stated
0.012 → realized 0.013; stated 0.191 → 0.121; stated 0.395 → 0.373; stated
0.600 → 0.627; stated 0.820 → 0.764; stated 0.986 → 0.956. Brier = 0.076.
Across all 2250 calls there was not a single case of staking YES with a
stated belief < 0.5 or NO with belief > 0.5, and zero JSON parse failures.

**3. Stakes scale with informedness — the "scale up bids" hypothesis
holds.** Spearman correlation of accepted stake with the agent's signal
precision is +0.86 (with per-agent total stake vs wealth: +0.97), and with the
agent's realized |signal error| it is −0.46. Belief extremity also predicts
stake size (+0.58). LLM agents do behave like informedness-weighted bettors,
which is the core mechanism decision markets rely on.

**4. But there is no within-market learning.** Mean belief drift between an
agent's first and last round is 0.027 (on a 0–1 scale), and drifts are as
likely to move away from the market majority as toward it (241 vs 217). Final
market price predicts x slightly *worse* (Pearson 0.837) than the pre-market
mean belief (0.890), because price weights stakes, and stake sizes add noise.
First-round, last-round, and price sign-accuracy are all identical (0.953).
The LLM agents effectively ignore the public history they are shown — the
sequential-rounds machinery bought nothing here.

**5. The LLM market redistributes from poor/noisy to rich/precise agents;
the rational market doesn't.** Total excess log-wealth by wealth level
(150 proposals): rational Arm A = {+11.3, +13.3, +13.4, +18.2, +10.5} — all
five profit from the subsidy. LLM Arm B = {−12.7, +0.8, +9.9, +11.1, +14.5}
— the poorest agent (noisiest signal) loses heavily. Haiku agents stake a
roughly constant ~21–23% of wealth (p50 0.214) regardless of edge, while the
rational agents' Kelly-style sizing makes participation individually
rational. Aggregate agent utility: 23.6 (LLM) vs 66.7 (rational). In a real
deployment this is an adverse-selection tax on unsophisticated LLM
participants — and, notably, the *decisions* stayed optimal anyway.

**6. The costly oracle almost never fires, and the LLM market is not
cheaper than asking one model.** Margins are wide (agents pile onto the
consensus side), so the 10%-margin oracle triggered on 1/150 proposals in
Arm B (2/150 in Arm A). Cost per decision in Arm B: 15 LLM calls, ~40 s
wall-clock (8-way sharded), roughly $0.06 (estimated at Haiku pricing with
the 1024-token thinking budget) — vs 1 call for the equally-accurate Arm D
manager and 5 calls for the Arm C poll. Mechanism subsidy cost dominates:
−1550 for the market arms vs 0 for poll/manager.

## Limitations (honest)

- **Ceiling effect.** The default MODEL.md parameters make the richest
  agent's signal nearly decisive (precision 104). All aggregation questions
  are compressed into ~7 coin-flip proposals near x ≈ 0. This is the main
  reason for the null result and must be fixed before drawing mechanism
  conclusions.
- **Scale.** 150 proposals, one wealth draw (one "population"), one seed.
  The per-agent utility findings rest on 5 wealth levels × 8 shard replicas.
- **Model tier.** Haiku 4.5 with a 1024-token thinking cap; the cap was
  chosen for latency, and uncapped thinking was only spot-checked, not
  evaluated at scale.
- **Prompt sensitivity.** One prompt per role, no ablations. The honest,
  quantitative prompt (exact noise stds, payout formula, Kelly hint) plausibly
  drives both the good calibration and the belief-consistent staking; a vaguer
  prompt could look very different.
- **No true price discovery.** The pari-mutuel market publishes pool sizes,
  not a tradable price, and 3 rounds is a short horizon; richer mechanisms
  (LMSR, order book) could reward within-market learning more.
- **Subsidy choice.** winner_subsidy = 10 (repo example value) was required
  for any rational-agent trade; results on mechanism profit scale with it.
- **Poll uses market-facing agents.** Arm C reuses Arm B round-0 beliefs
  (by design, zero extra calls); a standalone poll prompt without market
  framing might answer slightly differently.

## Suggested next experiments (ranked)

1. **De-saturate the environment and rerun B vs C vs D.** Set alpha so the
   *best* agent's noise std is ~0.5–1.0 (e.g. tau_j = 0.05·W_j), or cap
   wealth dispersion (sigma_W ≈ 0.5), and use 300+ proposals concentrated
   near |x| < 1. This is where market-vs-poll differences can actually
   appear, and it directly tests whether stake-weighting (finding 3) beats
   equal-weight averaging when no single agent dominates.
2. **Correlated / adversarial information.** Give agents partially correlated
   signals or plant one systematically biased (or strategic) agent. Polls
   average bias in; markets in theory let informed agents bet against it.
   This is the cleanest qualitative separation between Arm B and Arm C, and
   the decision-market analog of Galanis's manipulation questions.
3. **Make the oracle margin bind.** Raise the oracle trigger threshold (or
   price the oracle dynamically) so the mechanism actually faces the
   verify-vs-trust tradeoff, and measure whether LLM markets buy verification
   when and only when the value of information exceeds C. Today the oracle
   fired once in 150 proposals, so the "costly verification" half of the
   design was untested.

## Reproduction

```bash
python experiments/llm-decision-market/scripts/make_scenarios.py
python -m proposal_poker.simulate --scenario experiments/llm-decision-market/scenarios/arm_a.json \
  --output results/llm-decision-market/arm_a_report.json
experiments/llm-decision-market/scripts/run_arm_b.sh b        # 8 shards, ~100 min
python experiments/llm-decision-market/scripts/run_arm_d.py \
  --env-report results/llm-decision-market/arm_a_report.json \
  --output results/llm-decision-market/arm_d_decisions.json \
  --log experiments/llm-decision-market/logs/arm_d_calls.jsonl
experiments/llm-decision-market/scripts/run_arm_b.sh sonnet   # subsample
python experiments/llm-decision-market/scripts/analyze.py
```

Total LLM usage across the experiment: 2,870 CLI invocations (2,250 Arm B
haiku + 150 Arm D + 450 Sonnet + 20 smoke), zero JSON parse failures,
2 transport errors, well under the 5,000-call budget.

Artifacts in this directory: `metrics.json` (all numbers above),
`arm_a_report.json`, `arm_b_merged_report.json` (+ per-shard reports),
`arm_d_decisions.json`, `arm_b_sonnet_merged_report.json`, and
`raw_llm_logs/` (every prompt and raw response, 2850+ calls).

---

# Experiment v1 — de-saturated environment

v0's null (market = poll = manager = one good signal) was a ceiling effect:
the richest agent's signal was nearly noiseless, so every aggregation method
collapsed onto it. v1 re-runs everything in an environment where **no single
signal suffices but pooled signals do**: uniform wealth (sigma_W = 0,
W_j = 20.09 for all five agents) and precision ratio tau_j/W_j = 0.094, so
every agent's signal has noise std 0.727. On the realized seed-777 draws the
best-single-signal dictator scores 0.793 and the full-information Bayes
posterior 0.887 — a 9.3 pp aggregation headroom that v0 lacked. Everything
else (150 proposals, seed, mechanism, subsidy 10, 10% oracle margin, prompts,
Haiku 4.5 with the 1024-token thinking cap) is identical to v0; the
`deterministic_env` machinery again gives every arm the same (x, y, signal)
draws, so arms differ only in the decision process.

A transient API outage hit 159 of the 2250 Arm B calls (concentrated on 19
proposals); those 19 proposals were rerun individually on identical env draws
and spliced in (`metrics_v1.json:patched_proposals`). After splicing: 2250
calls, 0 parse failures, 0 transport errors.

## v1 headline table

| Arm | Decision process | Acc. vs 1(x>0) | Value Σxy·D | Regret (of 368.24) | Value ratio | LLM calls |
|---|---|---|---|---|---|---|
| A | rational Bayesian agents + market | 0.873 | 359.21 | 9.03 | 0.975 | 0 |
| B | **LLM (Haiku 4.5) agents + market** | **0.900** | 359.22 | 9.02 | 0.976 | 2250 |
| C | poll: unweighted mean of B's round-0 beliefs | 0.887 | 360.95 | 7.29 | 0.980 | 0 extra |
| C' | poll: precision-weighted (tau_j) mean | 0.887 | 360.95 | 7.29 | 0.980 | 0 extra |
| D | LLM manager, sees all 5 signals | 0.893 | 355.35 | 12.89 | 0.965 | 150 |
| E4 | best-signal dictator | 0.793 | 321.16 | 47.08 | 0.872 | 0 |
| — | full-information Bayes posterior (ceiling on signals) | 0.887 | 360.75 | 7.49 | 0.980 | 0 |

n = 150, so the SE on an accuracy of 0.89 is ±2.6 pp. Pairwise exact McNemar
tests: B vs C p = 0.77, B vs D p = 1.0, B vs A p = 0.34, B vs Bayes p = 0.75 —
none of the aggregating arms are distinguishable from each other. The only
significant separation is that every aggregating arm beats the single-signal
dictator (B vs E4: 22 vs 6 discordant proposals, p = 0.004). C and C' are
identical *by construction* here: with uniform wealth all precisions are
equal, so precision-weighting is a no-op (in v0, where wealths differed, the
saturated environment made the comparison vacuous instead; a v2 with unequal
wealths *and* headroom is needed to make weighting bite).

## v1 key findings

**1. De-saturation worked, and aggregation is real.** The dictator-to-Bayes
headroom (0.793 → 0.887) exists and every aggregation method — market, poll,
manager — captures essentially all of it. This is the positive result v0
could not show: five noisy LLM traders reliably reconstruct ~100% of the
pooled-signal optimum (B even lands 2 proposals above the signals-only Bayes
benchmark, see finding 3).

**2. Trading still adds nothing over polling the same models.** B beats C by
1.3 pp (0.900 vs 0.887, 7-vs-5 discordant, p = 0.77) — indistinguishable, and
the poll achieves *lower* importance-weighted regret (7.29 vs 9.02). The
distributed-vs-centralized comparison is equally null (B vs D, p = 1.0). One
asymmetry deserves emphasis and is a feature of the design, not a bug: **the
poll is not incentive-compatible** — agents are simply asked and have no
reason to lie, and these five have no hidden agenda — **while the market is**
(stakes are at risk; misreporting costs money in expectation). The v1 result
is therefore "when agents are honest, a poll is as good as a market and much
cheaper (5 vs 15 calls/proposal, no subsidy, no redistribution)"; what a
market buys is robustness when honesty cannot be assumed. Arm F below tests
exactly that premise.

**3. The oracle half of the mechanism finally activated — and it is the
market's only edge.** With noisy signals, margins tighten: the 10%-margin
verification oracle fired on 14/150 proposals in Arm B (11/150 in Arm A) vs
1/150 in v0. B's 2-proposal edge over the signals-only Bayes ceiling comes
from these purchases of outside information (the oracle's z has noise std
0.316, more precise than any agent). But the oracle proposals are the
near-zero-|x| coin flips, so even the oracle only gets 8/14 right, and the
mechanism paid C = 50 each time: −700 of Arm B's −2200 net mechanism profit
is oracle cost. Buying verification on close calls is the correct qualitative
behavior; at these parameters it isn't worth the money.

**4. v0's within-market-learning null survives de-saturation, including the
degradation.** Round-0 mean belief predicts x with Pearson r = 0.925;
last-round mean belief r = 0.924 (no learning; mean |belief drift| 0.026);
the final stake-weighted market price r = 0.882 — the market's *price* is
again a slightly worse aggregator than the unweighted average of the same
agents' pre-market beliefs, because stake sizes inject noise (sign accuracy:
0.887 round-0, 0.900 last-round, 0.893 price — all within noise of each
other). With real headroom to learn from and 3 rounds of published pools,
Haiku agents still do not update toward the market: the v0 conclusion was not
an artifact of saturation.

**5. Calibration degrades gracefully with signal quality; staking stays
belief-consistent, with a rational-looking exception.** Round-0 Brier is
0.140 (vs 0.076 in v0 — expected, signals are noisier) and the calibration
curve stays monotone (stated 0.02 → realized 0.00; 0.29 → 0.27; 0.51 → 0.55;
0.88 → 0.84; 0.98 → 0.99; n = 750). 96% of stakes (1573/1636) are on the side
of the stated belief. The 63 exceptions are *not* confusion: all occur in
rounds 1–2, and 60/63 bet *against* the market majority — small longshot
stakes on the thin side of a lopsided pari-mutuel pool, where the payout
multiple can justify a bet at odds worse than your belief. v0's "zero
inconsistencies" becomes "inconsistencies only where pot odds arguably
justify them."

**6. LLM agents still overpay to participate.** Total agent utility 4.9 (LLM)
vs 47.3 (rational) — with uniform wealth there is no poor agent to exploit,
so the v0 redistribution finding becomes a uniform tax: Haiku agents stake a
median 15% of wealth on noisy edges where Kelly sizing would stake far less,
and the difference is eaten by fees, participation costs, and losses to the
oracle-flipped settlements. Decisions were unharmed; wallets were not.

## v1 reproduction

```bash
python experiments/llm-decision-market/scripts/make_scenarios_v1.py --bounties 2.0 40.0
python -m proposal_poker.simulate --scenario experiments/llm-decision-market/scenarios_v1/arm_a.json \
  --output results/llm-decision-market/v1_arm_a_report.json
# 8 shards, scenarios_v1/arm_b_shard*.json -> v1_arm_b_shard*_report.json (~2.6 h)
python experiments/llm-decision-market/scripts/run_arm_d.py \
  --env-report results/llm-decision-market/v1_arm_a_report.json \
  --output results/llm-decision-market/v1_arm_d_decisions.json \
  --log experiments/llm-decision-market/logs/v1_arm_d_calls.jsonl
python experiments/llm-decision-market/scripts/analyze_v1.py
```

Artifacts: `metrics_v1.json`, `v1_arm_a_report.json`,
`v1_arm_b_merged_report.json` (+ shards + `v1_patch_*` reruns),
`v1_arm_d_decisions.json`; raw call logs in
`experiments/llm-decision-market/logs/calls_v1b*`, `calls_p*`,
`v1_arm_d_calls.jsonl`. Mean call latency 20.5 s; ~15 calls and ~$0.06 per
market decision vs 1 call for the equally-accurate manager.
