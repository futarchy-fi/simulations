# S5 — subsidy sizing for the proposal poker (claim C8, hub-w5xy/hub-vw1g)

**Question.** MODEL.md's key negative result is a no-information equilibrium:
without subsidy, everyone abstains and the default wins. Before real money
enters the poker, how large must the subsidy dial be? (Fable's 2026-07-10
decision-agenda item 6 made this a hard gate on real money; run 2026-07-16.)

**Method.** `s5_subsidy.py` imports the untouched `sim.py` engine
(wealth-lognormal agents, wealth-proportional signal precision, stake
escalation, 0.5× winner payout, futarchy fallback, default-reject) and adds
exactly three things, documented in its docstring: (1) an externally funded
per-proposal reward pool paid at settlement pro-rata to correct-side stakers;
(2) an entry rule adding the anticipated pool share (sole-deviator belief) to
the participation EV; (3) participation costs actually charged at settlement
(the engine only used them in the EV rule, which would understate the subsidy
needed for realized +EV). Grid: subsidy ∈ {0, 0.01, 0.03, 0.1, 0.2, 0.3, 1,
3, 10, 30, 50, 70, 100} (model wealth units; mean agent wealth ≈ 20) × four
cost regimes × 30 reps × 500 proposals. Baseline: always-default = reject-all
= efficiency 0. Raw: `results-s5-subsidy.json`.

## Results (participation / efficiency vs oracle optimum)

| regime (cost_per_tau, fee) | sub=0 | sub=0.3 | sub=3 | sub=30 | sub=100 | knee A: informed +EV | knee B: beats default |
|---|---|---|---|---|---|---|---|
| cheap (0.02, 0.5%) | 0.74 / **0.978** | 0.77 / 0.978 | 0.78 / 0.978 | 0.87 / 0.977 | 0.95 / 0.978 | **0** | 0 |
| default (0.1, 1%) | 0.53 / **0.978** | 0.62 / 0.975 | 0.67 / 0.978 | 0.82 / 0.978 | 0.94 / 0.978 | **0** | 0 |
| pricey (1.0, 5%) | 0.02 / 0.40 | 0.04 / 0.56 | 0.12 / 0.84 | 0.47 / 0.93 | 0.95 / 0.942 | **50** | 0 |
| prohibitive (2.0, 5%) | 0.00 / **0.000** | 0.004 / 0.12 | 0.08 / 0.69 | 0.38 / 0.91 | 0.71 / 0.929 | **100** | 0.3 |

Informed-PnL trace (per-proposal, wealth units): default regime already
+1.9 at sub=70 and positive at sub=0; pricey crosses zero at sub≈50
(−0.02) and is +0.40 at 70; prohibitive is deeply negative through sub=70
(−0.7) and turns +0.70 only at sub=100.

## Verdict

1. **The no-information equilibrium is a cost-regime phenomenon, not a
   universal property.** At the model's own default costs, the poker
   aggregates at 97.8% efficiency with **zero subsidy** — the smoke test
   reproduces full abstention only at 20× default participation costs.
2. **When subsidy is needed, it is heavy.** In the pricey/prohibitive
   regimes, individual rationality for informed stakers needs a per-proposal
   pool of ~50–100 wealth units ≈ **2.5–5× mean agent wealth per proposal**
   — and even then efficiency saturates at ~0.93–0.94, below the cheap-regime
   0.978. High participation costs both demand subsidy and cap what it buys.
3. **Aggregate efficiency is much cheaper than individual rationality.**
   Tiny pools (0.3–3) already lift prohibitive-regime efficiency from 0 to
   0.12–0.69 by attracting the few lowest-cost agents — but those agents lose
   money doing it, so this regime decays as they learn. Knee A (realized +EV)
   is the durable gate, not knee B.
4. **The C8 gate, restated operationally:** before real money, measure the
   live cost regime — per-proposal agent cost (inference + gas + attention)
   relative to stake scale. If costs/stake land in the cheap/default band
   (likely for AI agents on cheap chains), the dial can start at zero and be
   raised from telemetry. If the live regime is pricey-or-worse, budget
   ~2.5–5× mean stake scale per proposal or redesign costs downward first.

## Honest caveats

One-shot proposals (no reputation/learning across proposals); the entry EV
uses an optimistic sole-deviator share of the pool (over-entry at mildly
negative realized PnL is visible in the traces — knees would shift slightly
right under equilibrium entry); no collusion or sybil splitting of the pool
(pro-rata by stake invites stake-splitting only if entry costs are per-agent
— true here, so sybils are cost-punished); subsidy is externally funded, not
budget-balanced — sizing the *funding source* (treasury tap vs sponsor) is a
design decision this sweep does not make; engine quirks inherited as-is
(0.5× winner payout, futarchy fallback on contested high-importance stalls).

Reproduce: `PYTHONPATH=/home/kelvin/.local/lib/python3.9/site-packages python3 s5_subsidy.py`
