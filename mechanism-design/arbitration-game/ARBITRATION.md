# The FAO escalation game: kill-criteria sims (hub-w5xy)

**Question.** Which of the load-bearing claims behind FutarchyArbitration's
bond-escalation game survive contact with rational play? Targets the
kill-criteria ledger in `hub/docs/fao-load-bearing-claims.md` (claims C1, C2,
C4, C5, C6). Mechanism semantics mirror `futarchy-fi/FAO` `main` @ 8cee5bc
(`src/FutarchyArbitration.sol`): flip-only bids, match-only NO, per-proposal
timer, standing side wins both bonds at TIMEOUT, graduation threshold
`baseX·2^queueLen`, single-slot evaluation, `tryGraduate` requires a standing
NO bond. Defaults: TIMEOUT = 72h, m = 1, baseX = 8, oracle accuracy α.

**Method.** Deterministic seeded simulation (numpy), rational risk-neutral
marginal bidders per the 2026-02-05 Hanson-draft thresholds (marginal YES +EV
iff p > 2/3, marginal NO +EV iff p < 1/2), a two-belief crowd
(optimists/skeptics) for contested dynamics, and adversary/defender/watcher
agents per experiment. 9 unit tests pin the mechanism to the contract
(doubling, match-only, timer-resets-only-on-flips, graduation reachability,
conservation incl. the empty-NO burn sink, ±50%/100% contested ROIs).
`PYTHONPATH=/home/kelvin/.local/lib/python3.9/site-packages python3 <s*.py>`;
raw results in `results/*.json`.

## Analytic results (no sim needed, contract-verified)

1. **The (1/2, 2/3) "stall band" is an accept band.** Activation is YES-first
   and INACTIVE→NO is not allowed; in the band no marginal NO is +EV, so an
   activated proposal stands in YES and settles ACCEPTED by timeout at cost m.
   S1A confirms: accepted iff p ≥ 1/2, rejected iff p < 1/2, **zero
   graduations** under any common belief. The 2/3 threshold only ever bites
   once a skeptic exists.
2. **Uncontested proposals cannot graduate at all.** `tryGraduate` reverts on
   `noBond == 0` and flip-graduation implies a standing NO. The escape is
   **self-challenge**: the proposer bonds NO against themselves from a second
   wallet, then grad-jumps — owning both bonds makes the verdict
   payout-neutral, so "must survive escalation" degenerates for uncontested
   proposals into a carrying cost (capital lock + evaluation-market costs),
   a fee, not a filter.
3. **Timer-reset griefing is designed out** (kill-criterion 3): non-flip bids
   don't exist in the implemented game, and the global safety-mode veto
   (`totalActiveNoBonds ≥ baseX` blocking all YES-by-timeout org-wide) is gone
   from canonical main. Every delay costs an at-risk position.

## S1 — stall-band welfare and purchasable delay (C1, C2)

Common-belief sweep (M = 2000/point): the accept-band map above, with welfare
loss concentrated where miscalibrated acceptance executes bad proposals —
the game itself adds ≈ 1 timeout of latency and nothing else. Disagreement
grid (p_s ∈ {0.1..0.4} × p_o ∈ {0.7..0.9} × budget ratios): once both sides
are past their EV thresholds, **budgets, not beliefs, decide** — the poorer
side exhausts and the richer side wins by timeout, or the YES side reaches
baseX and graduates. Escalation depth 6–7 flips; graduation occurs iff the
optimist side can afford the threshold.

**Delay (S1C):** an adversary buying time against a p-good proposal:

| finding | value |
|---|---|
| cost per bought window at budget B=1 | ≈ Y·(2p−1): measured 0 / 0.066 / 0.215 / 0.414 / 0.793 at p = 0.5/0.55/0.6/0.7/0.9 (analytic 0 / 0.1 / 0.2 / 0.4 / 0.8) |
| max purchasable delay, ANY budget up to 256m | **≈ 2–2.8 timeout windows** — the log2(B) scaling is cut off by the graduation threshold: the flip war reaches baseX after ~3 doublings and the proposal graduates into evaluation |
| residual outcome risk while delaying at p≈0.5 | ~50% of runs end flipped (standing-side lottery in the band) |

**Verdict (C2): delay is structurally capped, not budget-limited.** A rich
adversary buys ~3 windows (~9 days at 72h) plus the evaluation duration —
compatible with the vision's stated tolerance ("delay but not permanently
block"). Near p = 1/2 the *first* window is nearly free, as predicted; it is
also the only cheap one.

**Verdict (C1): the welfare risk is not stalling — it's cheap acceptance.**
Everything the crowd rates > 1/2 executes by default at cost m unless
challenged. Agenda control therefore rests on challengers (C5) and tier caps,
not on the band's conservatism. The graduation-starvation half of the
kill-criterion is real but has a lawful bypass (self-challenge fee).

## S2 — two-tier endogeneity (C6)

Crowd sees a σ = 0.8 signal, optimist/skeptic band ±0.25 around the posterior;
disagreement-band proposals graduate to an α-oracle; the rest settle by
timeout on crowd information. Compared against the same α-oracle deciding
everything directly (M = 2000/cell, α ∈ [0.5, 1], q ∈ [0.1, 0.9]):

| α | mean Δ wrong-execution (system − oracle) |
|---|---|
| 0.50 | **−0.277** (system much better) |
| 0.70 | −0.103 |
| 0.85 | −0.017 |
| 0.88 | ≈ 0 (crossover ≈ σ) |
| 1.00 | **+0.102** (system worse) |

**Verdict (C6): the feared corruption-amplifying loop does not materialize.**
When the oracle is corrupt (α < σ), rational bidders *refuse to graduate*
(P_grad → 0) and the crowd's own information substitutes — the bond stage
insulates rather than inherits. The symmetric cost: when the oracle is
*good* (α > σ), the filter withholds mid-confidence proposals from it and
timeout-settles them at crowd accuracy, underperforming by up to ~0.15–0.20
at mid-q. The two-tier design is a hedge against bad oracles paid for by
taxing good ones; its accuracy floor is the crowd's information quality on
never-graduated proposals. **Untested residual (telemetry item):**
belief-reflexivity — p itself being bent by a manipulated oracle's track
record. Here crowd beliefs are exogenous to oracle corruption.

## S3 — capital robustness under mixed-quality spam (C4)

Spam floods λ ∈ {1, 4, 16}/TIMEOUT (plus a 160 stress row), bad fraction
β ∈ {1, 0.8, 0.5}, defender pools D0 ∈ {40, 80, 160}m, horizon 100·TIMEOUT,
congestion guard on/off:

- **Zero bad executions on the entire spec grid** (λ ≤ 16). Defender capital
  grows 1.00–1.04×/cycle — the Hanson-draft §6 "doubles per cycle" holds only
  at full deployment, but it isn't needed: defenders are never constrained.
- Overrun requires the 10× stress rate AND the smallest pool (λ = 160,
  D0 = 40, β = 1): 60 bad executions in 16k arrivals, at attacker cost
  **≈ 267m per bad execution**. D0 ≥ 80 restores zero even there.
- Mixed spam (β < 1) under these assumptions only thins the attack: good
  decoys execute harmlessly, defenders ignore them.
- Guard on/off is identical throughout this grid (graduation and
  timeout-accept never overlap here), so the implementation's dropped
  congestion guard is immaterial *in this regime* — it matters, if ever, in
  graduation-heavy congestion not reachable under rational spam.

**Verdict (C4): capital is the wrong bottleneck.** With the min-bond m sized
so that ~267m exceeds the tier-1 cap, spam-to-execute is strictly
unprofitable, and defense compounds mildly instead of bleeding. The honest
caveat: defenders here identify bad proposals perfectly. The real-world
failure route is **classification error and attention, not capital** — which
is exactly claim C5.

## S4 — challenger liveness on the unchallenged-YES route (C5)

W watchers, per-inspection cost c, spam λ/TIMEOUT with 5% bad, min bond m;
watchers inspect at break-even rates (watch iff +EV):

- Sharp threshold: **m\* = c/β** (measured m\* = 20c at β = 0.05,
  λ-independent). At m ≥ m\*, watching is self-financing, and even ONE
  watcher drives P(bad slips through) ≈ 0. Below m\*, no rational watcher
  watches and **every** bad proposal ships (P(slip) = 1) — inattention is
  indistinguishable from agreement, exactly the substitution the wild-dream
  map forbids.
- The economics are friendly to AI watchers: with agentic judging at
  c ≈ $0.01–0.1/proposal and β ≈ 1–5%, m\* is $0.2–$10 — trivially below any
  sensible min bond. A ≥1%-stake tokenholder additionally internalizes
  tier-cap losses, making watching +EV for them even without the bond reward.

**Verdict (C5): safe iff the bond floor respects m ≥ c/β and at least one
economically-motivated watcher exists.** This is a *deployment parameter and
an ops commitment*, not an open design question — but it is the single
claim where a live FAO fails silently if nobody wires the watcher. Launch
telemetry must publish time-to-first-challenge distributions and run at least
one house watcher with provenance labeling.

## Honest limitations

Rational risk-neutral marginal bidders with common/two-point beliefs — no
behavioral traders, no gas, no belief-reflexivity, no collusion between
spammer and defenders, perfect defender/watcher classification, evaluation
modeled as a black-box α-oracle (market microstructure lives in
MANIPULATION.md/BATCH.md, not here). Time is abstract (responses land 1h
after a state change). These sims falsify/confirm *mechanism-shape* claims;
they are model evidence (A-scale), not world evidence (E).

## Reproduce

```bash
cd ~/repos/simulations/mechanism-design/arbitration-game
PYTHONPATH=/home/kelvin/.local/lib/python3.9/site-packages python3 -m pytest tests/ -q
for s in s1_stall_delay s2_endogeneity s3_spam_capital s4_watchers; do
  PYTHONPATH=/home/kelvin/.local/lib/python3.9/site-packages python3 $s.py
done
```
