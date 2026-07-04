# STATE — batch-amm agent, follow-up: disclosure regimes / pseudo-anonymity: COMPLETE

Branch: `batch-amm-v0` (~/simulations-batch). Owner authorized merging tested work to main.
Venv: `mechanism-design/.venv-batch/bin/python`.
Constraint tested: NO per-trader order disclosure (traders see anonymous aggregates only).

## Status: done (2026-07-04)
- Config.disclosure in {full, aggregate, price}; aggregate==price PROVEN (deterministic AMM:
  clearing price pins net flow; bit-identical runs, unit-tested). Attribution is the only knob.
- Anonymous belief updaters: Gaussian mean-field inversion (N copies of s_bar, own-copy swap;
  exact for equal signals); Galanis exact Bayes on aggregate statistic T (bit-count classes).
- Sweeps: results/disclosure.json (N x R{2,3,5} x regime, CRN) + disclosure_manip.json
  (bounty x regime x seat, manipulator rotated with fixed draws). 36 tests green.
- BATCH.md: new section 9 "Disclosure regimes / pseudo-anonymity"; implications rewritten
  (anonymity mandates batch — sequential tape attributes by timing; recommended default:
  BATCH-LMSR damped R>=2 aggregate-disclosure + net caps + soft inversion).

## Verdicts
- Recovery: survives under (b)/(c). Gaussian: permanent but tiny plateau (logit err 0.11-0.18,
  LL +0.0004-0.0006 vs SEQ ~borderline CI, welfare cost ~0.1%); R=2/3/5 IDENTICAL (round-1-only
  info model) => anonymity costs zero extra rounds for ~99.9% of value, no R recovers the rest.
  Galanis: EXACTLY free (bit-count sufficient for symmetric payoff; R=2 anon == R=2 full).
- Manipulation delta: env-dependent sign. Gaussian: anonymity mildly DAMPENS (dPf -12%, cost +25%
  at B=0.5 N=3). Galanis: WORSE — jamming: manipulator's inconsistent order makes whole aggregate
  unexplainable -> strict-consistency update blocks ALL info -> R=3 floor 0.750 collapses to R=1
  damage (0.625 at B=0.02, 0.500 at B=0.2) at LOWER manipulator cost. Denial-of-aggregation attack.
- Seat-invariance holds under anonymity (0 violations at 1e-9, + unit test). Oscillation/damping
  finding unchanged (full sizing still diverges; damping still required).

## Caveat carried in BATCH.md
Myopic traders never discounted anyone even under FULL, so anonymity's lost-discounting cost
does not bind here (equilibrium-level bound: MANIPULATION.md phase 2c). Anon model mines round-1
aggregate only; plateau is an upper bound on the aggregation cost.

---

# STATE — batch-amm agent (batch vs sequential trading vs AMM): COMPLETE

Branch: `batch-amm-v0` in this worktree (~/simulations-batch). Owner authorized merging completed, tested work to main.
Venv: `mechanism-design/.venv-batch/bin/python` (py3.11, numpy/scipy/pytest/open_spiel; galanis-market -e --no-deps).
Deliverable: `mechanism-design/batch-amm/` (src, tests, scripts, results/, BATCH.md).

## Status: done (2026-07-04)
- Core engine: SEQ-LMSR / BATCH-LMSR (netted uniform-price clearing, precise rule in engine.py
  docstring) / BATCH-KYLE (linear matched-depth bridge). Sizings: full (single-trader==sequential,
  unit-tested) and competitive (1/N damped, stable; ==full at N=1).
- Envs: Gaussian (M=20k vectorized, CRN) and Galanis Easy t3s111y2 (exact 8-state enumeration,
  0.1/0.9 quote cap + b=0.01 for direct comparability with MANIPULATION.md CFR+ numbers; honest
  SEQ reproduces myopic.py exactly and matches CFR equilibrium totals: LL 0.105, acc 1.0, +0.0059).
- 28 unit tests green (netting identities, single-trader equivalence, conservation, LMSR
  path-independence, batch seat-invariance, myopic parity, manip-FOC argmax).
- Full sweeps done: results/core.json, manip.json, seats.json (+ sweep_full.log).
- Stretch DONE: scripts/cfr_batch.py — exact simultaneous-move CFR+ (RM+, linear avg) on one-shot
  batch game, NashConv <= 4.2e-8; results/cfr_batch.json. 6 infosets vs 182 sequential.
- BATCH.md written (mechanisms, welfare identity, all tables, limitations, implications).

## Headline results (details in batch-amm/BATCH.md)
- LMSR welfare identity: total trader PnL = b(ln2 − LogLoss(p_final)) — aggregate "leak to curve"
  depends ONLY on final price; batch cannot save aggregate welfare at matched information.
- Execution shortfall (honest, R=1): batch-competitive 2.2x (N=3) -> 4.1x (N=25) lower than SEQ;
  reverses at R=3 large N (+52% worse at N=25) — netting wins only under disagreement.
- Aggregation: one-shot batch strictly worse (myopic logit err ~log N; CFR equilibrium LL 0.171 vs
  seq 0.105 — structural, not behavioral); R>=2 batch-competitive == SEQ exactly.
- Ordering rents: SEQ seat 0 earns +0.019/market regardless of N (69x last seat at N=25);
  batch zeroes spread to machine zero. Galanis sign reverses (last seat best) — structure-dependent.
- Manipulation: SEQ seat gradient 1.9-5x (last-mover worst; Galanis last seat acc->0.5 at any
  bounty >=0.02, reproducing MANIPULATION.md 2a behaviorally). Batch seat-invariant (<=1e-10),
  worst case ~ sequential MEAN, but 2.4-4.2x CHEAPER per unit distortion (uniform-price fill
  subsidy; sometimes outright profitable under non-discounting traders). At CFR equilibrium batch
  bottoms at the 0.75 exclusion floor (no 0.5 collapse). BATCH-KYLE + repeated rounds = worst arm
  (static linear maker bleeds; full-sizing iteration divergent for both makers).

## Remaining
- Merge batch-amm-v0 -> main, push (in progress).
