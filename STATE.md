# STATE — batch-amm agent (batch vs sequential trading vs AMM)

Branch: `batch-amm-v0` in this worktree (~/simulations-batch). Owner authorized merging completed, tested work to main.
Venv: `mechanism-design/.venv-batch/bin/python` (py3.11, numpy/scipy/pytest/open_spiel; galanis-market installed -e --no-deps).
Task: quantitative comparison SEQ-LMSR vs BATCH-LMSR vs BATCH-KYLE, Gaussian + Galanis-Easy envs,
myopic-Bayes traders, N in {3,5,10,25} x R in {1,3}; price-improvement, aggregation, manipulation
(bounty sweeps, seat effects), ordering games. Deliverable: mechanism-design/batch-amm/ + BATCH.md.

## Done
- Core package `mechanism-design/batch-amm/` (src: lmsr_np, envs, engine, metrics).
  - Precise batch clearing rule (net flow into LMSR, uniform avg-price fills, pi=mid at X=0,
    pro-rata rationing at price cap) — documented in engine.py docstring.
  - Sizings: "full" (literal sequential order; single-trader==sequential exact) and
    "competitive" (1/N damped; stable across rounds; ==full at N=1). Both unit-tested.
  - Envs: GaussianEnv (vectorized over M reps, exact signal inversion) and GalanisEnv
    (t3s111y2, exact 8-state enumeration; SEQ R=1 reproduces galanis myopic.py exactly).
  - Manipulator: closed-form myopic LMSR target FOC (grid-verified argmax test); Kyle order form.
- 28 unit tests green (netting identities, single-trader equivalence, conservation,
  LMSR path-independence of MM revenue, seat-invariance of batch, myopic.py parity).

## Key analytical insight to carry into BATCH.md
LMSR MM revenue telescopes: aggregate "leak to the curve" depends ONLY on (p0, p_final),
not the path. So Kelvin's price-improvement claim lives in (i) per-trader execution/ordering
rents, (ii) different final prices across mechanisms, (iii) manipulation costs — NOT in
aggregate curve leak at equal final prices. Unit test test_seq_mm_revenue_is_path_independent
pins this.

## Next
1. scripts/run_sweeps.py: core Gaussian sweep (N x R x mech x sizing, M=20k CRN),
   Galanis core (exact), manip sweeps (seats x bounties), seat rotation sweep. -> results/*.json
2. BATCH.md with tables.
3. Stretch: one-shot simultaneous CFR+ (standalone, exact 8x9^3 enumeration) for BATCH-LMSR
   equilibrium check vs myopic; infoset-count scaling vs sequential game.
4. Merge to main when tests pass.

## Notes/decisions
- PRICE_EPS = 1e-4 price clip; extreme-tail reps saturate (info loss) — documented; exactness
  tests mask saturated reps.
- Gaussian belief updates only on first trade (round 1); Galanis every trade with
  no-update fallback for unexplainable (manipulated) trades — deviation from myopic.py's
  true-cell fallback, which would leak the manipulator's real signal.
- Bounty convention both envs: utility += bounty * (p_final - 0.5), Galanis-style.
