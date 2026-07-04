# STATE — kyle-batch agent (analytic + numerical corruption theory, Kyle batch decision markets)

(Previous STATE.md content — manipulator-sweeps + Arm G agents — was COMPLETE and merged; replaced 2026-07-04.)

Branch: `kyle-batch-v0` in this worktree (~/simulations-kyle). Owner authorized merging completed, tested work to main.
Venv: `mechanism-design/.venv-kyle/bin/python` (numpy, scipy, sympy, matplotlib, pytest — installed OK).
Never touch ~/simulations, ~/simulations-manip, ~/simulations-llm-dm, ~/simulations-batch.
Task: build mechanism-design/kyle-batch/ — src (SymPy closed forms + NumPy/SciPy numerics), tests, results/, KYLE.md.
Read mechanism-design/MANIPULATION.md first (CFR findings we must compare against).

## Task spec (condensed)
Model: v~N(0,1); N informed, s_i=v+ε_i, ε~N(0,σ_ε²); one batch of market orders; noise u~N(0,σ_u²);
Kyle MM p=E[v|y], y=Σx+u. Decision: approve w.p. q(p)=logistic(p/τ) (report τ sensitivity).
Settlement: outcome-settled, payoff x_i(v−p). Manipulator = trader 1, + B·1[approved].
Q1 baseline closed form (Holden–Subrahmanyam N-trader) + MC verify + aggregation/decision quality vs N,σ_ε,σ_u.
Q2 corruption fixed point (β_h, β_m, α_m, λ); price bias, quality delta, manip trading loss vs B;
   threshold-vs-smooth, blind-vs-biased; NOISE-BUDGET FRONTIER (σ_u = subsidy = dampener = camouflage) — headline plot.
Q3 entry: N honest vs +informed-manip vs +uninformed-manip; known (ρ=1) vs covert (ρ<1) MM; last-mover channel absent?; T2u "blurry" analog.
Q4 T batches, TWAP-of-batch-prices vs last-batch decision statistic, myopic-λ MM (state as such).
Q5 brief AMM (subsidized LMSR-ish curve) variant of Q2.
Validation: every equilibrium passes MC unilateral-deviation test on a strategy grid; report bound in KYLE.md per table.
Citations: only Kyle 1985 confidently; anything else [verify] unless fetched.
KYLE.md structure: model, baseline table, corruption+frontier plot, entry, TWAP verdict, comparison table vs MANIPULATION.md CFR, limitations.

## Key derivations so far (verify in code before trusting)
Baseline linear eq (symmetric): ρ_v=1/(1+σ_ε²); c≡λβ=ρ_v/(2+(N−1)ρ_v);
λ=sqrt(cN(1−c(N+σ_ε²)))/σ_u; β=c/λ. Kyle N=1,σ_ε=0 ⇒ λ=1/(2σ_u), β=σ_u ✓.
Var(p)=Cov(v,p)=cN ⇒ corr(p,v)=sqrt(cN), INDEPENDENT of σ_u (baseline quality σ_u-invariant).
Informed total profit = λσ_u² ∝ σ_u. Decision quality E[v·q(p)] = (Cov/Var)E[(p−m)q(p)] 1-d quadrature; oracle E[v1{v>0}]=1/√(2π).
Corruption (linear strategies): manipulator plays α_m+β_m s. MM (linear, knows presence w.p. ρ) subtracts ρα_m from y.
FOC ⇒ α_m=B·E[q'(p)]/(2−ρ). Price bias if present: λ(1−ρ)α_m; if absent: −λρα_m (suspicion tax) — T2u "blurry" analog.
ρ=1 (known): bias fully subtracted, β_m≈β_h at O(B) ⇒ NO info exclusion (contrast CFR!). Quality loss O(bias²) — smooth, no threshold.
Price bias ∝ λ·B·E[q'] ⇒ distortion INCREASES with λ (task's prediction B·q'/λ is inverted — check carefully numerically; frontier:
σ_u ↑ ⇒ resistance ↑ AND subsidy cost ↑, baseline quality flat).
Plan tiering: Tier1 = linear-strategy equilibrium via damped fixed point (quadrature for E[q], E[q']);
Tier2 = MC deviation test incl. manipulator's nonlinear pointwise best response (quantifies linear-restriction error).

## Progress
- [x] Worktree created from origin/main (9e158c5); MANIPULATION.md read.
- [x] venv installed
- [x] scaffold + pyproject; core modules: decision.py, closed_forms.py (sympy-verified),
      onebatch.py (affine Basis machinery, linear+Bayesian MM, T2u absent="honest", AMM mm="fixed"),
      mc.py (MC + grid + sup deviation tests), twap.py (exact affine multi-round, myopic MM, MC-verified).
- [x] 18 tests green (test_baseline, test_corruption, test_twap_amm).
- [x] sweeps done: results/{baseline,corruption,frontier,entry,twap,amm}.json + 4 figures (dataviz-palette PNGs)
- [x] KYLE.md complete (model, Q1-Q5, CFR comparison table, limitations, repro)
- [x] final: 18 tests green post-merge; merged origin/main (batch-amm) into branch, resolved STATE.md
      (stacked both agents); fast-forwarded main to b844252 and pushed. TASK COMPLETE 2026-07-04.

## Extra findings (beyond the list above, all in KYLE.md)
6. Frontier quantified: damage ∝ σ_u^{-1.8}, subsidy ∝ σ_u, baseline quality flat. 61x resistance per 10x subsidy at B=2.
7. Bayesian mixture MM ≈ linear MM (detection doesn't rescue thin markets; slightly WORSE DQ from curvature).
8. Entry: rho=1 informed entry pinned at BASE-3 for all B (floor strengthened vs CFR BASE-2);
   rho=0.25 crosses BASE-2 at B*≈7.3 (54x seat profit); rho=0.5 NEVER crosses (bias saturates ≈0.40
   via MM mixture-variance -> lambda collapse, self-limiting).
9. TWAP verdict: REVERSED vs CFR — last-batch dominates TWAP at all swept (T,B); T itself is the defense (30x damage cut T=1->8).
10. AMM: honest capture 81% of manip losses (Hanson transfer works vs responsive counterparties;
    covert Kyle MM only 7-20%); batch-netted curve >> sequential CFMM robustness (path dependence killed).

## Confirmed findings so far (all tested)
1. NEUTRALIZATION THEOREM (numeric): rho=1 known manipulator, linear MM, affine strategies ->
   bias exactly 0, b_m=b_h=baseline beta, lambda unchanged, DQ unchanged, manip pays nothing,
   approval stays 0.5 at ANY B. Subtraction != exclusion (contrast CFR info-deletion).
   Affine-restriction error (manip nonlinear sup gain): ~0.1% of position value at B=1, ~1% at B=5.
2. FOC identity: a_m(2-rho) = B E[q'(p)]; covert bias_present = lam(1-rho)a_m, absent = -lam rho a_m.
   Price bias INCREASES with lam (i.e. with 1/sigma_u) — task's predicted B q'/lam scaling inverted.
3. T2u analog exact: type uncertainty splits bias +lam(1-rho)(a_m-a_e) bribed / -lam rho(a_m-a_e) honest;
   symmetric at rho=.5; a_e>0 (honest type partially impersonates). "Blind->blurry" reproduced in closed form.
4. TWAP (T=4, N=3, covert uninformed pusher): baseline last-price DQ 0.284 > TWAP 0.271 (real baseline cost);
   corruption bias/DQ-delta similar for both rules at B=2 — TWAP does NOT dominate (reversal of sequential CFR);
   no last-mover channel in batch (MM prices every batch; pushes decay via honest price-anchored correction).
5. AMM (fixed kappa=lambda_Kyle): honest counter-trade a_h=-a_m/(N+1); honest profits RISE with B
   (Hanson transfer works); MM profits from push; DQ degrades smoothly/mildly.

## Notes
- Long runs: detached via `run_in_background`, poll (never park).
- Plots: read dataviz skill before writing plot code; commit PNGs into kyle-batch/results/.

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
