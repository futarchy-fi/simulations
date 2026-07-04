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
- [ ] scaffold kyle-batch/{src,tests,results}, pyproject
- [ ] Q1 baseline module + sympy verify + MC + tests
- [ ] Q2 corruption fixed point + deviation test + sweeps + frontier plot
- [ ] Q3 entry sweeps
- [ ] Q4 TWAP multi-round
- [ ] Q5 AMM brief
- [ ] KYLE.md, commit/push/merge

## Notes
- Long runs: detached via `run_in_background`, poll (never park).
- Plots: read dataviz skill before writing plot code; commit PNGs into kyle-batch/results/.
