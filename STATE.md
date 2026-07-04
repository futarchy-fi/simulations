# STATE — twap-info-arrival agent (Q7: TWAP under information arrival)

Branch: `exp/twap-info-arrival` in ~/simulations-twap-info. Merge to origin/main when tested (pre-authorized).
Venv: `mechanism-design/.venv-kyle/bin/python` (create fresh in THIS worktree — worktrees don't share venvs).
Never touch ~/simulations or other worktrees.

## Task (owner hypothesis, 2026-07-04)
Q4b's K*=1 may be an artifact of ZERO info arrival during the window (static model: all signals at t=0,
push enters posterior and persists -> attacker re-times to window start). With info arriving during
trading (live stock price), a push should wash out; sustaining it costs per batch -> TWAP may recover value.
Extend kyle-batch with arrival, re-run windowed-TWAP comparison:
(a) staggered private signals (trader i's signal arrives at t_i); (b) PRIORITY: public stream z_t = v + eta_t
each batch, MM conditions on it. Sweep: arrival fraction phi in {0,.25,.5,.75}, B grid, K in 1..T, T in {4,8}.
Manipulator best-responds (exact open-loop optimum; also report re-time-to-window-start + sustained schedules).
Read rules: last, win:K, full TWAP, concealed uniform-K from Q4b. Metrics: damage/-dDQ vs manip cost,
baseline cost per K, push half-life vs phi, K*(phi) frontier. Deliverable: KYLE.md "Q7" section + verdict on
whether Q4b headline survives / is scoped to static info. Price-only disclosure (no attribution) throughout.

## Design decisions
- Parameterization: hold TOTAL end-of-window fundamental info fixed at reference (N=3, sig_eps=1 => private
  precision Pi=3). phi = fraction of Pi arriving via public stream during batches 2..T, spread evenly:
  private sig_eps^2 = N/((1-phi)Pi); each z_t (t=2..T) precision phi*Pi/(T-1). phi=0 == exact Q4b model.
- z_t released at START of batch t (before clearing), so p_t reflects z_t. Read rules stay price-only.
- Push linearity survives (alpha enters constants only) => reuse push-response D-matrix fast solver,
  stat_weights, concealed-mixture machinery from twap.py. Actual z contains NO push => z-innovation
  z_t - m_{t-1} is biased AGAINST the push (market overpredicts z) — this is the wash-out channel.
- Staggered variant (a): N=4, arrival times spread over 2..T for n_late = phi*N traders; per-round
  heterogeneous-beta myopic fixed point (linear system given lambda, damped iteration on lambda).

## Progress — COMPLETE 2026-07-04
- [x] Oriented: KYLE.md (esp Q4b), twap.py, decision.py, run_windowed.py, test_windowed.py read
- [x] venv installed; 31 pre-existing tests green
- [x] src/kyle_batch/arrival.py — BOTH variants: (b) public stream (obs order y1,z2,y2,...;
      Mrows/Grows raw-obs functionals) + (a) staggered (per-trader G rows, per-round
      heterogeneous-beta linear system + lambda fixed point); shared evaluate/push_response/
      mixture/named-schedule helpers reusing twap fast solver
- [x] tests/test_arrival.py green (11 tests: phi=0 == Q4b to 1e-12, all-at-1 staggered ==
      symmetric to 1e-9, precision identity, D-matrix linearity, MC x2, washout monotone,
      600-dir deviation checks, schedules dominated, concealed consistency); full suite 42/42
- [x] scripts/run_arrival.py -> results/arrival.json (both variants x T{4,8} x phi{0,.25,.5,.75}
      x K 1..T x B{0,.5,1,2,5,10,20} + schedules + concealed + buffer sub-sweep + 16 MC checks all 4se)
- [x] fig_arrival.png (make_plots.fig_arrival, dataviz palette validated); KYLE.md Q7 section +
      Q4b scope note + CFR row + repro updated

## Findings (Q7)
1. WASH-OUT PREMISE TRUE: push half-life falls with arrival rate phi in BOTH variants
   (T=8: 1.90->1.67 public, 2.02->1.62 staggered); staggered attacker switches from re-time
   to SUSTAINED schedule (the hypothesized mechanism operates).
2. BUT HYPOTHESIS CONCLUSION INVERTS for public-stream arrival (the realistic "live stock
   price" case): arrival HARDENS the last-batch read (damage at K=1,B=5 falls 7x at phi=.75;
   lam_T 0.054->0.024 starves the flow channel) and RAISES averaging's baseline cost
   (early prices miss not-yet-arrived info). K*=1 EVERYWHERE incl. the old (T=4,B=20)
   exception. Q4b headline STRENGTHENED, not weakened.
3. K*>1 emerges ONLY in staggered (private-through-flow) variant at extreme B: K*=2 at B=10,
   K*=3 at B=20 for phi>=.5 (T=8); late arrivals keep lam_T hot (0.054->0.144) => last price
   cheap to corrupt. Net gains material only at B=20~150x seat profit (+0.021).
4. Decisive quantity is NOT arrival rate but arrival CHANNEL (cost of end-window bias
   ~ b^2/lam_T): public feeds price AROUND flow (lam_T down), staggered THROUGH flow (lam_T up).
5. SETTLEMENT BUFFER > settlement window: moving arrivals 2 batches off the read batch
   ((1,1,3,6) vs (1,1,2,8)) cuts K=1 damage 2.4x, restores K*=1 at B=10, shrinks B=20
   window gain to +0.0015. Concealed-K still the robust extreme-bounty choice where last
   price is fragile; dominated by committed K=1 under public arrival.
- [x] committed 26231c1 on exp/twap-info-arrival; origin/main fast-forwarded 43cd720..26231c1
      and pushed. TASK COMPLETE 2026-07-04.

---

# STATE — kyle-batch EXTENSION agent (windowed TWAP + subsidy comparison)

Branch: `kyle-batch-v0` in ~/simulations-kyle (previously merged to main; merge again when tested — owner authorized).
Venv: `mechanism-design/.venv-kyle/bin/python`. Never touch other worktrees.

## Extension task (from Kelvin, 2026-07-04)
1. WINDOWED TWAP (priority): Q4 compared only K=1 (last) vs K=T (full TWAP) — confounds
   "averaging bad" with "remembering early rounds bad". Rerun multi-round sweep with statistic =
   mean of LAST K batch prices, K in {1,2,4,T}, T in {4,8,16}, same B grid. Report K*(T,B); decompose
   per K: baseline accuracy cost (B=0) vs manipulation-damage reduction (delta at B>0).
   CONCEALED-WINDOW variant: manip BR to known K vs uniform-random K in {1,2,4} drawn after trading.
   Extend KYLE.md Q4; supersede old verdict explicitly if it flips for late windows.
2. SUBSIDY COMPARISON: grid of budgets S; (a) noise flow with lambda*sigma_u^2 = S;
   (b) AMM depth (Q5 fixed-kappa machinery) with b sized so worst-case maker loss = S
   (LMSR bridge: p = (tau/b)*Q exactly via logit link => kappa = tau*ln2/S, worst-case loss b*ln2).
   Metrics at reference B=2: bias, -dDQ, baseline DQ. One resistance-per-dollar figure.
   Flag equilibrium-rescaling caveat: sigma_u-invariance needs beta scaling; LLM results
   (results/llm-decision-market/RESULTS.md finding v1-4: no within-market learning) => behavioral
   noise-subsidy may dampen aggregation — mark open experiment; quantify frozen-beta pessimistic bound.
Validation discipline unchanged: MC + unilateral-deviation tests with stated bounds per table.

## Extension progress — COMPLETE 2026-07-04
- [x] Oriented: read KYLE.md, twap.py, onebatch.py, closed_forms.py, decision.py, mc.py, run_sweeps.py, tests
- [x] twap.py: windowed statistic (win:K), push-response-matrix fast solver + gradient, concealed mixture
- [x] tests/test_windowed.py green (8 tests: endpoint reductions, D-matrix identities, solver x-check,
      1200 perturbation deviation checks, MC agreement, concealed-mixture consistency)
- [x] windowed sweep -> results/twap_windowed.json + fig_twap_windowed.png (B grid extended to 20)
- [x] subsidy sweep -> results/subsidy.json + fig_subsidy.png; tests/test_subsidy.py green (5 tests:
      noise sizing identity to 1e-8, sigma_u-invariance to 1e-9, exact LMSR logit bridge, kappa=lam* anchor)
- [x] KYLE.md: Q4b section (supersedes old Q4 reasoning, verdict sharpened), Q6 section, validation
      bounds updated, CFR table TWAP row extended, repro updated (31 tests)
- [x] full pytest 31/31 green; committed incrementally on kyle-batch-v0

## Extension findings
1. K*(T,B) = 1 EVERYWHERE for B <= 10 (T in {4,8,16}); sole exception (T=4, B=20 ~ 150x seat profit): K*=2.
   Anti-TWAP verdict does NOT flip for late windows — it sharpens. Kelvin's confound resolved:
   late-window averaging buys ~ZERO damage reduction (T=8/16: 0 to negative) because a push at the
   window's opening batch persists through the posterior into every window price (manip re-times:
   alpha=[.07,.26,.70,.04] at T=4 K=2). Old full-TWAP "damage reduction" was early-price dilution.
2. Concealed window (uniform K in {1,2,4} drawn after trading) recovers 30-40% of manipulation damage
   vs K-informed attacker, but K=1 committed still beats the concealed mixture for B<=5 (baseline cost
   of reading K in {2,4} sometimes > concealment gain). At B>=10 (T=8,16) / B>=20 (T=4) the concealed
   mixture beats EVERY deterministic window incl. K=1 — randomization is the extreme-bounty defence.
3. Subsidy comparison (covert, B_ref=2, S grid 0.1..3.2): noise damage 2.0-2.3x LOWER than AMM-depth
   at matched worst-case budget S>=0.8 (per-dollar hardening winner: noise). BUT AMM dollars also buy
   baseline aggregation (dq0 0.245->0.285; noise flat 0.2586 by sigma_u-invariance) and its expected
   spend < worst-case budget (0.26/0.91/2.02 at S=0.8/1.6/3.2; maker PROFITS for S<=0.4): absolute DQ
   under attack crosses — noise wins S<~0.5, AMM wins S>=0.8 (0.2852 vs 0.2585 at S=3.2).
   Aware honest traders cut AMM damage ~4x (Hanson counter-trade).
4. Behavioral caveat quantified: frozen-beta noise leg collapses (corr 0.77->0.19, dq0 0.259->0.023 at
   S=3.2) while frozen-beta AMM depth only miscalibrates (corr invariant under p=kappa*y rescaling) —
   behavioral robustness favors depth; LLM RESULTS.md v1-4 (no within-market adaptation) makes this the
   flagged open experiment (sigma_u x {1,4} stake-response test).
- [x] merged to main + pushed (see git log)

Key design notes (extension):
- Price means are LINEAR in push vector alpha (affine propagation, alpha enters constants only):
  bias = D @ alpha with D[t,r] = mean price_t under alpha=e_r; Var/cov(v,P) alpha-independent.
  Manip open-loop objective U(alpha) = -alpha'D alpha + B*E_q(w'D alpha, sd_P) — solve with BFGS +
  exact gradient (T=16 cheap); cross-check vs Nelder-Mead at T=4 and vs existing evaluate().
- Concealed window: single alpha vs uniform mixture over K in {1,2,4}: U = -alpha'D alpha +
  B*mean_K E_q(w_K'D alpha, sd_K); realized DQ = mean_K DQ_K(alpha). Compare vs known-K mixture.
- Subsidy noise leg: whole affine equilibrium scales with sigma_u (flows in sigma_u units), so
  lambda0*sigma_u^2 = lambda0(1)*sigma_u exactly => sigma_u = S/lambda0(1); verify numerically.
- Covert-covert is the apples-to-apples threat model (rho=0 both leg); AMM with aware honest
  traders (Q5 rho=1) reported as secondary variant.

---

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

# STATE — kyle-batch agent v0 (analytic + numerical corruption theory, Kyle batch decision markets) — COMPLETE

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
