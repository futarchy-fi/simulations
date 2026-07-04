# BATCH-LMSR Limit Orders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend BATCH-LMSR with limit prices (uniform-price call-auction clearing against the LMSR curve), re-run the four headline comparisons vs the market-order baselines under CRN, and write BATCH.md §12.

**Architecture:** A new pure clearing module (`clearing.py`) computes the fixed point pi = phi(D(pi)) where phi(X) is the LMSR average execution price of net flow X and D(pi) is the eligible net demand (buys fill iff pi <= limit, sells iff pi >= limit), with marginal pro-rata fills at jump points. `engine.py` gains a `batch_lmsr_limit` mechanism that maps targets -> (qty, limit) orders, applies fills, and does price-only belief disclosure from the *executed* flow. `envs.py` gains an exact Galanis price-only updater for the limit mechanism plus jam instrumentation.

**Tech Stack:** numpy/scipy vectorized over M Monte Carlo reps, pytest, existing batch-amm conventions.

## Global Constraints

- Price-only disclosure, no per-trader attribution anywhere (owner's standing pseudo-anonymity rule).
- Seeds fixed: SEED = 20260704, Gaussian env seed = SEED + n per N (matches run_sweeps.py) — CRN vs baselines.
- With all limits at +/-inf the limit engine must reproduce `batch_lmsr` EXACTLY (bit-exact asserts on seeded runs).
- A single trader with limit = posterior must never execute past their posterior.
- Match existing code style (module docstrings explaining mechanisms, vectorized numpy, dataclass Config).
- b = 0.1 Gaussian, b = 0.01 Galanis; M = 20000 full runs; competitive sizing is the headline arm.

---

## Mechanism math (shared reference for all tasks)

Posted price p, liquidity b. Order i: signed qty x_i (buy > 0), limit l_i in [0,1] or +/-inf.
- Eligible at pi: buys with l_i >= pi, sells with l_i <= pi. D(pi) = sum of eligible x_i — a
  nonincreasing step function of pi.
- AMM average-price schedule: phi(X) = [C(p_X) - C(p)] / X with logit p_X = logit p + X/b,
  C(p) = -b ln(1-p); phi(0) = p (continuous), phi strictly increasing.
- Clearing: unique crossing of g(pi) = phi(clip(D(pi))) - pi (strictly decreasing).
  - Continuity fixed point: pi* = phi(D(pi*)) with no order's limit at pi* -> all eligible fill in full.
  - Jump at limit value l*: phi(D(pi*-)) > pi* > phi(D(pi*+)) with pi* = l* EXACTLY. Solve
    phi(X*) = pi* for X* by bisection in X on [D+, D-]; marginal orders (|l_i - pi*| <= 1e-9)
    fill pro-rata: with B_m = sum marginal buys, S_m = sum marginal sells, need
    beta_b B_m + beta_s S_m = X* - D_strict; fill the light side fully first (try beta_s = 1,
    then beta_b = 1), i.e. volume-maximizing standard treatment.
  - X = 0 (|X| < 1e-14): pi = p, eligible orders "fill" at mid (net zero — crossing at mid).
  - Price cap: X clipped so logit p' stays in [-L, L], L = logit(1 - PRICE_EPS); if binding, all
    eligible fills scale pro-rata by alpha = X_cap / D (same as base engine's rationing).
- Cash: every filled unit pays pi; sum pi x_exec_i = C(p') - C(p) exactly (conservation as before).

Honest orders: qty x_i = s b (logit t_i - logit p) (s = 1/N competitive), limit l_i = t_i + slack
in price space, signed by direction (buy: min(t_i + slack, 1); sell: max(t_i - slack, 0));
slack = 0 tight, slack = inf reproduces market orders. Manipulator: qty gamma * s b (logit t*(B) -
logit p) with the existing closed-form t*, limit +/-inf (they want fills); gamma swept for
empirical best response.

Price-only disclosure with limits: the clearing price still pins down the EXECUTED net flow
X_exec (invertible AMM), but no longer the submitted aggregate T. Gaussian observers keep the
mean-field inversion applied to X_exec: t_total_obs = X_exec/(s b) + N logit p (self-consistent:
N identical average traders' limits never bind). Galanis observers do the exact consistency
update on X_exec: keep omegas whose predicted honest limit-order clearing yields X_pred = X_obs;
unexplainable X -> no update (jam), counted in state["jams"].

---

### Task 1: clearing.py — the limit-order uniform-price clearing core

**Files:**
- Create: `mechanism-design/batch-amm/src/batch_amm/clearing.py`
- Test: `mechanism-design/batch-amm/tests/test_limit_clearing.py`

**Interfaces:**
- Produces: `clear_limit_batch(p, x, limits, b) -> dict(fill=(N,M) fractions in [0,1], p1=(M,),
  pi=(M,), x_exec=(N,M))` — pure function, no env coupling; `avg_price(p, X, b) -> (M,)` (= phi).
  Imports only lmsr_np and PRICE_EPS (defined locally to avoid circular import? No — envs.py does
  not import clearing's PRICE_EPS; clearing imports PRICE_EPS from envs is WRONG (envs will import
  clearing for the Galanis updater). Define `PRICE_EPS` usage via parameter `eps` default 1e-4 and
  keep clearing dependent only on lmsr_np.)

- [ ] Write failing tests (test_limit_clearing.py): infinite limits == plain netted move; single
  buyer tight limit never exceeded (pi <= l, full fill since phi(x) < t); dropped-order case
  (targets .9/inf-limit + .55/tight -> trader 2 fully out, pi = phi(x1)); marginal pro-rata case
  (one small buy l=inf + three buys l=0.6, phi(small) < 0.6 < phi(all) -> pi == 0.6 exactly,
  equal partial fills, phi(X_exec) == 0.6); offsetting tight orders cross at mid; cash-conservation
  property test over random targets/limits/slacks; price-cap rationing conserves cash; sell-side
  eligibility (pi >= limit).
- [ ] Run: `.venv-batch/bin/python -m pytest batch-amm/tests/test_limit_clearing.py -q` — FAIL (no module).
- [ ] Implement clearing.py per the math above: vectorized bisection on pi (~80 iters) over (M,)
  with (N,M) eligibility masks; jump detection via marginal-order tolerance 1e-9; X* bisection
  (~80 iters); final quantities recomputed with the exact base-engine arithmetic on x*fill.
- [ ] Tests pass; commit "batch-amm: limit-order uniform-price clearing core".

### Task 2: engine.py — batch_lmsr_limit mechanism

**Files:**
- Modify: `mechanism-design/batch-amm/src/batch_amm/engine.py`
- Test: extend `tests/test_limit_clearing.py` (engine-level)

**Interfaces:**
- Consumes: `clear_limit_batch` from Task 1.
- Produces: `Config(mech="batch_lmsr_limit", limit_slack=0.0, manip_scale=1.0, ...)`;
  `run_market` returns the usual dict + `volume_submitted (M,N)` and `fill_shortfall` via volumes.
  Limit runs require disclosure in ("price", "aggregate") — assert in `__post_init__`
  (attribution is meaningless when fills are partial and anonymous; price-only is the task rule).

- [ ] Failing engine tests: (a) seeded GaussianEnv runs, slack=inf, disclosure="price", R in {1,3},
  sizing in {competitive, full}: np.array_equal against mech="batch_lmsr" same disclosure on
  final_price/cash/holdings/mm_cash (BIT-EXACT); (b) single trader limit=posterior: pi <= posterior
  and final price == posterior-move (competitive N=1); (c) manipulator extreme limit fills fully at
  slack=0 for honest others; (d) cash conservation all rounds; (e) welfare identity
  sum pnl == b(ln2 - LL(p_final)) on a seeded run; (f) seat invariance with manip.
- [ ] Implement: new branch in run_market — targets, manip seat override (manip_target_lmsr,
  qty scaled by cfg.manip_scale, limit +/-inf by direction), honest limits t +/- slack clipped to
  (0,1) open interval... use min(t+slack, 1.0) with 1.0 meaning "never binds for pi < 1" —
  representable since pi <= 1 - eps < 1; but use np.inf for slack=inf to keep the bit-exact path
  trivially all-eligible. Call clear_limit_batch, apply cash/holdings/slip/volume with pi and
  x_exec; disclose from executed flow: Gaussian t_total_obs (above); Galanis
  env.reveal_batch_anon_limit(p_open, x_obs, state, b, scale, slack, first_time).
  Dispatch: `hasattr(env, "reveal_batch_anon_limit")`.
- [ ] Tests pass; commit "batch-amm: batch_lmsr_limit mechanism in engine".

### Task 3: envs.py — Galanis exact price-only updater under limits + jam counter

**Files:**
- Modify: `mechanism-design/batch-amm/src/batch_amm/envs.py`
- Test: extend `tests/test_limit_clearing.py`

**Interfaces:**
- Consumes: `clear_limit_batch` (envs may import clearing — clearing must NOT import envs).
- Produces: `GalanisEnv.reveal_batch_anon_limit(p_open, x_obs, state, b, scale, slack, first_time)`;
  `state["jams"]` counter in both GalanisEnv.make_state anon paths (reveal_batch_anon and
  reveal_batch_anon_limit increment on empty consistency mask).

- [ ] Failing tests: (a) honest R=2 limit run with slack=inf produces the same final beliefs/prices
  as market-order anon run (masks coincide: X <-> T bijection); (b) honest R=2 tight-limit run:
  decision acc 1.0 retained (bit-count classes should still separate — verify empirically, else
  document what coarsens); (c) jam counter: manipulated market-order anon run at B=0.2 R=3 has
  jams > 0; honest run has jams == 0.
- [ ] Implement per omega-consistency spec above (support omegas, predict 3 honest orders given
  round-open belief row + cells[w], clear scalar, match X within 1e-9).
- [ ] Tests pass; commit "batch-amm: Galanis price-only consistency update under limit orders".

### Task 4: scripts/run_limits.py — sweeps

**Files:**
- Create: `mechanism-design/batch-amm/scripts/run_limits.py`
- Output: `results/limits_core.json`, `results/limits_manip.json`, `results/limits_jam.json`

**Interfaces:**
- Consumes: Config/run_market, summarize/paired_diff, GaussianEnv/GalanisEnv; SEED = 20260704.

Sweeps (competitive sizing throughout; slack grid SLACKS = [0.0, 0.02, 0.05, inf]):
- [ ] core: env x N x R in {1,2,3,5} x slack; arms seq_lmsr (baseline), batch_lmsr
  (disclosure="price", baseline), batch_lmsr_limit per slack. Record summarize + paired_vs_seq +
  paired_vs_batch_market + fill rates. -> Q2, Q3.
- [ ] manip: gaussian N in {3,10}, galanis; R in {1,3}; bounties per env as run_sweeps; SEQ per-seat
  baseline rows; batch market-order rows; batch limit rows per slack in {0, 0.05, inf} with gamma
  best-response grid GAMMAS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5] (pick argmax mean manip utility =
  market PnL + bounty*(pf-0.5); record full grid + chosen). Seat invariance spot-check (0, n-1).
  -> Q1.
- [ ] jam: galanis, R=3, B grid, slack {0, inf}, price disclosure: acc/LL + jams count + honest-run
  jams. Also decision acc vs the §9 anon numbers (0.625/0.500). -> Q4.
- [ ] `--fast` flag (M=2000) smoke run first; then full M=20000 detached via nohup, poll log.
- [ ] Commit "batch-amm: limit-order sweeps + results".

### Task 5: BATCH.md §12

**Files:**
- Modify: `mechanism-design/batch-amm/BATCH.md` (new §12 before Reproduction; update Reproduction
  block: test count, run_limits.py line, results files; one pointer sentence in the top verdict).

- [ ] Tables: (1) manip cost/unit-distortion — SEQ seats vs batch market vs batch tight/loose,
  discount ratio; (2) execution shortfall + fill rate vs N — the 2.2-4.1x line re-derived;
  (3) LL/logit-err by R x slack — extra-rounds question; (4) jam table acc/jam-rate.
  Plain verdicts per question + explicit statement on the two headline numbers
  (survive / shrink / reverse).
- [ ] Commit "batch-amm: BATCH.md §12 limit orders".

### Task 6: verify + ship

- [ ] Full test suite green; verify results JSONs referenced in §12 match quoted numbers (spot-check).
- [ ] STATE.md final update; commit.
- [ ] Merge to origin/main: fetch, merge origin/main into branch (resolve), push branch, then
  merge branch into main and push (pre-authorized).
