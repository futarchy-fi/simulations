# STATE — limit-orders agent (BATCH-LMSR with limit prices)

Branch: `exp/limit-orders` in ~/simulations-limit. Merge to origin/main when tested (pre-authorized).
Venv: `mechanism-design/.venv-batch/bin/python` (python3.11, numpy scipy pytest, -e galanis-market -e batch-amm).
Never touch ~/simulations or other worktrees.

## Task (owner, 2026-07-04)
Extend BATCH-LMSR (mechanism-design/batch-amm) with limit prices; re-run headline comparisons
vs the market-order baselines (CRN). Orders become (direction, qty, limit); buy fills iff
uniform clearing price pi <= limit, sell iff pi >= limit; clearing = fixed point of
pi = avg-exec-price(eligible net flow), standard uniform-price treatment with marginal
pro-rata fills. Honest: limit = own posterior (tight) + loose variant (posterior +/- slack);
qty = competitive sizing as now. Manipulator: bounty-B, best-responds, extreme limits OK.
Price-only disclosure, no attribution, seeds fixed.
Measure: (1) manip cost per unit distortion — does the 2.4-4.2x batch discount close?
(2) honest execution quality — does the 2.2-4.1x per-trader advantage shrink?
(3) aggregation speed/accuracy — extra rounds R needed; accuracy at fixed R.
(4) jamming/denial-of-aggregation (BATCH.md §9) — does limit protection change it?
Deliverable: BATCH.md new section "12. Limit orders" + tables + plain verdicts on all 4
and on whether the two headline numbers survive/shrink/reverse. Commit, merge to origin/main, push.

## Design decisions
- New module src/batch_amm/clearing.py: vectorized fixed-point clearing. D(pi) = eligible net
  demand (step, nonincreasing); phi(X) = LMSR avg exec price (increasing); bisect g(pi)=phi(D(pi))-pi.
  Continuity fixed point -> full fills; jump at a limit value -> pi* = that limit EXACTLY,
  X* solves phi(X*)=pi*, marginal orders (|limit-pi*|<1e-9) filled pro-rata (fill light marginal
  side fully first if both sides marginal). Price-cap: X clipped to bounds, pro-rata alpha as base engine.
- engine.py: mech "batch_lmsr_limit"; Config.limit_slack (0=tight, np.inf==market-order, reproduces
  batch_lmsr BIT-EXACTLY — final computation uses identical arithmetic on x*f with f=1);
  Config.manip_scale (gamma, best-response grid over order-size multiplier); manip limit = +/-inf.
- Disclosure (price-only): Gaussian — engine converts executed net flow to t_total_obs =
  X_exec/(s*b) + n*logit(p), feeds existing reveal_batch_anon (self-consistent: in the mean-field
  model limits never bind). Galanis — new reveal_batch_anon_limit(p_open, x_obs, ...): exact
  consistency on executed flow; per omega predict honest orders+limits, clear, keep omegas with
  X_pred == X_obs (1e-9); empty mask -> no update + state["jams"] += 1 (jam instrumentation,
  also added to reveal_batch_anon).
- Manipulator hypothesis to test for Q4: honest tight limits PIN pi at an honest posterior
  (marginal pro-rata) -> manipulated outcome pools with honest outcomes -> strict-consistency
  update stays nonempty -> jam partially defeated.
- Baselines re-run in-script with same env instances (SEED=20260704, SEED+n per N) for CRN pairing.

## Plan
docs/superpowers/plans/2026-07-04-limit-orders.md (tasks 1-6).

## Progress
- [x] Oriented: BATCH.md, engine.py, envs.py, metrics.py, tests, run_sweeps.py, run_disclosure.py read
- [x] STATE.md + plan written
- [ ] venv created, 36 pre-existing tests green
- [ ] Task 1: clearing.py + tests (incl. bit-exact market equivalence, single-trader limit bound)
- [ ] Task 2: engine.py batch_lmsr_limit branch + tests
- [ ] Task 3: envs.py Galanis anon-limit updater + jam counter + tests
- [ ] Task 4: scripts/run_limits.py sweeps -> results/limits*.json (nohup, poll log)
- [ ] Task 5: BATCH.md §12 + Reproduction update
- [ ] Task 6: full test suite, commit, merge to origin/main, push
