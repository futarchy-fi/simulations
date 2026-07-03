# STATE — entry-sweep / TWAP / T2u agent (updated 2026-07-03 ~15:50 BST)

Branch: `manipulator-sweeps-v0` in this worktree (~/simulations-manip). Owner authorized free merging to main.
Venv: `mechanism-design/.venv-manip/bin/python` (open_spiel installed).
Task spec: see the phase-2 instructions — entry sweep (done), TWAP variants (running), T2u type uncertainty (implementing), MANIPULATION.md update + HTML explainer (pending).

## Done & committed (f16ede9, pushed)
- game.py/solve.py: num_players/signals/decision_rule support; entry_sweep.py script.
- Final-price entry sweeps solved: `mechanism-design/galanis-market/results/entry_sweep_{BASE-2,T1,T2,T3}.json`.
  Headline: T2-first at high bonus degrades exactly to BASE-2 (0.75, info exclusion floor);
  ANY bribed last-mover (even uninformed T1) drives acc → 0.5 < BASE-2 from bonus 0.02. BASE-3 = T2@bonus0 (acc 1.0).

## Running now (detached nohup, started ~15:45)
- PID 91859: `entry_sweep.py --only BASE-2,T1 --twap` → log /tmp/twap_t1.log → results/entry_sweep_BASE-2_T1_twap.json
- PID 91871: `entry_sweep.py --only T2 --twap` → /tmp/twap_t2.log → results/entry_sweep_T2_twap.json
- PID 91879: `entry_sweep.py --only REPL,T3 --twap` → /tmp/twap_t3.log → results/entry_sweep_REPL_T3_twap.json
  (REPL = phase-1 replacement config at bonuses 0/0.05/0.2, TWAP rule; entry_sweep.py has the new REPL block, uncommitted)
Each config ~3-5 min; full runs ~25-40 min. If dead, rerun same commands.

## Also running (started ~15:55)
- PID 58614: `t2u_sweep.py --positions first --qs 0.25,0.5 --iterations 300` → /tmp/t2u_first.log → results/t2u_first_q0.25_0.5.json
- PID 58615: `t2u_sweep.py --positions last --qs 0.25,0.5 --iterations 300` → /tmp/t2u_last.log → results/t2u_last_q0.25_0.5.json
  (bonus 0.2 saturation; type node implemented in game.py, sanity-tested: bonus only to bribed type,
  type private to entrant, prob=1 backward compatible, pytest green, walkers handle mid-game chance nodes)

## Next steps in order
3. When TWAP results land: comparison tables, extend mechanism-design/MANIPULATION.md with entry/TWAP/T2u sections.
4. mechanism-design/results-site/manipulation.html self-contained explainer (phase 1 + phase 2), tone of galanis-market/results/index.html.
5. Merge tested pieces to main, push.
