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

## Progress notes (~16:55)
- MANIPULATION.md phase-2a section committed (e783012).
- results-site/manipulation.html drafted: phase 1 + 2a complete with validated-palette SVG charts,
  hover tooltips, dark mode; TWAP chart reads window.__TWAP_DATA__ (to inject), T2u/TWAP tables + verdict
  paragraphs are empty placeholders (ids: twap-tbody, twap-verdict, t2u-tbody, t2u-verdict, TWAP-TABLE-SUB token).
- TWAP so far: REPL 0/0.05/0.2 -> 1.0/0.750/0.750 (vs final 1.0/0.796/0.750 -- threshold NOT raised);
  T2-first 0.02 -> 0.875 (WORSE than final 0.999: early pollution baked into average);
  T2-last 0.02 -> 0.9999, 0.05 -> 0.875 (vs final 0.500 -- last-mover attack killed);
  T1-last 0.02/0.05 -> 0.750 (floor held). T1-first/T3-first all 0.750 flat.
- T2u q=0.25: first acc_honest=1.0 acc_bribed=0.75, last acc_honest=1.0 acc_bribed=0.5 --
  exactly the known-type mixture; no hiding effect at q=0.25. q=0.5 solving.
- local preview server: python3 -m http.server 8931 in results-site (PID 22946, kill when done).

## Next steps in order
3. When TWAP results land: comparison tables, extend mechanism-design/MANIPULATION.md with entry/TWAP/T2u sections.
4. mechanism-design/results-site/manipulation.html self-contained explainer (phase 1 + phase 2), tone of galanis-market/results/index.html.
5. Merge tested pieces to main, push.
