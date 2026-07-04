# Results

CFR+ equilibria of the four Galanis (2026) market structures, computed
on a tabular OpenSpiel formulation of the game.

## Outputs

| File pattern                              | What it contains                                |
|-------------------------------------------|-------------------------------------------------|
| `cfr_a{A}_i{I}_{struct}_r{R}.json`        | Raw run data: NashConv trace, per-omega prices  |
| `cfr_a{A}_i{I}_{struct}_r{R}.txt`         | Human-readable per-omega equilibrium table      |
| `run_a{A}_i{I}_r{R}.log`                  | Stdout/stderr of the solver run                 |

`{A}` = `--num-actions` (price-grid resolution); `{I}` = CFR+ iterations;
`{R}` = number of trading rounds.

## What "the equilibrium" means here

We solve the game with **tabular CFR+** in 3-player general-sum mode.
For >2-player general-sum games, CFR+ converges to a coarse correlated
equilibrium (CCE), not necessarily to a Nash equilibrium. We track
**NashConv** (sum of best-response improvements across players) as the
convergence signal -- low NashConv means each player's best-response gain
over the current average policy is small, but it does not certify Nash.

For the Galanis market specifically the gap between CCE and NE is small
(see paper / discussion notes): the LMSR market maker creates a
positive-EV unilateral deviation for any non-aggregating profile, which
rules out babbling-style CCEs. The CCE we converge to is in practice
indistinguishable from the focal myopic Bayes-Nash equilibrium for the
"easy" structures.

## Why we stop at 3 rounds (for now)

Tabular CFR's information-state count grows like
`cells_per_trader * num_actions ** (moves_before_player_acts_again)`.
At 9 rounds with 11 actions the very-hard structure has
~`4 * 11^8 ≈ 1.7e9` info states -- intractable in tabular Python.

To extend to 6 / 9 rounds we would need (a) coarse-action MCCFR, or
(b) move to OpenSpiel's C++ CFR with a C++ game definition. Both are
in scope for follow-up work but were excluded from this first pass to
keep the result reproducible on a laptop CPU.

## Reproducing

```bash
pip install -e .
python scripts/solve_all.py \
    --iterations 50 --num-actions 11 \
    --structures t3s111y2,t3s110,t3s111,t3s111o2ye2 \
    --rounds 3
```
