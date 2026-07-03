"""Entry sweep: does ADDING a manipulator-entrant to a market ever make
prices/decisions worse than that player (and their information) not
existing at all?

Baselines and treatments (Galanis Easy `t3s111y2`, LMSR b=0.01,
9-action grid):

* BASE-2   -- 2 informed traders observing bits b, c. Bit a exists in
              nature (X still depends on it) but nobody observes it.
              This is "the entrant and their information absent."
* T1       -- BASE-2 + an entrant with NO private info + price bonus.
* T2       -- BASE-2 + an entrant who observes bit a (unique info).
* T3       -- BASE-2 + an entrant who observes bit b (redundant with
              player 0's signal).

Each treatment runs with the entrant in seat 0 (moves FIRST; honest
traders can correct afterwards) and seat 2 (moves LAST; the decision
rule reads the final price, so nobody corrects). Bonus sweeps over
BONUSES; bonus=0 rows are honest-entrant baselines (T2@0 == a 3-trader
fully-informed market, i.e. BASE-3).

Usage:  python entry_sweep.py [--only T1,T2] [--twap]

--twap re-solves every config with the decision statistic (and the
manipulator's price-target bonus) computed on the time-averaged price
over the trade sequence instead of the raw final price.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from galanis_market.game import GalanisMarketGame  # noqa: E402
from galanis_market.solve import expected_profits, solve  # noqa: E402
from galanis_market.structures import STATES, STATE_LABELS  # noqa: E402


BONUSES = [0.0, 0.02, 0.05, 0.2]
ITERATIONS = 300
ACTIONS = 9
STRUCTURE = "t3s111y2"

ENTRANT_SIGNAL = {"T1": "none", "T2": "a", "T3": "b"}


def price_trajectory(game, policy):
    """Per-omega expected price at each timestep t=0..num_rounds."""
    n_steps = game.num_rounds + 1
    out = {}
    for omega_idx, label in enumerate(STATE_LABELS):
        sums = [0.0] * n_steps
        total = [0.0] * n_steps

        def walk(state, weight):
            hist = state.price_history
            for t, p in enumerate(hist):
                sums[t] += weight * p
                total[t] += weight
            if state.is_terminal():
                return
            for action, prob in policy.action_probabilities(state).items():
                if prob > 0.0:
                    walk(state.child(action), weight * prob)

        root = game.new_initial_state()
        root.apply_action(omega_idx)
        # Weight leaves only: recurse and accumulate at terminals.
        sums = [0.0] * n_steps

        def walk_leaf(state, weight):
            if state.is_terminal():
                hist = state.price_history
                for t, p in enumerate(hist):
                    sums[t] += weight * p
                return
            for action, prob in policy.action_probabilities(state).items():
                if prob > 0.0:
                    walk_leaf(state.child(action), weight * prob)

        walk_leaf(root, 1.0)
        out[label] = [round(s, 6) for s in sums]
    return out


def run_config(name, params, bonus=None):
    if bonus is not None:
        params = dict(params, manipulator_bonus=bonus)
    game = GalanisMarketGame(params)
    t0 = time.time()
    res = solve(game, iterations=ITERATIONS, log_every=None, verbose=False)
    elapsed = time.time() - t0
    nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
    profits = expected_profits(game, res.policy)
    agg = profits["__aggregate__"]
    traj = price_trajectory(game, res.policy)
    print(
        f"[{name}] bonus={bonus if bonus is not None else '-'} "
        f"done in {elapsed:.0f}s nc={nc:.2e} LE={res.mean_log_error:.4f} "
        f"acc={agg['decision_accuracy']:.4f}",
        flush=True,
    )
    return {
        "params": {k: v for k, v in params.items()},
        "elapsed": elapsed,
        "nash_conv": nc,
        "mean_log_error": res.mean_log_error,
        "aggregate": agg,
        "by_omega": {
            label: {
                "x": res.price_by_omega[label]["x"],
                "mean_price": res.price_by_omega[label]["mean_price"],
                "p_high": profits[label]["p_high"],
                "returns": profits[label]["returns"],
                "market_pnl": profits[label]["market_pnl"],
                "price_trajectory": traj[label],
            }
            for label in STATE_LABELS
        },
    }


def main() -> None:
    only = None
    twap = "--twap" in sys.argv
    rule = "twap" if twap else "final"
    for i, arg in enumerate(sys.argv):
        if arg == "--only":
            only = set(sys.argv[i + 1].split(","))

    t_start = time.time()
    results = {}

    def wanted(tag):
        return only is None or tag in only

    if wanted("BASE-2"):
        results["BASE-2"] = {"0": run_config("BASE-2", {
            "structure": STRUCTURE, "num_players": 2, "num_rounds": 2,
            "num_actions": ACTIONS, "signals": "b,c",
            "decision_rule": rule,
        })}

    for variant in ("T1", "T2", "T3"):
        if not wanted(variant):
            continue
        e = ENTRANT_SIGNAL[variant]
        for pos, signals, seat in (
            ("first", f"{e},b,c", 0),
            ("last", f"b,c,{e}", 2),
        ):
            cname = f"{variant}-{pos}"
            results[cname] = {}
            for bonus in BONUSES:
                params = {
                    "structure": STRUCTURE, "num_players": 3, "num_rounds": 3,
                    "num_actions": ACTIONS, "signals": signals,
                    "manipulator_player": seat, "manipulator_direction": 1,
                    "decision_rule": rule,
                }
                results[cname][str(bonus)] = run_config(cname, params, bonus)

    suffix = "_".join(sorted(only)) if only else "all"
    if twap:
        suffix += "_twap"
    out_path = _REPO / "results" / f"entry_sweep_{suffix}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "structure": STRUCTURE, "bonuses": BONUSES,
                              "decision_rule": rule},
                   "results": results}, f, indent=2)

    print()
    print(f"{'config':>10} {'bonus':>6} | {'LEfinal':>7} {'LEstat':>7} "
          f"{'dec_acc':>7} | {'entrantMkt':>10} {'informed':>9}")
    for cname, rows in results.items():
        for bonus, r in rows.items():
            agg = r["aggregate"]
            mk = agg["market_pnl"]
            stat_le = agg.get("stat_log_error", float("nan"))
            if len(mk) == 2:
                ent, inf = float("nan"), mk[0] + mk[1]
            else:
                seat = r["params"].get("manipulator_player", 2)
                ent = mk[seat]
                inf = sum(mk) - ent
            print(f"{cname:>10} {bonus:>6} | {r['mean_log_error']:7.4f} "
                  f"{stat_le:7.4f} {agg['decision_accuracy']:7.4f} | "
                  f"{ent:+10.5f} {inf:+9.5f}")
    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
