"""Sweep manipulator bonus values on Galanis Easy and record the
equilibrium per-omega mean price under each value of `bonus`.

Player 0 is the manipulator (direction +1: wants high final price).
Players 1 and 2 are Bayesian best-responders. Reports the shift in
equilibrium mean price relative to the no-manipulator baseline, which
defines the per-bonus manipulation surplus.
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
from galanis_market.structures import STATE_LABELS  # noqa: E402


BONUSES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
ITERATIONS = 400
ACTIONS = 9
STRUCTURE = "t3s111y2"


def main() -> None:
    results = {}
    t_start = time.time()
    for bonus in BONUSES:
        print(f"=== bonus = {bonus} ===", flush=True)
        params = {
            "structure": STRUCTURE,
            "num_rounds": 3,
            "num_actions": ACTIONS,
            "manipulator_player": 0,
            "manipulator_direction": 1,
            "manipulator_bonus": bonus,
        }
        game = GalanisMarketGame(params)
        t0 = time.time()
        res = solve(game, iterations=ITERATIONS, log_every=None, verbose=False)
        elapsed = time.time() - t0
        nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
        profits = expected_profits(game, res.policy)
        agg = profits["__aggregate__"]
        print(f"  done in {elapsed:.1f}s  nash_conv={nc:.3e}  "
              f"mean log err={res.mean_log_error:.4f}  "
              f"decision_acc={agg['decision_accuracy']:.4f}  "
              f"manip market pnl={agg['market_pnl'][0]:+.5f}", flush=True)
        results[bonus] = {
            "elapsed": elapsed,
            "nash_conv": nc,
            "mean_log_error": res.mean_log_error,
            "aggregate": agg,
            "by_omega": {
                label: {
                    "x": res.price_by_omega[label]["x"],
                    "mean_price": res.price_by_omega[label]["mean_price"],
                    "median_price": res.price_by_omega[label]["median_price"],
                    "p_high": profits[label]["p_high"],
                    "returns": profits[label]["returns"],
                    "market_pnl": profits[label]["market_pnl"],
                }
                for label in STATE_LABELS
            },
        }

    out_path = _REPO / "results" / f"manipulator_sweep_{STRUCTURE}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "structure": STRUCTURE, "bonuses": BONUSES},
                   "results": results}, f, indent=2)

    print()
    print(f"=== Manipulation surplus on {STRUCTURE} ===")
    print("bonus   |  E[p|ω=a] |  E[p|ω=h] | mean LE | dec_acc | manip mkt pnl | informed pnl")
    for bonus in BONUSES:
        a = results[bonus]["by_omega"]["a"]["mean_price"]
        h = results[bonus]["by_omega"]["h"]["mean_price"]
        le = results[bonus]["mean_log_error"]
        agg = results[bonus]["aggregate"]
        informed = agg["market_pnl"][1] + agg["market_pnl"][2]
        print(f"{bonus:6.2f}  |   {a:.4f}  |   {h:.4f}  |  {le:.4f} | "
              f"{agg['decision_accuracy']:.4f}  |   {agg['market_pnl'][0]:+.5f}    |  {informed:+.5f}")
    print()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
