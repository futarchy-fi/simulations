"""Sweep manipulator bonus values on the Hanson conditional game.

Player 0 = manipulator preferring policy A. We measure how much
Pr(A wins) shifts away from the Bayesian baseline as the bonus grows,
and what happens to overall decision accuracy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hanson_conditional.game import HansonConditionalGame  # noqa: E402
from hanson_conditional.solve import solve  # noqa: E402


BONUSES = [0.0, 0.005, 0.02, 0.05, 0.2]
ITERATIONS = 40
ACTIONS = 7
ROUNDS = 3


def main() -> None:
    results = {}
    t_start = time.time()
    for bonus in BONUSES:
        print(f"=== bonus = {bonus} ===", flush=True)
        params = {
            "num_rounds": ROUNDS,
            "num_actions": ACTIONS,
            "manipulator_player": 0,
            "manipulator_prefers_A": 1,
            "manipulator_bonus": bonus,
        }
        game = HansonConditionalGame(params)
        t0 = time.time()
        res = solve(game, iterations=ITERATIONS, verbose=False)
        elapsed = time.time() - t0
        nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
        print(f"  done in {elapsed:.1f}s  nash_conv={nc:.3e}  "
              f"decision_accuracy={res.decision_accuracy:.4f}", flush=True)
        results[bonus] = {
            "elapsed": elapsed,
            "nash_conv": nc,
            "decision_accuracy": res.decision_accuracy,
            "by_omega": {label: dict(stats) for label, stats in res.by_omega.items()},
        }

    out_path = _REPO / "results" / "hanson_manipulator_sweep.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "rounds": ROUNDS, "bonuses": BONUSES},
                   "results": results}, f, indent=2)

    print()
    print("=== Hanson manipulation surplus ===")
    print(f"bonus   |  decision_accuracy  |  P(A wins | ω=f)  |  P(A wins | ω=g)")
    for bonus in BONUSES:
        r = results[bonus]
        f_p = r["by_omega"]["f"]["decision_A_prob"]
        g_p = r["by_omega"]["g"]["decision_A_prob"]
        print(f"{bonus:6.3f}  |       {r['decision_accuracy']:.4f}      |     {f_p:.4f}       |     {g_p:.4f}")
    print()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
