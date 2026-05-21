"""Manipulator bonus sweep on MetaDAO CFMM (parallel to Hanson)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from metadao_cfmm.game import MetaDAOGame  # noqa: E402
from metadao_cfmm.solve import solve  # noqa: E402


BONUSES = [0.0, 0.005, 0.02, 0.05, 0.2]
ITERATIONS = 40
ACTIONS = 7
ROUNDS = 3
K = 0.001


def main() -> None:
    results = {}
    t_start = time.time()
    for bonus in BONUSES:
        print(f"=== bonus = {bonus} ===", flush=True)
        params = {
            "num_rounds": ROUNDS,
            "num_actions": ACTIONS,
            "K": K,
            "manipulator_player": 0,
            "manipulator_prefers_A": 1,
            "manipulator_bonus": bonus,
        }
        game = MetaDAOGame(params)
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

    out_path = _REPO / "results" / "metadao_manipulator_sweep.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "rounds": ROUNDS, "K": K, "bonuses": BONUSES},
                   "results": results}, f, indent=2)

    print()
    print("=== MetaDAO manipulation surplus ===")
    print(f"bonus   |  decision_accuracy")
    for bonus in BONUSES:
        print(f"{bonus:6.3f}  |       {results[bonus]['decision_accuracy']:.4f}")
    print()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
