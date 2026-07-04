"""Player-type sweep on Hanson conditional markets."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hanson_conditional.game import HansonConditionalGame  # noqa: E402
from hanson_conditional.solve import solve  # noqa: E402


ITERATIONS = 40
ACTIONS = 7
ROUNDS = 3


def run_one(label: str, extra_params: dict) -> dict:
    print(f"=== {label} ===", flush=True)
    params = {"num_rounds": ROUNDS, "num_actions": ACTIONS, **extra_params}
    game = HansonConditionalGame(params)
    t0 = time.time()
    res = solve(game, iterations=ITERATIONS, verbose=False)
    elapsed = time.time() - t0
    nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
    print(f"  done in {elapsed:.1f}s  nash_conv={nc:.3e}  "
          f"decision_accuracy={res.decision_accuracy:.4f}", flush=True)
    return {
        "label": label,
        "elapsed": elapsed,
        "nash_conv": nc,
        "decision_accuracy": res.decision_accuracy,
        "by_omega": {label_: dict(stats) for label_, stats in res.by_omega.items()},
    }


def main() -> None:
    results = {}
    t_start = time.time()
    results["baseline"] = run_one("baseline (3 Bayesian)", {})
    results["naive_p0"] = run_one("naive p0", {"naive_player": 0})
    results["insider_p0"] = run_one("insider p0", {"insider_player": 0})

    out_path = _REPO / "results" / "hanson_player_types.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "rounds": ROUNDS},
                   "results": results}, f, indent=2)

    print()
    print("=== Hanson player-type comparison ===")
    for k, r in results.items():
        print(f"  {r['label']:>25}  decision_accuracy = {r['decision_accuracy']:.4f}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
