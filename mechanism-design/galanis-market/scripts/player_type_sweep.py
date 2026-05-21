"""Player-type sweep on Galanis Easy.

Three configurations:
  baseline:  3 Bayesian (existing baseline from Chapter 1)
  naive:     player 0 = naive flat-prior (forced to play price = 0.5),
             players 1, 2 = Bayesian best-response
  insider:   player 0 = insider (observes all 3 signals = omega exactly),
             players 1, 2 = Bayesian best-response

Reports mean log error per configuration. Intuition: a naive player
adds noise to the price-discovery process, so accuracy should drop.
An insider player has more information, so accuracy should improve
(potentially to the discretisation floor exactly).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from galanis_market.game import GalanisMarketGame  # noqa: E402
from galanis_market.solve import solve  # noqa: E402
from galanis_market.structures import STATE_LABELS  # noqa: E402


ITERATIONS = 40
ACTIONS = 9
STRUCTURE = "t3s111y2"


def run_one(label: str, extra_params: dict) -> dict:
    print(f"=== {label} ===", flush=True)
    params = {
        "structure": STRUCTURE,
        "num_rounds": 3,
        "num_actions": ACTIONS,
        **extra_params,
    }
    game = GalanisMarketGame(params)
    t0 = time.time()
    res = solve(game, iterations=ITERATIONS, log_every=None, verbose=False)
    elapsed = time.time() - t0
    nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
    print(f"  done in {elapsed:.1f}s  nash_conv={nc:.3e}  "
          f"mean log err={res.mean_log_error:.4f}", flush=True)
    return {
        "label": label,
        "elapsed": elapsed,
        "nash_conv": nc,
        "mean_log_error": res.mean_log_error,
        "median_log_error": res.median_log_error,
        "by_omega": {
            label_: {
                "x": res.price_by_omega[label_]["x"],
                "mean_price": res.price_by_omega[label_]["mean_price"],
                "median_price": res.price_by_omega[label_]["median_price"],
            }
            for label_ in STATE_LABELS
        },
    }


def main() -> None:
    results = {}
    t_start = time.time()
    results["baseline"] = run_one("baseline (3 Bayesian)", {})
    results["naive_p0"] = run_one("naive p0", {"naive_player": 0})
    results["insider_p0"] = run_one("insider p0", {"insider_player": 0})

    out_path = _REPO / "results" / f"player_types_{STRUCTURE}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "structure": STRUCTURE},
                   "results": results}, f, indent=2)

    print()
    print("=== Player-type comparison on Galanis Easy ===")
    print(f"{'config':>25}  {'mean LE':>8}  {'median LE':>10}")
    for k, r in results.items():
        print(f"{r['label']:>25}  {r['mean_log_error']:>8.4f}  {r['median_log_error']:>10.4f}")
    print()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
