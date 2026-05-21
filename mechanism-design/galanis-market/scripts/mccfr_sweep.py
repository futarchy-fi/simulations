"""Run MCCFR on Galanis 4 structures × 9 rounds, write JSON."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from galanis_market.game import GalanisMarketGame  # noqa: E402
from galanis_market.solve import mccfr_solve  # noqa: E402
from galanis_market.structures import STATE_LABELS  # noqa: E402


ITERATIONS = 50000
ACTIONS = 5
ROUNDS = 9
MC_SAMPLES = 2000


def run_one(structure: str) -> dict:
    print(f"=== {structure} rounds={ROUNDS} actions={ACTIONS} ===", flush=True)
    g = GalanisMarketGame({
        "structure": structure,
        "num_rounds": ROUNDS,
        "num_actions": ACTIONS,
    })
    t0 = time.time()
    res = mccfr_solve(
        g, iterations=ITERATIONS,
        skip_nash_conv=True, use_mc_stats=True, mc_samples=MC_SAMPLES,
        verbose=True,
    )
    print(f"  done in {time.time()-t0:.1f}s  mean LE={res.mean_log_error:.4f}", flush=True)
    return {
        "structure": structure,
        "iterations": res.iterations,
        "num_rounds": res.num_rounds,
        "num_actions": res.num_actions,
        "elapsed_seconds": res.elapsed_seconds,
        "elapsed_seconds_solve": res.elapsed_seconds_solve,
        "mean_log_error": res.mean_log_error,
        "median_log_error": res.median_log_error,
        "by_omega": {
            label: {
                "x": res.price_by_omega[label]["x"],
                "mean_price": res.price_by_omega[label]["mean_price"],
                "median_price": res.price_by_omega[label]["median_price"],
            }
            for label in STATE_LABELS
        },
    }


def main() -> None:
    results = []
    t_start = time.time()
    for structure in ["t3s111y2", "t3s110", "t3s111", "t3s111o2ye2"]:
        results.append(run_one(structure))
    out_path = _REPO / "results" / f"mccfr_9r_a{ACTIONS}_i{ITERATIONS}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": ITERATIONS, "actions": ACTIONS,
                              "rounds": ROUNDS, "mc_samples": MC_SAMPLES},
                   "results": results}, f, indent=2)
    print()
    print("=== Summary ===")
    print(f"{'structure':>16}  {'mean LE':>8}  {'median LE':>10}  {'time (s)':>9}")
    for r in results:
        print(f"{r['structure']:>16}  {r['mean_log_error']:>8.4f}  "
              f"{r['median_log_error']:>10.4f}  {r['elapsed_seconds']:>9.1f}")
    print()
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
