"""End-to-end driver: solve all (structure, rounds) combinations with CFR+,
compute the equilibrium final-price distribution per chance outcome, and
write a JSON + plain-text report to ``results/``.

Usage::

    python scripts/solve_all.py --iterations 200 --num-actions 19 \
        --structures t3s111y2,t3s110,t3s111,t3s111o2ye2 \
        --rounds 3,6,9

The default configuration is tuned to run on a laptop CPU in minutes for
3-round games; 6- and 9-round games take longer because the info-state
count grows as O(num_actions ** (moves_per_player - 1)).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

# Allow running without `pip install -e .` by adding ../src to sys.path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from galanis_market.galanis_empirics import (  # noqa: E402
    EMPIRICAL_MEAN_LOG_ERROR_3R,
    EMPIRICAL_MEDIAN_LOG_ERROR,
    EMPIRICAL_MEDIAN_PRICE_AT_X1,
)
from galanis_market.myopic import myopic_final_prices  # noqa: E402
from galanis_market.solve import solve_all  # noqa: E402
from galanis_market.structures import STATE_LABELS, STRUCTURES  # noqa: E402


def _comma_list(spec: str, cast: Callable = str) -> List:
    return [cast(s.strip()) for s in spec.split(",") if s.strip()]


def _serialise(results, args) -> Dict:
    out = {
        "config": {
            "iterations": args.iterations,
            "num_actions": args.num_actions,
            "structures": args.structures,
            "rounds": args.rounds,
        },
        "runs": [],
    }
    for r in results:
        myopic = myopic_final_prices(
            STRUCTURES[r.structure], num_rounds=r.num_rounds
        )
        run = {
            "structure": r.structure,
            "num_rounds": r.num_rounds,
            "num_actions": r.num_actions,
            "iterations": r.iterations,
            "elapsed_seconds": r.elapsed_seconds,
            "nash_conv_trace": r.nash_conv_trace,
            "mean_log_error": r.mean_log_error,
            "median_log_error": r.median_log_error,
            "price_by_omega": {
                label: {
                    "x": r.price_by_omega[label]["x"],
                    "cfr_mean_price": r.price_by_omega[label]["mean_price"],
                    "cfr_median_price": r.price_by_omega[label]["median_price"],
                    "myopic_price": myopic[label],
                    "log_error_cfr": r.price_by_omega[label]["log_error"],
                    "median_log_error_cfr": r.price_by_omega[label][
                        "median_log_error"
                    ],
                }
                for label in STATE_LABELS
            },
        }
        out["runs"].append(run)
    return out


def _format_text(data: Dict) -> str:
    lines = []
    lines.append("Galanis Market — CFR+ equilibria vs paper empirics")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"iterations={data['config']['iterations']}  "
        f"num_actions={data['config']['num_actions']}"
    )
    lines.append("")
    for run in data["runs"]:
        struct = run["structure"]
        emp_mean = EMPIRICAL_MEAN_LOG_ERROR_3R.get(struct)
        emp_median = EMPIRICAL_MEDIAN_LOG_ERROR.get(struct)
        emp_price = EMPIRICAL_MEDIAN_PRICE_AT_X1.get(struct)
        lines.append(
            f"--- {struct}  rounds={run['num_rounds']}  "
            f"NashConv_final={run['nash_conv_trace'][-1][1]:.3e}  "
            f"t={run['elapsed_seconds']:.1f}s"
        )
        lines.append(
            f"  CFR mean log err    = {run['mean_log_error']:.4f}   "
            f"vs Galanis mean   = {emp_mean}"
        )
        lines.append(
            f"  CFR median log err  = {run['median_log_error']:.4f}   "
            f"vs Galanis median = {emp_median}   "
            f"(typical empirical price when X=1: {emp_price})"
        )
        lines.append(
            "  omega |  X  |  mean p CFR+ | median p CFR+ |  myopic Bayes"
        )
        lines.append(
            "  ------+-----+--------------+---------------+---------------"
        )
        for label in STATE_LABELS:
            d = run["price_by_omega"][label]
            lines.append(
                f"    {label}   | {int(d['x'])}.0 |    "
                f"{d['cfr_mean_price']:.4f}    |    "
                f"{d['cfr_median_price']:.4f}     |     "
                f"{d['myopic_price']:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-actions", type=int, default=19)
    parser.add_argument(
        "--structures",
        type=str,
        default="t3s111y2,t3s110,t3s111,t3s111o2ye2",
    )
    parser.add_argument("--rounds", type=str, default="3")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Compute NashConv every N iterations (0 = only at end).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(_REPO / "results")
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.structures = _comma_list(args.structures)
    args.rounds = _comma_list(args.rounds, cast=int)

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    results = solve_all(
        structures=args.structures,
        rounds=args.rounds,
        num_actions=args.num_actions,
        iterations=args.iterations,
        log_every=args.log_every if args.log_every > 0 else None,
        verbose=not args.quiet,
    )

    data = _serialise(results, args)
    tag = f"a{args.num_actions}_i{args.iterations}_" + "+".join(args.structures)
    stem = f"cfr_{tag}_r" + "+".join(str(r) for r in args.rounds)

    json_path = os.path.join(args.output_dir, f"{stem}.json")
    txt_path = os.path.join(args.output_dir, f"{stem}.txt")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(_format_text(data))

    print()
    print(_format_text(data))
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
