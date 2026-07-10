"""Run the parallel-small-markets experiment and write results JSONs.

Default usage runs a pilot (M=2,000) and then the BATCH.md-scale experiment
(M=20,000).  ``--pilot-only`` and ``--scaled-only`` are convenient for
iteration; both stages use identical seeds and nested common random numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from batch_amm.multi_market import (
    STRATEGIES,
    paired_summary,
    prepare_panel,
    run_strategy,
)

K_SWEEP = [1, 2, 4, 8, 16, 32]
BUDGETS = [0.05, 0.15, 0.5]
N_TOTAL = 96
SEED = 20260704
RESULTS = Path(__file__).resolve().parent / "results"


def run_stage(m: int, grid_steps: int) -> dict:
    records = []
    k1_raw = {}
    for k in K_SWEEP:
        panel = prepare_panel(k, N_TOTAL, m, seed=SEED)
        for budget in BUDGETS:
            concentrate_raw = None
            for strategy in STRATEGIES:
                record, raw = run_strategy(
                    panel, budget, strategy, grid_steps=grid_steps
                )
                if k == 1:
                    k1_raw[(budget, strategy)] = raw
                else:
                    record["paired_vs_K1"] = paired_summary(
                        raw, k1_raw[(budget, strategy)]
                    )
                if strategy == "concentrate":
                    concentrate_raw = raw
                else:
                    record["paired_vs_concentrate"] = paired_summary(
                        raw, concentrate_raw
                    )
                records.append(record)
        print(f"K={k} M={m} done", flush=True)
    return {
        "method": {
            "environment": "independent Gaussian v~N(0,1), s=v+eps",
            "mechanism": "one-round batch_lmsr, competitive sizing",
            "importance": "equal weights summing to one",
            "N_total_honest": N_TOTAL,
            "adversary": "one informed adversary present in every market",
            "K": K_SWEEP,
            "budgets": BUDGETS,
            "strategies": list(STRATEGIES),
            "greedy_grid_steps": grid_steps,
            "seed": SEED,
            "M": m,
            "crn": "market-index seeds and nested honest-signal prefixes across K",
        },
        "records": records,
    }


def dump(name: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    with path.open("w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pilot-only", action="store_true")
    mode.add_argument("--scaled-only", action="store_true")
    ap.add_argument("--pilot-m", type=int, default=2_000)
    ap.add_argument("--m", type=int, default=20_000)
    ap.add_argument("--grid-steps", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    if not args.scaled_only:
        dump("multi_pilot.json", run_stage(args.pilot_m, args.grid_steps))
    if not args.pilot_only:
        dump("multi.json", run_stage(args.m, args.grid_steps))
    print(f"multi-market sweeps done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
