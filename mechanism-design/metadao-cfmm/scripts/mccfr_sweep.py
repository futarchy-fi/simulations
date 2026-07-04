"""Run MCCFR on binary CFMM (MetaDAO-inspired) × 9 rounds."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from metadao_cfmm.game import MetaDAOGame  # noqa: E402
from metadao_cfmm.solve import mccfr_solve  # noqa: E402


ITERATIONS = 50000
ACTIONS = 5
ROUNDS = 9
K = 0.001
MC_SAMPLES = 2000


def main() -> None:
    print(f"=== Binary CFMM MCCFR rounds={ROUNDS} actions={ACTIONS} K={K} iters={ITERATIONS} ===", flush=True)
    g = MetaDAOGame({"num_rounds": ROUNDS, "num_actions": ACTIONS, "K": K})
    t0 = time.time()
    res = mccfr_solve(g, iterations=ITERATIONS, skip_nash_conv=True,
                     mc_samples=MC_SAMPLES, verbose=True)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s, decision_accuracy={res.decision_accuracy:.4f}", flush=True)
    out_path = _REPO / "results" / f"metadao_mccfr_9r_a{ACTIONS}_i{ITERATIONS}.json"
    out_path.parent.mkdir(exist_ok=True)
    data = {
        "config": {"iterations": ITERATIONS, "actions": ACTIONS,
                   "rounds": ROUNDS, "K": K, "mc_samples": MC_SAMPLES},
        "decision_accuracy": res.decision_accuracy,
        "elapsed_seconds": res.elapsed_seconds,
        "by_omega": res.by_omega,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
