"""Canonical Deep CFR run for one Galanis structure at 9 rounds.

Usage:
    python canonical_run.py <structure> [iterations] [hidden] [train_steps]

Writes results to results/canonical_dcfr_9r_<structure>.json
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import jax

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from jax_futarchy.game import GalanisGame  # noqa: E402
from jax_futarchy.dcfr_canonical import (  # noqa: E402
    CanonicalDCFRConfig, make_canonical_dcfr, evaluate_canonical,
)


def main():
    structure = sys.argv[1] if len(sys.argv) > 1 else "t3s111y2"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    train_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    print(f"JAX: {jax.__version__} {jax.devices()}", flush=True)
    print(f"structure={structure} iter={iterations} hidden={hidden} ts={train_steps}", flush=True)

    game = GalanisGame(structure=structure, num_rounds=9, num_actions=5, b=0.01)
    config = CanonicalDCFRConfig(
        iterations=iterations,
        traversals_per_iter=1024,
        train_steps=train_steps,
        train_batch=1024,
        hidden=hidden,
        depth=3,
        lr=1e-3,
        buffer_capacity=500_000,
        retrain_from_scratch=False,
        seed=42,
    )
    train_loop, regret_net, strategy_net = make_canonical_dcfr(game, config)
    t0 = time.time()
    regret_params, strategy_params = train_loop(jax.random.PRNGKey(0), verbose=True)
    elapsed = time.time() - t0
    print(f"trained in {elapsed:.1f}s = {elapsed/60:.1f} min", flush=True)

    result = evaluate_canonical(game, strategy_net, strategy_params, n_samples=2000)
    print(f"mean LE: {result['mean_log_error']:.4f}", flush=True)
    print(f"median LE: {result['median_log_error']:.4f}", flush=True)
    for label, d in result["by_omega"].items():
        print(f"  {label}: X={d['x']} mean={d['mean_price']:.4f} median={d['median_price']:.4f}", flush=True)

    out_path = _REPO / "results" / f"canonical_dcfr_9r_{structure}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "structure": structure,
            "config": vars(config),
            "result": result,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
