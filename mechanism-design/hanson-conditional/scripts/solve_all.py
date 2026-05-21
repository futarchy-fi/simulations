"""Driver: solve Hanson conditional at the requested (rounds, actions, iters)
and emit a JSON + text report to results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hanson_conditional.game import (  # noqa: E402
    HansonConditionalGame, STATE_LABELS, metric_under_policy
)
from hanson_conditional.solve import solve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-actions", type=int, default=7)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    game = HansonConditionalGame(
        {"num_rounds": args.num_rounds, "num_actions": args.num_actions}
    )
    t0 = time.time()
    res = solve(game, iterations=args.iterations, verbose=not args.quiet)

    data = {
        "config": {
            "num_rounds": args.num_rounds,
            "num_actions": args.num_actions,
            "iterations": args.iterations,
        },
        "elapsed_seconds": res.elapsed_seconds,
        "final_nash_conv": res.nash_conv_trace[-1][1] if res.nash_conv_trace else None,
        "decision_accuracy": res.decision_accuracy,
        "by_omega": {
            label: {
                **stats,
                "metric_A": metric_under_policy(0, omega_idx),
                "metric_B": metric_under_policy(1, omega_idx),
            }
            for omega_idx, label in enumerate(STATE_LABELS)
            for stats in [res.by_omega[label]]
        },
    }

    tag = f"a{args.num_actions}_i{args.iterations}_r{args.num_rounds}"
    json_path = os.path.join(args.output_dir, f"hanson_{tag}.json")
    txt_path = os.path.join(args.output_dir, f"hanson_{tag}.txt")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    lines = []
    lines.append(f"Hanson conditional CFR+ — rounds={args.num_rounds}, "
                 f"actions={args.num_actions}, iters={args.iterations}")
    lines.append(f"NashConv (final): {data['final_nash_conv']:.4e}")
    lines.append(f"Decision accuracy: {data['decision_accuracy']:.4f}")
    lines.append(f"Time: {data['elapsed_seconds']:.1f}s")
    lines.append("")
    lines.append("  omega | M(A) M(B) | E[p_A] | E[p_B] | Pr(A wins) | Pr(metric=1)")
    lines.append("  ------+-----------+--------+--------+------------+--------------")
    for label in STATE_LABELS:
        s = data["by_omega"][label]
        lines.append(
            f"    {label}   |   {s['metric_A']}    {s['metric_B']}    |"
            f" {s['p_A_mean']:.4f} | {s['p_B_mean']:.4f} |"
            f"   {s['decision_A_prob']:.4f}   |    {s['metric_realised_prob']:.4f}"
        )
    text = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(text + "\n")

    print()
    print(text)
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
