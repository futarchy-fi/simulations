"""T2u: entry with TYPE UNCERTAINTY.

Variant of the entry-sweep T2 treatment (2 informed incumbents + an
entrant who uniquely observes bit a) in which nature draws the entrant's
type after omega: BRIBED (receives the saturation-level price bonus)
with probability q, HONEST with probability 1-q. The realised type is
private to the entrant. q = 1 recovers the common-knowledge manipulator
of the plain entry sweep.

Question this adjudicates: under common knowledge the equilibrium
defence is full discounting (information exclusion). Can a *possibly*
bribed entrant hide in the honest pool and retain price influence --
and does that make average decisions better (his info is used with
prob 1-q) or worse (the bribed type now pollutes prices that others
partially trust)?

For each config we solve CFR+ and then report statistics CONDITIONAL
on the realised type: mean decision statistic, p_high, decision
accuracy, and per-player market P&L per omega, plus the aggregate.

Usage: python t2u_sweep.py [--qs 0.25,0.5] [--positions first,last]
                           [--bonus 0.2] [--iterations 400]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from galanis_market.game import GalanisMarketGame  # noqa: E402
from galanis_market.solve import solve  # noqa: E402
from galanis_market.structures import STATES, STATE_LABELS  # noqa: E402

STRUCTURE = "t3s111y2"
ACTIONS = 9
THRESHOLD = 0.5


def conditional_stats(game, policy, type_action):
    """Per-omega expected stats conditional on the entrant's realised
    type (0 = honest, 1 = bribed). Walks the tree from each
    (omega, type) pair, so the chance weights of the type node are
    conditioned away."""
    n = game.num_players()
    out = {}
    acc = 0.0
    agg_market = np.zeros(n)
    agg_stat = 0.0
    for omega_idx in range(len(STATES)):
        root = game.new_initial_state()
        root.apply_action(omega_idx)
        root.apply_action(type_action)  # type chance node
        x = game.structure.x_of(STATES[omega_idx])
        market_acc = np.zeros(n)
        p_high = 0.0
        stat_acc = 0.0
        total_w = 0.0

        def _walk(state, weight):
            nonlocal p_high, stat_acc, total_w, market_acc
            if state.is_terminal():
                market_acc += weight * np.asarray(state._trader_profit)
                stat = state.decision_price()
                stat_acc += weight * stat
                if stat >= THRESHOLD:
                    p_high += weight
                total_w += weight
                return
            for action, prob in policy.action_probabilities(state).items():
                if prob > 0.0:
                    _walk(state.child(action), weight * prob)

        _walk(root, 1.0)
        market_acc /= total_w
        p_high /= total_w
        stat_acc /= total_w
        out[STATE_LABELS[omega_idx]] = {
            "x": float(x),
            "market_pnl": market_acc.tolist(),
            "p_high": float(p_high),
            "mean_stat": float(stat_acc),
        }
        acc += (p_high if x == 1 else 1.0 - p_high) / len(STATES)
        agg_market += market_acc / len(STATES)
        agg_stat += stat_acc / len(STATES)
    out["__aggregate__"] = {
        "decision_accuracy": float(acc),
        "market_pnl": agg_market.tolist(),
        "mean_stat": float(agg_stat),
    }
    return out


def main() -> None:
    qs = [0.25, 0.5]
    positions = ["first", "last"]
    bonus = 0.2
    iterations = 400
    args = sys.argv
    for i, a in enumerate(args):
        if a == "--qs":
            qs = [float(x) for x in args[i + 1].split(",")]
        elif a == "--positions":
            positions = args[i + 1].split(",")
        elif a == "--bonus":
            bonus = float(args[i + 1])
        elif a == "--iterations":
            iterations = int(args[i + 1])

    results = {}
    t_start = time.time()
    for pos in positions:
        signals, seat = (("a,b,c", 0) if pos == "first" else ("b,c,a", 2))
        for q in qs:
            cname = f"T2u-{pos}-q{q}"
            params = {
                "structure": STRUCTURE, "num_players": 3, "num_rounds": 3,
                "num_actions": ACTIONS, "signals": signals,
                "manipulator_player": seat, "manipulator_direction": 1,
                "manipulator_bonus": bonus, "manipulator_prob": q,
                "decision_rule": "final",
            }
            game = GalanisMarketGame(params)
            t0 = time.time()
            res = solve(game, iterations=iterations, log_every=None,
                        verbose=False)
            elapsed = time.time() - t0
            nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
            by_type = {
                "honest": conditional_stats(game, res.policy, 0),
                "bribed": conditional_stats(game, res.policy, 1),
            }
            acc_h = by_type["honest"]["__aggregate__"]["decision_accuracy"]
            acc_b = by_type["bribed"]["__aggregate__"]["decision_accuracy"]
            acc_mix = (1 - q) * acc_h + q * acc_b
            print(
                f"[{cname}] done in {elapsed:.0f}s nc={nc:.2e} "
                f"acc_mix={acc_mix:.4f} acc_honest={acc_h:.4f} "
                f"acc_bribed={acc_b:.4f}",
                flush=True,
            )
            results[cname] = {
                "params": params,
                "elapsed": elapsed,
                "nash_conv": nc,
                "mean_log_error": res.mean_log_error,
                "acc_mixture": acc_mix,
                "by_type": by_type,
            }

    tag = "_".join(positions) + "_q" + "_".join(str(q) for q in qs)
    out_path = _REPO / "results" / f"t2u_{tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"iterations": iterations, "bonus": bonus,
                              "actions": ACTIONS, "structure": STRUCTURE,
                              "qs": qs, "positions": positions},
                   "results": results}, f, indent=2)
    print(f"\nTotal elapsed {time.time() - t_start:.0f}s; wrote {out_path}")


if __name__ == "__main__":
    main()
