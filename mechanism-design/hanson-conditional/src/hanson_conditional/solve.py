"""CFR+ solver and equilibrium reporting for the Hanson conditional game.

Mirrors the galanis-market solver but emits Hanson-specific stats:
per-omega final price of *both* markets, the resulting decision (which
policy wins), and the probability that the decision is correct under
the chosen metric.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability as expl_mod
from open_spiel.python.algorithms import external_sampling_mccfr

from hanson_conditional.game import (
    HansonConditionalGame,
    STATES,
    STATE_LABELS,
    metric_under_policy,
)


@dataclass
class HansonSolveResult:
    num_rounds: int
    num_actions: int
    iterations: int
    nash_conv_trace: List[Tuple[int, float]] = field(default_factory=list)
    # Per-omega: {label: {"p_A_mean", "p_B_mean", "decision_A_prob", "ex_metric"}}
    by_omega: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decision_accuracy: float = 0.0  # P[chosen policy yields M=1] averaged over omega
    elapsed_seconds: float = 0.0
    # CFR average policy (referenced; not serialised by default).
    policy: Optional[pyspiel.Policy] = None


def _walk(state: pyspiel.State, policy: pyspiel.Policy, weight: float = 1.0):
    """Yield (final_state_view_dict, weight) tuples under the policy."""
    if state.is_terminal():
        yield {
            "p_a": float(state.final_market_prices()[0]),
            "p_b": float(state.final_market_prices()[1]),
            "winning": int(state._winning_market()),
        }, weight
        return
    action_probs = policy.action_probabilities(state)
    for action, prob in action_probs.items():
        if prob <= 0.0:
            continue
        child = state.child(action)
        yield from _walk(child, policy, weight * prob)


def solve(
    game: HansonConditionalGame,
    iterations: int = 50,
    log_every: Optional[int] = None,
    verbose: bool = False,
) -> HansonSolveResult:
    solver = cfr.CFRPlusSolver(game)
    result = HansonSolveResult(
        num_rounds=game.num_rounds,
        num_actions=game.num_actions,
        iterations=iterations,
    )
    t0 = time.time()
    for it in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if log_every and (it % log_every == 0 or it == iterations):
            policy = solver.average_policy()
            nc = float(expl_mod.nash_conv(game, policy))
            result.nash_conv_trace.append((it, nc))
            if verbose:
                print(f"  iter {it:4d}  nash_conv = {nc:.6e}")
    if not log_every:
        policy = solver.average_policy()
        nc = float(expl_mod.nash_conv(game, policy))
        result.nash_conv_trace.append((iterations, nc))
        if verbose:
            print(f"  final nash_conv = {nc:.6e}")
    result.elapsed_seconds = time.time() - t0
    avg_policy = solver.average_policy()
    result.policy = avg_policy

    # Per-omega aggregation under the equilibrium policy.
    correct_total = 0.0
    for omega_idx, label in enumerate(STATE_LABELS):
        state = game.new_initial_state()
        state.apply_action(omega_idx)
        p_a_sum = 0.0
        p_b_sum = 0.0
        decision_A_prob = 0.0
        correct_prob = 0.0
        total_weight = 0.0
        for view, w in _walk(state, avg_policy):
            p_a_sum += view["p_a"] * w
            p_b_sum += view["p_b"] * w
            if view["winning"] == 0:
                decision_A_prob += w
            m = metric_under_policy(view["winning"], omega_idx)
            correct_prob += m * w
            total_weight += w
        result.by_omega[label] = {
            "p_A_mean": p_a_sum / total_weight,
            "p_B_mean": p_b_sum / total_weight,
            "decision_A_prob": decision_A_prob / total_weight,
            "metric_realised_prob": correct_prob / total_weight,
        }
        correct_total += correct_prob / total_weight
    result.decision_accuracy = correct_total / len(STATES)
    return result


def expected_profits(
    game: HansonConditionalGame,
    policy: pyspiel.Policy,
) -> Dict[str, Dict[str, object]]:
    """Expected per-player payoffs per omega under `policy`.

    For each omega records:

    * ``returns``     -- expected terminal returns per player, INCLUDING
                         the manipulator's decision bonus;
    * ``market_pnl``  -- expected pure market P&L per player (the
                         manipulator's bonus is stripped out);
    * ``decision_A_prob`` -- probability policy A is implemented.

    ``__aggregate__`` averages over the uniform prior and includes
    ``decision_accuracy`` = P[metric realised under the chosen policy].
    """
    import numpy as np

    mp = game.manipulator_player
    bonus = game.manipulator_bonus
    preferred = 0 if game.manipulator_prefers_A == 1 else 1
    out: Dict[str, Dict[str, object]] = {}
    agg_returns = np.zeros(3)
    agg_market = np.zeros(3)
    acc = 0.0
    for omega_idx, label in enumerate(STATE_LABELS):
        root = game.new_initial_state()
        root.apply_action(omega_idx)
        returns_acc = np.zeros(3)
        market_acc = np.zeros(3)
        a_prob = 0.0
        metric_prob = 0.0
        total_w = 0.0

        def _walk(state, weight):
            nonlocal returns_acc, market_acc, a_prob, metric_prob, total_w
            if state.is_terminal():
                ret = np.asarray(state.returns())
                market = ret.copy()
                winning = state._winning_market()
                if 0 <= mp <= 2 and bonus != 0.0 and winning == preferred:
                    market[mp] -= bonus
                returns_acc += weight * ret
                market_acc += weight * market
                if winning == 0:
                    a_prob += weight
                metric_prob += weight * metric_under_policy(winning, omega_idx)
                total_w += weight
                return
            for action, prob in policy.action_probabilities(state).items():
                if prob > 0.0:
                    _walk(state.child(action), weight * prob)

        _walk(root, 1.0)
        returns_acc /= total_w
        market_acc /= total_w
        a_prob /= total_w
        metric_prob /= total_w
        out[label] = {
            "returns": returns_acc.tolist(),
            "market_pnl": market_acc.tolist(),
            "decision_A_prob": float(a_prob),
            "metric_realised_prob": float(metric_prob),
        }
        agg_returns += returns_acc / len(STATES)
        agg_market += market_acc / len(STATES)
        acc += metric_prob / len(STATES)
    out["__aggregate__"] = {
        "returns": agg_returns.tolist(),
        "market_pnl": agg_market.tolist(),
        "decision_accuracy": float(acc),
    }
    return out


def _mc_per_omega_stats(game, policy, n_samples: int, seed: int = 0) -> dict:
    """Monte Carlo per-omega aggregation for Hanson conditional."""
    import random
    rng = random.Random(seed)
    out = {}
    correct_total = 0.0
    for omega_idx, label in enumerate(STATE_LABELS):
        p_a_sum = 0.0
        p_b_sum = 0.0
        a_wins = 0
        correct = 0
        for _ in range(n_samples):
            state = game.new_initial_state()
            state.apply_action(omega_idx)
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    acts, probs = zip(*outcomes)
                    action = rng.choices(acts, weights=probs, k=1)[0]
                else:
                    ap = policy.action_probabilities(state)
                    acts = list(ap.keys())
                    probs = list(ap.values())
                    action = rng.choices(acts, weights=probs, k=1)[0]
                state.apply_action(action)
            pa, pb = state.final_market_prices()
            winning = state._winning_market()
            p_a_sum += pa
            p_b_sum += pb
            if winning == 0:
                a_wins += 1
            m = metric_under_policy(winning, omega_idx)
            correct += m
        out[label] = {
            "p_A_mean": p_a_sum / n_samples,
            "p_B_mean": p_b_sum / n_samples,
            "decision_A_prob": a_wins / n_samples,
            "metric_realised_prob": correct / n_samples,
        }
        correct_total += correct / n_samples
    return {"by_omega": out, "decision_accuracy": correct_total / len(STATES)}


def mccfr_solve(
    game: HansonConditionalGame,
    iterations: int = 50000,
    skip_nash_conv: bool = False,
    mc_samples: int = 3000,
    verbose: bool = False,
) -> HansonSolveResult:
    """Hanson MCCFR solver. For 9 rounds, NashConv is intractable;
    set ``skip_nash_conv=True``. Per-omega stats are always sampled
    (matching Galanis's mc_populate_price_stats approach)."""
    solver = external_sampling_mccfr.ExternalSamplingSolver(game)
    result = HansonSolveResult(
        num_rounds=game.num_rounds,
        num_actions=game.num_actions,
        iterations=iterations,
    )
    t0 = time.time()
    for _ in range(iterations):
        solver.iteration()
    policy = solver.average_policy()
    if not skip_nash_conv:
        nc = float(expl_mod.nash_conv(game, policy))
        result.nash_conv_trace.append((iterations, nc))
        if verbose:
            print(f"  final nash_conv = {nc:.6e}", flush=True)
    result.elapsed_seconds = time.time() - t0
    result.policy = policy
    mc = _mc_per_omega_stats(game, policy, n_samples=mc_samples)
    result.by_omega = mc["by_omega"]
    result.decision_accuracy = mc["decision_accuracy"]
    return result


__all__ = ["HansonSolveResult", "solve", "mccfr_solve", "expected_profits"]
