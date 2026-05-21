"""CFR+ solver for the MetaDAO CFMM game (parallel to Hanson's)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability as expl_mod

from metadao_cfmm.game import (
    MetaDAOGame,
    STATES,
    STATE_LABELS,
    metric_under_policy,
)


@dataclass
class MetaDAOSolveResult:
    num_rounds: int
    num_actions: int
    iterations: int
    nash_conv_trace: List[Tuple[int, float]] = field(default_factory=list)
    by_omega: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decision_accuracy: float = 0.0
    elapsed_seconds: float = 0.0


def _walk(state: pyspiel.State, policy: pyspiel.Policy, weight: float = 1.0):
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
    game: MetaDAOGame,
    iterations: int = 40,
    verbose: bool = False,
) -> MetaDAOSolveResult:
    solver = cfr.CFRPlusSolver(game)
    result = MetaDAOSolveResult(
        num_rounds=game.num_rounds,
        num_actions=game.num_actions,
        iterations=iterations,
    )
    t0 = time.time()
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    policy = solver.average_policy()
    nc = float(expl_mod.nash_conv(game, policy))
    result.nash_conv_trace.append((iterations, nc))
    if verbose:
        print(f"  final nash_conv = {nc:.6e}")
    result.elapsed_seconds = time.time() - t0
    correct_total = 0.0
    for omega_idx, label in enumerate(STATE_LABELS):
        state = game.new_initial_state()
        state.apply_action(omega_idx)
        p_a_sum = 0.0
        p_b_sum = 0.0
        decision_A_prob = 0.0
        correct_prob = 0.0
        total_weight = 0.0
        for view, w in _walk(state, policy):
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


__all__ = ["MetaDAOSolveResult", "solve"]
