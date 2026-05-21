"""CFR+ solver for the MetaDAO CFMM game (parallel to Hanson's)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability as expl_mod
from open_spiel.python.algorithms import external_sampling_mccfr

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


def _mc_per_omega_stats(game, policy, n_samples: int, seed: int = 0) -> dict:
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
    game: MetaDAOGame,
    iterations: int = 50000,
    skip_nash_conv: bool = False,
    mc_samples: int = 3000,
    verbose: bool = False,
) -> MetaDAOSolveResult:
    """MetaDAO MCCFR solver, analogous to Hanson's."""
    solver = external_sampling_mccfr.ExternalSamplingSolver(game)
    result = MetaDAOSolveResult(
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
    mc = _mc_per_omega_stats(game, policy, n_samples=mc_samples)
    result.by_omega = mc["by_omega"]
    result.decision_accuracy = mc["decision_accuracy"]
    return result


__all__ = ["MetaDAOSolveResult", "solve", "mccfr_solve"]
