"""CFR+ solver and equilibrium analysis utilities.

Runs OpenSpiel's tabular CFR+ on a GalanisMarketGame instance, reports
exploitability over iterations, and computes derived statistics that map
the equilibrium policy back to economic quantities -- in particular the
expected final price of the security conditional on each chance outcome
(omega).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability as expl_mod

from galanis_market.game import GalanisMarketGame
from galanis_market.structures import STATES, STATE_LABELS


@dataclass
class SolveResult:
    structure: str
    num_rounds: int
    num_actions: int
    iterations: int
    # Tuples of (iter, nash_conv). For 2p zero-sum NashConv == exploitability;
    # for general-sum N-player it is the sum across players of best-response
    # improvement over the current average policy. Useful as a convergence
    # signal even though it lacks the formal Nash guarantee in N>=3 general-sum.
    nash_conv_trace: List[Tuple[int, float]] = field(default_factory=list)
    # Final-price stats per omega (8 states): {omega_label: {"mean": .., "p50": .., "x": 0/1}}
    price_by_omega: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Overall summary (averaged over omegas, weighted by uniform prior).
    mean_log_error: float = 0.0
    median_log_error: float = 0.0
    # CFR average policy (referenced; not serialised by default).
    policy: Optional[pyspiel.Policy] = None
    elapsed_seconds: float = 0.0


def solve(
    game: GalanisMarketGame,
    iterations: int = 200,
    log_every: Optional[int] = None,
    nash_conv_at_end: bool = True,
    verbose: bool = False,
) -> SolveResult:
    """Run CFR+ for `iterations` steps and return aggregate stats.

    Parameters
    ----------
    iterations
        Number of CFR+ iterations.
    log_every
        If set, compute NashConv every `log_every` iterations (expensive --
        each call walks the full game tree per player). If None, NashConv
        is computed once at the end (cheap).
    nash_conv_at_end
        Whether to compute a final NashConv even when log_every is None.
    """
    solver = cfr.CFRPlusSolver(game)
    result = SolveResult(
        structure=game.structure_name,
        num_rounds=game.num_rounds,
        num_actions=game.num_actions,
        iterations=iterations,
    )
    t0 = time.time()
    for it in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if log_every is not None and (it % log_every == 0 or it == iterations):
            policy = solver.average_policy()
            nc = float(expl_mod.nash_conv(game, policy))
            result.nash_conv_trace.append((it, nc))
            if verbose:
                print(f"  iter {it:4d}  nash_conv = {nc:.6e}")
    if log_every is None and nash_conv_at_end:
        policy = solver.average_policy()
        nc = float(expl_mod.nash_conv(game, policy))
        result.nash_conv_trace.append((iterations, nc))
        if verbose:
            print(f"  iter {iterations:4d}  nash_conv (final) = {nc:.6e}")
    result.elapsed_seconds = time.time() - t0
    result.policy = solver.average_policy()
    _populate_price_stats(game, result.policy, result)
    return result


def _populate_price_stats(
    game: GalanisMarketGame,
    policy: pyspiel.Policy,
    result: SolveResult,
) -> None:
    """Compute final-price distribution per chance outcome.

    Walks the game tree from each chance outcome, weighting paths by the
    average policy's action probabilities. Records the mean final price,
    median final price, log error (computed from mean), and the
    full (price, prob) histogram under the equilibrium.
    """
    eps = 1e-15
    mean_log_err_accum = 0.0
    median_log_err_accum = 0.0
    for omega_idx in range(len(STATES)):
        state = game.new_initial_state()
        state.apply_action(omega_idx)
        hist = _final_price_distribution(state, policy)
        mean_price = sum(p * w for p, w in hist)
        median_price = _weighted_median([(p, w) for p, w in hist])
        x = game.structure.x_of(STATES[omega_idx])

        def _log_err(p_val: float) -> float:
            p = float(np.clip(p_val, eps, 1 - eps))
            return -(x * np.log(p) + (1 - x) * np.log(1 - p))

        result.price_by_omega[STATE_LABELS[omega_idx]] = {
            "x": float(x),
            "mean_price": mean_price,
            "median_price": median_price,
            "log_error": _log_err(mean_price),
            "median_log_error": _log_err(median_price),
        }
        mean_log_err_accum += _log_err(mean_price)
        median_log_err_accum += _log_err(median_price)
    result.mean_log_error = mean_log_err_accum / len(STATES)
    result.median_log_error = median_log_err_accum / len(STATES)


def _final_price_distribution(
    state: pyspiel.State,
    policy: pyspiel.Policy,
    weight: float = 1.0,
) -> List[Tuple[float, float]]:
    """Return list of (final_price, probability) under `policy` from `state`."""
    if state.is_terminal():
        return [(float(state.final_price()), weight)]
    action_probs = policy.action_probabilities(state)
    out: List[Tuple[float, float]] = []
    for action, prob in action_probs.items():
        if prob <= 0.0:
            continue
        child = state.child(action)
        out.extend(_final_price_distribution(child, policy, weight * prob))
    return out


def _weighted_median(items: List[Tuple[float, float]]) -> float:
    """Compute the 50th percentile of a (value, weight) sample."""
    if not items:
        return float("nan")
    items_sorted = sorted(items, key=lambda x: x[0])
    total = sum(w for _, w in items_sorted)
    cum = 0.0
    for v, w in items_sorted:
        cum += w
        if cum >= total / 2.0:
            return v
    return items_sorted[-1][0]


def solve_all(
    structures: List[str],
    rounds: List[int],
    num_actions: int = 9,
    iterations: int = 200,
    log_every: Optional[int] = None,
    verbose: bool = True,
) -> List[SolveResult]:
    """Solve all (structure, rounds) combinations."""
    results: List[SolveResult] = []
    for structure in structures:
        for r in rounds:
            game = GalanisMarketGame(
                {
                    "structure": structure,
                    "num_rounds": r,
                    "num_actions": num_actions,
                }
            )
            if verbose:
                print(f"=== {structure} rounds={r} actions={num_actions} ===")
            res = solve(
                game,
                iterations=iterations,
                log_every=log_every,
                verbose=verbose,
            )
            if verbose:
                final_nc = res.nash_conv_trace[-1][1] if res.nash_conv_trace else float("nan")
                print(
                    f"  done in {res.elapsed_seconds:.1f}s  "
                    f"final nash_conv = {final_nc:.4e}  "
                    f"mean log err = {res.mean_log_error:.4f}"
                )
            results.append(res)
    return results


__all__ = ["SolveResult", "solve", "solve_all"]
