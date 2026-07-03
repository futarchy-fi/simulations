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
from open_spiel.python.algorithms import external_sampling_mccfr

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
    elapsed_seconds_solve: float = 0.0  # solver-only time (excl. stats)
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


def mccfr_solve(
    game: GalanisMarketGame,
    iterations: int = 20000,
    nash_conv_every: Optional[int] = None,
    skip_nash_conv: bool = False,
    use_mc_stats: bool = False,
    mc_samples: int = 5000,
    verbose: bool = False,
) -> SolveResult:
    """Solve via OpenSpiel's external-sampling MCCFR.

    MCCFR samples paths through the game tree rather than enumerating
    all infosets. Per-iteration cost is O(tree depth * actions) rather
    than O(infosets), so it scales to the 6/9-round Galanis variants
    that are intractable for tabular CFR+.

    For large games (9 rounds), the NashConv computation and exhaustive
    tree-walk stats are intractable. Set ``skip_nash_conv=True`` to skip
    NashConv and ``use_mc_stats=True`` to use Monte Carlo sampling for
    per-omega price aggregation.
    """
    solver = external_sampling_mccfr.ExternalSamplingSolver(game)
    result = SolveResult(
        structure=game.structure_name,
        num_rounds=game.num_rounds,
        num_actions=game.num_actions,
        iterations=iterations,
    )
    t0 = time.time()
    for it in range(1, iterations + 1):
        solver.iteration()
        if not skip_nash_conv and nash_conv_every and (it % nash_conv_every == 0 or it == iterations):
            policy = solver.average_policy()
            nc = float(expl_mod.nash_conv(game, policy))
            result.nash_conv_trace.append((it, nc))
            if verbose:
                print(f"  iter {it:8d}  nash_conv = {nc:.6e}", flush=True)
    policy = solver.average_policy()
    if not skip_nash_conv and not result.nash_conv_trace:
        nc = float(expl_mod.nash_conv(game, policy))
        result.nash_conv_trace.append((iterations, nc))
        if verbose:
            print(f"  final nash_conv = {nc:.6e}", flush=True)
    result.elapsed_seconds_solve = time.time() - t0
    result.policy = policy
    if use_mc_stats:
        mc_populate_price_stats(game, policy, result, n_samples=mc_samples)
    else:
        _populate_price_stats(game, policy, result)
    result.elapsed_seconds = time.time() - t0
    return result


def mc_populate_price_stats(
    game: GalanisMarketGame,
    policy: pyspiel.Policy,
    result: SolveResult,
    n_samples: int = 5000,
    seed: int = 0,
) -> None:
    """Monte Carlo per-omega stats for large games where exhaustive tree
    walks are intractable. For each omega, sample n_samples trajectories
    from the policy and aggregate final prices.
    """
    import random
    rng = random.Random(seed)
    eps = 1e-15
    mean_log_err_accum = 0.0
    median_log_err_accum = 0.0
    for omega_idx in range(len(STATES)):
        prices: List[float] = []
        x = game.structure.x_of(STATES[omega_idx])
        for _ in range(n_samples):
            state = game.new_initial_state()
            state.apply_action(omega_idx)
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    action = rng.choices(actions, weights=probs, k=1)[0]
                else:
                    action_probs = policy.action_probabilities(state)
                    actions = list(action_probs.keys())
                    probs = list(action_probs.values())
                    action = rng.choices(actions, weights=probs, k=1)[0]
                state.apply_action(action)
            prices.append(float(state.final_price()))
        prices.sort()
        mean_price = sum(prices) / len(prices)
        median_price = prices[len(prices) // 2]

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


def expected_profits(
    game: GalanisMarketGame,
    policy: pyspiel.Policy,
    decision_threshold: float = 0.5,
) -> Dict[str, Dict[str, object]]:
    """Expected per-player payoffs and a decision proxy, per omega.

    Walks the full game tree under `policy` and records, for each chance
    outcome omega:

    * ``returns``      -- expected terminal returns per player, INCLUDING
                          any manipulator bonus term;
    * ``market_pnl``   -- expected pure LMSR trading profit per player
                          (excludes the manipulator's out-of-market bonus);
    * ``p_high``       -- probability that the decision statistic (final
                          price, or TWAP if game.decision_rule == 'twap')
                          is at or above ``decision_threshold`` (a decision
                          proxy for a would-be sponsor who acts iff the
                          market says the event is more likely than not);
    * ``mean_stat``    -- expected value of the decision statistic.

    Also returns an ``__aggregate__`` entry averaging over the uniform
    prior, including ``decision_accuracy`` = P[1{stat >= thr} == X].
    """
    n = game.num_players()
    out: Dict[str, Dict[str, object]] = {}
    agg_returns = np.zeros(n)
    agg_market = np.zeros(n)
    acc = 0.0
    stat_le = 0.0
    for omega_idx in range(len(STATES)):
        root = game.new_initial_state()
        root.apply_action(omega_idx)
        x = game.structure.x_of(STATES[omega_idx])
        returns_acc = np.zeros(n)
        market_acc = np.zeros(n)
        p_high = 0.0
        stat_acc = 0.0
        total_w = 0.0

        def _walk(state, weight):
            nonlocal p_high, stat_acc, total_w, returns_acc, market_acc
            if state.is_terminal():
                returns_acc += weight * np.asarray(state.returns())
                market_acc += weight * np.asarray(state._trader_profit)
                stat = state.decision_price()
                stat_acc += weight * stat
                if stat >= decision_threshold:
                    p_high += weight
                total_w += weight
                return
            for action, prob in policy.action_probabilities(state).items():
                if prob > 0.0:
                    _walk(state.child(action), weight * prob)

        _walk(root, 1.0)
        returns_acc /= total_w
        market_acc /= total_w
        p_high /= total_w
        stat_acc /= total_w
        out[STATE_LABELS[omega_idx]] = {
            "x": float(x),
            "returns": returns_acc.tolist(),
            "market_pnl": market_acc.tolist(),
            "p_high": float(p_high),
            "mean_stat": float(stat_acc),
        }
        agg_returns += returns_acc / len(STATES)
        agg_market += market_acc / len(STATES)
        acc += (p_high if x == 1 else 1.0 - p_high) / len(STATES)
        p_clip = float(np.clip(stat_acc, 1e-15, 1 - 1e-15))
        stat_le += -(x * np.log(p_clip) + (1 - x) * np.log(1 - p_clip)) / len(STATES)
    out["__aggregate__"] = {
        "returns": agg_returns.tolist(),
        "market_pnl": agg_market.tolist(),
        "decision_accuracy": float(acc),
        "stat_log_error": float(stat_le),
    }
    return out


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


__all__ = ["SolveResult", "solve", "solve_all", "expected_profits"]
