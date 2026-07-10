"""CRN-paired parallel-market manipulation experiment.

This is the deliberately narrow v0 described in ``MULTI.md``: K independent
one-round Gaussian BATCH-LMSR markets, competitive sizing, a fixed total
number of honest traders split evenly across markets, and one informed
adversary present in every market.  A budget allocation is the per-market
bounty coefficient from ``engine.manip_target_lmsr``; coefficients sum to B.

Draws are nested across K.  Market j always uses the same latent value,
adversary signal, and prefix of honest signals, so comparisons across K and
allocation strategies use common random numbers.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from batch_amm.envs import PRICE_EPS, clip_price
from batch_amm.lmsr_np import cost_to_move

STRATEGIES = ("concentrate", "uniform", "greedy")
_LOGIT_MAX = float(logit(1.0 - PRICE_EPS))


def prepare_panel(
    k: int,
    n_total: int,
    m: int,
    *,
    seed: int = 20260704,
    b: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Generate K independent Gaussian markets with nested CRN draws."""
    if k < 1 or m < 1 or n_total < 1 or n_total % k:
        raise ValueError("k and m must be positive and n_total divisible by k")
    n_honest = n_total // k
    v = np.empty((m, k))
    honest_signals = np.empty((m, k, n_honest))
    manip_signal = np.empty((m, k))
    for j in range(k):
        v[:, j] = np.random.default_rng(
            np.random.SeedSequence([seed, j, 0])
        ).standard_normal(m)
        # Generate the full prefix bank so market-j signal r is identical for
        # every K in which r survives the N/K split.
        eps = np.random.default_rng(
            np.random.SeedSequence([seed, j, 1])
        ).standard_normal((n_total, m))
        honest_signals[:, j, :] = v[:, j, None] + eps[:n_honest].T
        manip_signal[:, j] = v[:, j] + np.random.default_rng(
            np.random.SeedSequence([seed, j, 2])
        ).standard_normal(m)

    q_honest = clip_price(norm.cdf(honest_signals / np.sqrt(2.0)))
    q_manip = clip_price(norm.cdf(manip_signal / np.sqrt(2.0)))
    n_agents = n_honest + 1
    p_star = clip_price(
        norm.cdf(
            (honest_signals.sum(axis=2) + manip_signal) / np.sqrt(1.0 + n_agents)
        )
    )
    return {
        "k": k,
        "m": m,
        "n_total": n_total,
        "n_honest": n_honest,
        "n_agents": n_agents,
        "seed": seed,
        "b": b,
        "v": v,
        "payout": (v > 0.0).astype(float),
        "honest_signals": honest_signals,
        "manip_signal": manip_signal,
        "q_honest": q_honest,
        "q_manip": q_manip,
        "p_star": p_star,
        "weights": np.full(k, 1.0 / k),
        "scale": 1.0 / n_agents,
    }


def _manip_target(q: np.ndarray, bounty: np.ndarray, b: float) -> np.ndarray:
    """Vector form of engine.manip_target_lmsr, including zero bounties."""
    a = np.asarray(bounty, dtype=float)
    safe = np.where(a > 0.0, a, 1.0)
    target = ((safe - b) + np.sqrt((safe - b) ** 2 + 4.0 * safe * b * q)) / (
        2.0 * safe
    )
    return clip_price(np.where(a > 0.0, target, q))


def _price(panel: Dict, allocation: np.ndarray) -> np.ndarray:
    target = _manip_target(panel["q_manip"], allocation, panel["b"])
    total_logit = logit(panel["q_honest"]).sum(axis=2) + logit(target)
    return expit(np.clip(panel["scale"] * total_logit, -_LOGIT_MAX, _LOGIT_MAX))


def allocate_budget(
    panel: Dict,
    total_budget: float,
    strategy: str,
    *,
    grid_steps: int = 20,
) -> np.ndarray:
    """Return per-rep bounty allocations of shape (M,K), each summing to B.

    ``greedy`` is an oracle response-curve stress test, not an equilibrium:
    each grid quantum goes to the market with the largest next weighted price
    displacement.  It observes simulated price response, never v or payout.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if total_budget < 0.0 or grid_steps < 1:
        raise ValueError("total_budget must be nonnegative and grid_steps positive")
    m, k = panel["m"], panel["k"]
    allocation = np.zeros((m, k))
    if total_budget == 0.0:
        return allocation
    if k == 1:
        allocation[:, 0] = total_budget
        return allocation
    if strategy == "concentrate":
        allocation[:, 0] = total_budget
    elif strategy == "uniform":
        allocation[:] = total_budget / k
    else:
        quantum = total_budget / grid_steps
        current = _price(panel, allocation)
        rows = np.arange(m)
        for _ in range(grid_steps):
            candidate = _price(panel, allocation + quantum)
            marginal = panel["weights"][None, :] * (candidate - current)
            chosen = marginal.argmax(axis=1)
            allocation[rows, chosen] += quantum
            current[rows, chosen] = candidate[rows, chosen]
    return allocation


def _clear(panel: Dict, allocation: np.ndarray) -> Dict[str, np.ndarray]:
    """Clear all panel markets with BATCH.md's one-round LMSR arithmetic."""
    b, scale = panel["b"], panel["scale"]
    target_m = _manip_target(panel["q_manip"], allocation, b)
    x_h = scale * b * logit(panel["q_honest"])
    x_m = scale * b * logit(target_m)
    x_net = x_h.sum(axis=2) + x_m
    lp1 = np.clip(x_net / b, -_LOGIT_MAX, _LOGIT_MAX)
    x_exec_net = b * lp1
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(np.abs(x_net) >= 1e-14, x_exec_net / x_net, 1.0)
    x_h_exec = x_h * alpha[:, :, None]
    x_m_exec = x_m * alpha
    p1 = expit(lp1)
    dc = cost_to_move(np.full_like(p1, 0.5), p1, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(np.abs(x_exec_net) < 1e-14, 0.5, dc / x_exec_net)
    payout = panel["payout"]
    honest_pnl = (x_h_exec * (payout[:, :, None] - pi[:, :, None])).sum(axis=2)
    manip_pnl = x_m_exec * (payout - pi)
    honest_leak = ((pi - 0.5)[:, :, None] * x_h_exec).sum(axis=2)
    honest_volume = np.abs(x_h_exec).sum(axis=2)
    mm_pnl = dc - x_exec_net * payout
    conservation = honest_pnl + manip_pnl + mm_pnl
    return {
        "price": p1,
        "pi": pi,
        "honest_pnl": honest_pnl,
        "manip_pnl": manip_pnl,
        "honest_leak": honest_leak,
        "honest_volume": honest_volume,
        "conservation": conservation,
    }


def _stat(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    se = float(np.sqrt(np.mean((x - mean) ** 2) / len(x)))
    return {"mean": mean, "se": se, "ci95": 1.96 * se}


def run_strategy(
    panel: Dict,
    total_budget: float,
    strategy: str,
    *,
    grid_steps: int = 20,
) -> tuple[Dict, Dict[str, np.ndarray]]:
    """Run one allocation strategy and return JSON summary plus per-rep data."""
    allocation = allocate_budget(
        panel, total_budget, strategy, grid_steps=grid_steps
    )
    honest = _clear(panel, np.zeros_like(allocation))
    attacked = _clear(panel, allocation)
    w = panel["weights"][None, :]
    abs_error = np.abs(attacked["price"] - panel["p_star"])
    abs_error_h = np.abs(honest["price"] - panel["p_star"])
    distortion = np.abs(attacked["price"] - honest["price"])
    manip_cost = (honest["manip_pnl"] - attacked["manip_pnl"]).sum(axis=1)

    raw = {
        "iw_mean_abs_price_error": (abs_error * w).sum(axis=1),
        "iw_mean_abs_price_error_honest": (abs_error_h * w).sum(axis=1),
        "iw_mean_abs_price_error_damage": ((abs_error - abs_error_h) * w).sum(axis=1),
        "max_abs_price_error": abs_error.max(axis=1),
        "max_abs_price_error_honest": abs_error_h.max(axis=1),
        "max_abs_price_error_damage": abs_error.max(axis=1) - abs_error_h.max(axis=1),
        "iw_mean_abs_distortion": (distortion * w).sum(axis=1),
        "sum_abs_distortion": distortion.sum(axis=1),
        "max_abs_distortion": distortion.max(axis=1),
        "manip_cost": manip_cost,
        "honest_execution_shortfall": (attacked["honest_leak"] * w).sum(axis=1),
        "honest_execution_shortfall_honest": (honest["honest_leak"] * w).sum(axis=1),
        "honest_volume": (attacked["honest_volume"] * w).sum(axis=1),
        "honest_volume_honest": (honest["honest_volume"] * w).sum(axis=1),
        "honest_pnl": (attacked["honest_pnl"] * w).sum(axis=1),
        "honest_pnl_honest": (honest["honest_pnl"] * w).sum(axis=1),
    }
    metrics = {name: _stat(values) for name, values in raw.items()}
    mean_dist = metrics["iw_mean_abs_distortion"]["mean"]
    mean_sum_dist = metrics["sum_abs_distortion"]["mean"]
    mean_vol = metrics["honest_volume"]["mean"]
    ratios = {
        "manip_cost_per_iw_distortion": metrics["manip_cost"]["mean"]
        / max(mean_dist, 1e-300),
        "manip_cost_per_sum_distortion": metrics["manip_cost"]["mean"]
        / max(mean_sum_dist, 1e-300),
        "honest_leak_per_volume": metrics["honest_execution_shortfall"]["mean"]
        / max(mean_vol, 1e-300),
    }
    if total_budget:
        share = allocation / total_budget
        allocation_summary = {
            "market_mean_budget": allocation.mean(axis=0).tolist(),
            "mean_active_markets": float((allocation > 1e-15).sum(axis=1).mean()),
            "mean_max_share": float(share.max(axis=1).mean()),
            "mean_hhi": float((share**2).sum(axis=1).mean()),
        }
    else:
        allocation_summary = {
            "market_mean_budget": allocation.mean(axis=0).tolist(),
            "mean_active_markets": 0.0,
            "mean_max_share": 0.0,
            "mean_hhi": 0.0,
        }
    record = {
        "K": panel["k"],
        "N_total_honest": panel["n_total"],
        "N_honest_per_market": panel["n_honest"],
        "N_agents_per_market": panel["n_agents"],
        "M": panel["m"],
        "b": panel["b"],
        "total_budget": total_budget,
        "strategy": strategy,
        "grid_steps": grid_steps if strategy == "greedy" else None,
        "metrics": metrics,
        "ratios": ratios,
        "allocation": allocation_summary,
        "checks": {
            "budget_max_abs_error": float(
                np.abs(allocation.sum(axis=1) - total_budget).max()
            ),
            "conservation_maxabs": float(
                max(
                    np.abs(honest["conservation"]).max(),
                    np.abs(attacked["conservation"]).max(),
                )
            ),
        },
    }
    return record, raw


def paired_summary(raw: Dict[str, np.ndarray], reference: Dict[str, np.ndarray]) -> Dict:
    """CRN-paired A-reference summaries for every headline per-rep metric."""
    return {name: _stat(values - reference[name]) for name, values in raw.items()}


__all__ = [
    "STRATEGIES",
    "prepare_panel",
    "allocate_budget",
    "run_strategy",
    "paired_summary",
]
