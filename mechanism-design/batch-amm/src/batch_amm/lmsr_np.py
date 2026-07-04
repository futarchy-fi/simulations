"""Vectorised (numpy) LMSR primitives.

Same conventions as galanis_market.lmsr (q_N = 0 normalisation: the market
state is fully described by the marginal Yes price p; C(p) = -b*log(1-p);
shares to move p0 -> p1 are b*(logit(p1) - logit(p0))), but operating on
numpy arrays so we can run tens of thousands of Monte Carlo markets at once.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit as sigmoid  # noqa: F401  (re-exported)
from scipy.special import logit  # noqa: F401  (re-exported)


def cost_at_price(p: np.ndarray, b: float) -> np.ndarray:
    """C(q_Y, 0) with q_Y/b = logit(p). Equals -b * log(1 - p)."""
    return -b * np.log1p(-np.asarray(p, dtype=float))


def cost_to_move(p_from: np.ndarray, p_to: np.ndarray, b: float) -> np.ndarray:
    """Cash paid to the market maker to move the price p_from -> p_to."""
    return cost_at_price(p_to, b) - cost_at_price(p_from, b)


def shares_to_move(p_from: np.ndarray, p_to: np.ndarray, b: float) -> np.ndarray:
    """Net Yes shares acquired by the trader who moves p_from -> p_to."""
    return b * (logit(p_to) - logit(p_from))


__all__ = ["cost_at_price", "cost_to_move", "shares_to_move", "logit", "sigmoid"]
