"""LMSR pricing primitives in JAX. All functions are JIT-compatible.

We use the q_N = 0 normalisation: the state is a single marginal Yes
price p in (0, 1). LMSR cost to push p_from -> p_to is

    cost(p_from, p_to) = -b * log(1 - p_to) + b * log(1 - p_from)

with the convention that positive cost means the trader paid cash.
The trader's net Yes-share position is

    shares(p_from, p_to) = b * (logit(p_to) - logit(p_from))

(negative = short Yes). At resolution with outcome = Yes:

    payoff = shares - cost (since cost was paid in cash, shares pay 1 each)

For outcome = No, in the q_N = 0 representation: payoff = 0 - cost = -cost.
"""

from __future__ import annotations

import jax.numpy as jnp


def logit(p: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(p / (1.0 - p))


def lmsr_cost(p_from: jnp.ndarray, p_to: jnp.ndarray, b: float) -> jnp.ndarray:
    """LMSR cost-to-move (positive = trader pays)."""
    return b * (jnp.log(1.0 - p_from) - jnp.log(1.0 - p_to))


def lmsr_shares(p_from: jnp.ndarray, p_to: jnp.ndarray, b: float) -> jnp.ndarray:
    """Net Yes shares acquired by trader moving p_from -> p_to."""
    return b * (logit(p_to) - logit(p_from))


def lmsr_payoff(
    p_from: jnp.ndarray, p_to: jnp.ndarray, outcome_yes: jnp.ndarray, b: float
) -> jnp.ndarray:
    """Trader's terminal payoff = shares (if Yes wins) or -cost (if No)."""
    shares = lmsr_shares(p_from, p_to, b)
    cost = lmsr_cost(p_from, p_to, b)
    return jnp.where(outcome_yes, shares, -cost)


__all__ = ["logit", "lmsr_cost", "lmsr_shares", "lmsr_payoff"]
