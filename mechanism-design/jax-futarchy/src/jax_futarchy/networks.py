"""Flax neural network for the regret approximator in Deep CFR.

One small MLP per player; input is the info-state vector, output is a
vector of regret estimates (one per action).

The strategy is derived from the regret via regret-matching:

    pi(a) = max(R(a), 0) / sum_a max(R(a), 0)

with a uniform fallback when all regrets are <= 0.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class RegretNet(nn.Module):
    """MLP that maps info_state -> per-action regret estimate."""

    num_actions: int
    hidden: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.depth):
            x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.num_actions)(x)


def regret_matching(regrets: jnp.ndarray) -> jnp.ndarray:
    """Convert regret estimates to a strategy distribution."""
    positive = jnp.maximum(regrets, 0.0)
    z = positive.sum()
    uniform = jnp.ones_like(positive) / regrets.shape[-1]
    return jnp.where(z > 0, positive / z, uniform)


def sample_action(key: jnp.ndarray, regrets: jnp.ndarray) -> jnp.ndarray:
    """Sample one action proportional to regret-matched probabilities."""
    strategy = regret_matching(regrets)
    return jax.random.categorical(key, jnp.log(strategy + 1e-12))


__all__ = ["RegretNet", "regret_matching", "sample_action"]
