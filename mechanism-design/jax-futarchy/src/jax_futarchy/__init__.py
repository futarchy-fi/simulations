"""JAX-native futarchy games + Deep CFR.

Modules:
    game: pure-functional Galanis market in JAX (JIT-compiled step, vmap-able)
    lmsr: LMSR pricing primitives in JAX
    networks: Flax neural networks (regret + strategy)
    dcfr: Deep CFR training loop
"""

from jax_futarchy.game import GalanisGame, GalanisState
from jax_futarchy.lmsr import lmsr_cost, lmsr_shares, lmsr_payoff

__all__ = [
    "GalanisGame",
    "GalanisState",
    "lmsr_cost",
    "lmsr_shares",
    "lmsr_payoff",
]
