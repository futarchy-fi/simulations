"""Logarithmic Market Scoring Rule (LMSR) primitives.

The cost function for a binary market is

    C(q_Y, q_N) = b * log(exp(q_Y / b) + exp(q_N / b))

where b is the liquidity parameter and q_Y, q_N are the outstanding share
counts of the Yes and No assets. The marginal price of Yes is

    p_Y = exp(q_Y / b) / (exp(q_Y / b) + exp(q_N / b))

A trader who moves the state from (q_Y, q_N) to (q'_Y, q'_N) pays
C(q'_Y, q'_N) - C(q_Y, q_N) to the market maker and receives
(q'_Y - q_Y) Yes shares plus (q'_N - q_N) No shares.

At resolution, a Yes share pays 1 if the true outcome is Yes, 0 otherwise,
and symmetrically for No shares.

We parameterise the state by the marginal Yes price `p` and track each
trader's Yes-share holdings. Internally we use the log-odds representation
log(q_Y/b - q_N/b) which corresponds 1:1 with `p`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class LMSR:
    """Stateless LMSR helpers parameterised by liquidity `b`.

    Convention: we normalise the share state so q_N = 0 always, and shift
    q_Y to encode the current price. This is a WLOG reparameterisation
    because LMSR depends only on q_Y - q_N.
    """

    b: float

    def cost_at_price(self, p: float) -> float:
        """C(q_Y, 0) where q_Y/b = logit(p). Equals -b * log(1 - p)."""
        return -self.b * math.log1p(-p)

    def cost_to_move(self, p_from: float, p_to: float) -> float:
        """Cash paid to the market maker to move price p_from -> p_to."""
        return self.cost_at_price(p_to) - self.cost_at_price(p_from)

    def shares_to_move(self, p_from: float, p_to: float) -> float:
        """Net Yes shares acquired by the trader who moves p_from -> p_to.

        Positive => bought Yes (sold No). Negative => sold Yes (bought No).
        """
        return self.b * (logit(p_to) - logit(p_from))

    def trade_payoff(
        self, p_from: float, p_to: float, outcome_yes: bool
    ) -> float:
        """Profit at resolution for a trader who moves p_from -> p_to.

        In the q_N = 0 normalisation, `shares` is signed: positive means
        long Yes, negative means short Yes (equivalent to long No). At
        resolution a Yes share pays 1 iff Yes wins, else 0. Profit is
        the share payout minus the cost paid to the market maker, where
        `cost` is signed (negative cost = MM paid trader to take the
        short-Yes position).
        """
        shares = self.shares_to_move(p_from, p_to)
        cost = self.cost_to_move(p_from, p_to)
        payout = shares if outcome_yes else 0.0
        return payout - cost


__all__ = ["LMSR", "logit", "sigmoid"]
