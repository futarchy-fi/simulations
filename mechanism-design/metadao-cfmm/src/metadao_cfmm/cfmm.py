"""Binary constant-product market-maker primitives for MetaDAO-style
conditional prediction markets.

Each pool holds two token types: Yes-tokens and No-tokens. The pool
maintains the invariant ``y_yes * y_no = K`` (the constant product).
The marginal price of Yes is the ratio of No-tokens to total tokens,

    p_yes = y_no / (y_yes + y_no)

To increase ``p_yes`` from ``p`` to ``p'`` the trader swaps ``y_no``
tokens for ``y_yes`` tokens (giving up No, receiving Yes). The cost
in No-tokens to move the price from ``p`` to ``p'`` is

    delta_no = y_no_initial - K**0.5 * sqrt((1-p') / p')

This module exposes the small set of pure functions needed by the
OpenSpiel game wrapper: cost-to-move, shares-acquired, payoff-at-
resolution. We follow the same q_N = 0 normalisation convention as
the LMSR primitives in galanis_market.lmsr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryCFMM:
    """Stateless helpers for a binary CFMM with constant product K = 1.

    We parameterise by the marginal Yes price p. The reserves at price
    p (given total constant product K) are

        y_yes = sqrt(K * p / (1 - p))   # if you parameterise differently,
        y_no  = sqrt(K * (1 - p) / p)   # adjust here.

    For simplicity (and to make manipulation analysis interpretable)
    we set K = 1. Real-world MetaDAO pools have K much larger; the
    qualitative geometry is the same.
    """

    K: float = 1.0

    def reserves_at_price(self, p: float) -> tuple[float, float]:
        """(y_yes, y_no) consistent with the constant-product invariant
        and a marginal Yes price of `p`. The marginal price of Yes is
        p = y_no / (y_yes + y_no): when there is little Yes in the
        pool, swapping one Yes for No yields lots of No back, so Yes
        is expensive (p high). With K = y_yes * y_no, we have

            y_yes = sqrt(K * (1 - p) / p)
            y_no  = sqrt(K * p / (1 - p))

        so y_yes * y_no = K and y_no / (y_yes + y_no) = p, as required.
        """
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie strictly in (0, 1)")
        scale = math.sqrt(self.K)
        y_yes = scale * math.sqrt((1.0 - p) / p)
        y_no = scale * math.sqrt(p / (1.0 - p))
        return y_yes, y_no

    def cost_to_move(self, p_from: float, p_to: float) -> float:
        """Number of *No* tokens the trader must surrender to push the
        Yes price from `p_from` up to `p_to`.

        Moving Yes price up requires adding No tokens to the pool
        (and receiving Yes tokens out). Hence cost > 0 when p_to > p_from
        and cost < 0 when p_to < p_from (the trader is net receiving No).
        """
        _, y_no_from = self.reserves_at_price(p_from)
        _, y_no_to = self.reserves_at_price(p_to)
        return y_no_to - y_no_from

    def shares_to_move(self, p_from: float, p_to: float) -> float:
        """Net Yes shares acquired by the trader who moves p_from -> p_to.

        Moving the price up removes Yes from the pool (the trader buys
        them); so shares > 0 when p_to > p_from.
        """
        y_yes_from, _ = self.reserves_at_price(p_from)
        y_yes_to, _ = self.reserves_at_price(p_to)
        return y_yes_from - y_yes_to

    def trade_payoff(
        self, p_from: float, p_to: float, outcome_yes: bool
    ) -> float:
        """Payoff at resolution for a trader who moves p_from -> p_to.

        Trader holds `shares_to_move` Yes shares net (signed). If Yes
        wins they pay/receive that many tokens; the cost-of-No they
        gave up (or received) settles symmetrically. The payoff is
        the resolved value of the trader's net token position.
        """
        shares = self.shares_to_move(p_from, p_to)  # change in Yes
        cost = self.cost_to_move(p_from, p_to)  # No tokens surrendered
        if outcome_yes:
            return shares  # No tokens are worthless; Yes tokens worth 1
        # outcome No: Yes tokens worthless, but trader is short on No
        # by (cost) units (they surrendered cost No tokens). So they
        # owe (-cost) at resolution.
        return -cost


__all__ = ["BinaryCFMM"]
