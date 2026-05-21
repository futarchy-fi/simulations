"""JAX-native Hanson conditional market.

Mirrors the JAX Galanis game but with two parallel LMSR markets, one per
policy. Action space = 2 * num_actions (pool choice * price target).
The winning pool resolves; the losing pool refunds (net zero on that side).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_futarchy.lmsr import lmsr_cost, lmsr_shares


SIGNAL_TABLE = jnp.array([
    [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0],
    [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0],
], dtype=jnp.int32)


# Metric: M(A) = at-least-2-yes; M(B) = at-least-1-yes.
METRIC_A = jnp.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=jnp.int32)
METRIC_B = jnp.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=jnp.int32)


class HansonState(NamedTuple):
    omega: jnp.ndarray
    signals: jnp.ndarray            # int32 [3]
    price_a_history: jnp.ndarray    # float32 [num_rounds + 1]
    price_b_history: jnp.ndarray    # float32 [num_rounds + 1]
    action_history: jnp.ndarray     # int32 [num_rounds]  (pool * K + price_idx)
    cur_step: jnp.ndarray           # int32 scalar
    holdings_shares: jnp.ndarray    # float32 [3, 2]  per (trader, market)
    holdings_cost: jnp.ndarray      # float32 [3, 2]
    finished: jnp.ndarray


@dataclass(frozen=True)
class HansonGame:
    num_rounds: int = 9
    num_actions: int = 5
    b: float = 0.01
    initial_price: float = 0.5

    @property
    def num_players(self) -> int:
        return 3

    @property
    def num_combined_actions(self) -> int:
        return 2 * self.num_actions

    @property
    def price_grid(self) -> jnp.ndarray:
        step = 1.0 / (self.num_actions + 1)
        return jnp.arange(1, self.num_actions + 1) * step

    def init(self, key: jnp.ndarray) -> HansonState:
        omega = jax.random.randint(key, (), 0, 8)
        return HansonState(
            omega=omega.astype(jnp.int32),
            signals=SIGNAL_TABLE[omega],
            price_a_history=jnp.zeros(self.num_rounds + 1)
                .at[0].set(self.initial_price),
            price_b_history=jnp.zeros(self.num_rounds + 1)
                .at[0].set(self.initial_price),
            action_history=jnp.zeros(self.num_rounds, dtype=jnp.int32),
            cur_step=jnp.array(0, dtype=jnp.int32),
            holdings_shares=jnp.zeros((3, 2), dtype=jnp.float32),
            holdings_cost=jnp.zeros((3, 2), dtype=jnp.float32),
            finished=jnp.array(False),
        )

    def current_player(self, state):
        return state.cur_step % 3

    def step(self, state: HansonState, action: jnp.ndarray) -> HansonState:
        active = self.current_player(state)
        market = action // self.num_actions
        price_idx = action % self.num_actions
        p_to = self.price_grid[price_idx]

        p_a_from = state.price_a_history[state.cur_step]
        p_b_from = state.price_b_history[state.cur_step]
        p_from = jnp.where(market == 0, p_a_from, p_b_from)

        shares = lmsr_shares(p_from, p_to, self.b)
        cost = lmsr_cost(p_from, p_to, self.b)

        new_holdings_shares = state.holdings_shares.at[active, market].add(shares)
        new_holdings_cost = state.holdings_cost.at[active, market].add(cost)

        # Update only the chosen market's price.
        new_price_a_history = jnp.where(
            market == 0,
            state.price_a_history.at[state.cur_step + 1].set(p_to),
            state.price_a_history.at[state.cur_step + 1].set(p_a_from),
        )
        new_price_b_history = jnp.where(
            market == 1,
            state.price_b_history.at[state.cur_step + 1].set(p_to),
            state.price_b_history.at[state.cur_step + 1].set(p_b_from),
        )

        new_step = state.cur_step + 1
        finished = new_step >= self.num_rounds
        return HansonState(
            omega=state.omega,
            signals=state.signals,
            price_a_history=new_price_a_history,
            price_b_history=new_price_b_history,
            action_history=state.action_history.at[state.cur_step].set(action),
            cur_step=new_step,
            holdings_shares=new_holdings_shares,
            holdings_cost=new_holdings_cost,
            finished=finished,
        )

    def info_state(self, state: HansonState, player: jnp.ndarray) -> jnp.ndarray:
        own_signal = state.signals[player]
        player_oh = jax.nn.one_hot(player, 3)
        signal_oh = jax.nn.one_hot(own_signal, 2)
        step_oh = jax.nn.one_hot(state.cur_step, self.num_rounds + 1)
        return jnp.concatenate([
            player_oh, signal_oh, step_oh,
            state.price_a_history, state.price_b_history,
        ])

    def info_state_dim(self) -> int:
        return 3 + 2 + (self.num_rounds + 1) + 2 * (self.num_rounds + 1)

    def terminal_profits(self, state: HansonState) -> jnp.ndarray:
        """At terminal: decision = market with higher final price; that
        market's payoff resolves, the other refunds cost (net zero)."""
        p_a = state.price_a_history[-1]
        p_b = state.price_b_history[-1]
        winning = jnp.where(p_a >= p_b, 0, 1)  # 0=A, 1=B
        omega = state.omega
        m = jnp.where(winning == 0, METRIC_A[omega], METRIC_B[omega])

        # Per trader profit in winning market:
        # If M = 1: payoff = shares; if M = 0: payoff = -cost.
        shares = state.holdings_shares[:, winning]  # [3]
        cost = state.holdings_cost[:, winning]
        payoff = jnp.where(m == 1, shares, -cost)
        # Losing market refunds (net zero, already excluded).
        return payoff

    def decision_a(self, state: HansonState) -> jnp.ndarray:
        return state.price_a_history[-1] >= state.price_b_history[-1]


__all__ = ["HansonGame", "HansonState"]
