"""JAX-native Galanis market.

Design principles:
  - All state is a NamedTuple of JAX arrays (PyTree). No Python objects.
  - Step is a pure function (state, action, key) -> (new_state, reward, done).
  - JIT-compatible: no Python control flow that depends on traced values.
  - vmap-able: a single function call can process B parallel games at once.

State arrays:
  - omega: int [, ] = realized state index in {0..7}, drawn at init.
  - signals: int [3] = the three binary signals (d_a, d_b, d_c) derived
    from omega.
  - price_history: float [max_steps + 1] = public price after each step.
  - action_history: int [max_steps] = sequence of player actions (price
    grid indices).
  - cur_step: int = number of actions taken so far (0..max_steps).
  - trader_profits: float [3] = accumulated LMSR profit per trader.
  - finished: bool = whether the game is over.

Information state for player p (the input to a neural net policy):
  - own_signal: int (0 or 1 for "easy" partitions; 0..3 for "very hard")
  - price_history (real values), padded with zeros for unfilled slots
  - cur_step (one-hot)
  - player ID (one-hot)
This gives a fixed-length vector input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_futarchy.lmsr import lmsr_payoff


# 8 omega states encoded as 3 binary signals: rows omega -> (d_a, d_b, d_c).
SIGNAL_TABLE = jnp.array([
    [1, 1, 1],  # a
    [1, 1, 0],  # b
    [1, 0, 1],  # c
    [1, 0, 0],  # d
    [0, 1, 1],  # e
    [0, 1, 0],  # f
    [0, 0, 1],  # g
    [0, 0, 0],  # h
], dtype=jnp.int32)


# Structures: each gives an X(omega) value.
# Easy (t3s111y2): at-least-2-yes.
# Medium (t3s110): all-yes (true state has 2 yes -> X=0).
# Hard (t3s111): all-yes (true state has 3 yes -> X=1). Same payoff as Medium.
# Very Hard (t3s111o2ye2): exactly-2-yes.
STRUCTURE_X_TABLE = {
    "t3s111y2":     jnp.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=jnp.int32),
    "t3s110":       jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
    "t3s111":       jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
    "t3s111o2ye2":  jnp.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=jnp.int32),
}


class GalanisState(NamedTuple):
    omega: jnp.ndarray         # int32 scalar
    signals: jnp.ndarray       # int32 [3]
    price_history: jnp.ndarray  # float32 [max_steps + 1]
    action_history: jnp.ndarray  # int32 [max_steps]
    cur_step: jnp.ndarray      # int32 scalar
    trader_profits: jnp.ndarray  # float32 [3]
    finished: jnp.ndarray      # bool scalar


@dataclass(frozen=True)
class GalanisGame:
    """Stateless JAX game (config + functions).

    Use `init` to draw a new game state from a PRNG key, `step` to apply
    an action, and `info_state` to extract the input vector for a player's
    network policy.
    """

    structure: str = "t3s111y2"
    num_rounds: int = 9
    num_actions: int = 11
    b: float = 0.01
    initial_price: float = 0.5

    @property
    def x_table(self) -> jnp.ndarray:
        return STRUCTURE_X_TABLE[self.structure]

    @property
    def price_grid(self) -> jnp.ndarray:
        # K equally-spaced prices on (0, 1).
        step = 1.0 / (self.num_actions + 1)
        return jnp.arange(1, self.num_actions + 1) * step

    @property
    def num_players(self) -> int:
        return 3

    def init(self, key: jnp.ndarray) -> GalanisState:
        omega = jax.random.randint(key, (), 0, 8)
        signals = SIGNAL_TABLE[omega]
        price_history = (
            jnp.zeros(self.num_rounds + 1, dtype=jnp.float32)
            .at[0]
            .set(self.initial_price)
        )
        return GalanisState(
            omega=omega.astype(jnp.int32),
            signals=signals,
            price_history=price_history,
            action_history=jnp.zeros(self.num_rounds, dtype=jnp.int32),
            cur_step=jnp.array(0, dtype=jnp.int32),
            trader_profits=jnp.zeros(3, dtype=jnp.float32),
            finished=jnp.array(False),
        )

    def current_player(self, state: GalanisState) -> jnp.ndarray:
        return state.cur_step % 3

    def step(self, state: GalanisState, action: jnp.ndarray) -> GalanisState:
        """Apply `action` (price grid index) to the current state.

        Trader who acts is determined by cur_step % 3. They snap the price
        to grid[action]. LMSR profit accrues based on the resolved X
        (computed at terminal time, but accumulated incrementally here so
        we don't need state.x_table inside the network input).
        """
        active = self.current_player(state)
        p_from = state.price_history[state.cur_step]
        p_to = self.price_grid[action]
        x = self.x_table[state.omega]
        outcome_yes = x == 1
        profit_delta = lmsr_payoff(p_from, p_to, outcome_yes, self.b)
        # Update trader_profits[active] += profit_delta
        new_profits = state.trader_profits.at[active].add(profit_delta)
        new_price_history = state.price_history.at[state.cur_step + 1].set(p_to)
        new_action_history = state.action_history.at[state.cur_step].set(action)
        new_step = state.cur_step + 1
        finished = new_step >= self.num_rounds
        return GalanisState(
            omega=state.omega,
            signals=state.signals,
            price_history=new_price_history,
            action_history=new_action_history,
            cur_step=new_step,
            trader_profits=new_profits,
            finished=finished,
        )

    def info_state(self, state: GalanisState, player: jnp.ndarray) -> jnp.ndarray:
        """Per-player observation vector for the neural net.

        Encoding:
          - player one-hot [3]
          - own signal one-hot [2]
          - cur_step one-hot [num_rounds + 1]
          - price_history [num_rounds + 1] (real values; future slots are 0)
        Total dimension = 3 + 2 + (R+1) + (R+1) = 5 + 2*(R+1).
        """
        own_signal = state.signals[player]
        player_oh = jax.nn.one_hot(player, self.num_players)
        signal_oh = jax.nn.one_hot(own_signal, 2)
        step_oh = jax.nn.one_hot(state.cur_step, self.num_rounds + 1)
        return jnp.concatenate([player_oh, signal_oh, step_oh, state.price_history])

    def info_state_dim(self) -> int:
        return self.num_players + 2 + (self.num_rounds + 1) + (self.num_rounds + 1)


__all__ = ["GalanisGame", "GalanisState", "SIGNAL_TABLE", "STRUCTURE_X_TABLE"]
