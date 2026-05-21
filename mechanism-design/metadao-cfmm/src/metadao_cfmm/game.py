"""OpenSpiel implementation of MetaDAO-style two-sided conditional
CFMM prediction markets. Mirrors Hanson conditional (Chapter 2) but
swaps the LMSR market-maker for two binary constant-product pools,
one per policy.

Setup (minimal version):
* K = 2 policies (A, B), each with its own binary CFMM pool.
* 8 chance omegas as in Galanis Easy partitions.
* 3 traders, rotating; each turn pick (pool, target_price_grid_idx).
* Decision = pool with higher final implied probability of M = 1.
* Winning pool resolves via CFMM payoff. Losing pool refunds (the
  trader's net position there is set to zero -- net 0 on losing side).
* Metrics: M(A) = at-least-two-yes, M(B) = at-least-one-yes (same as
  Hanson default).

Manipulator support is identical in shape to Hanson: a designated
player gets a flat bonus when their preferred policy wins.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyspiel

from metadao_cfmm.cfmm import BinaryCFMM


STATES: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
    (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0),
)
STATE_LABELS: Tuple[str, ...] = tuple("abcdefgh")

NUM_POLICIES = 2


def metric_under_policy(policy: int, omega_idx: int) -> int:
    state = STATES[omega_idx]
    if policy == 0:
        return int(sum(state) >= 2)
    return int(sum(state) >= 1)


def _default_price_grid(num_actions: int) -> List[float]:
    if num_actions < 2:
        raise ValueError("num_actions must be at least 2")
    step = 1.0 / (num_actions + 1)
    return [round((i + 1) * step, 6) for i in range(num_actions)]


_GAME_TYPE = pyspiel.GameType(
    short_name="python_metadao_cfmm",
    long_name="Python MetaDAO CFMM",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=3,
    min_num_players=3,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={
        "num_rounds": 3,
        "num_actions": 7,
        "K": 1.0,
        "initial_price": 0.5,
        "manipulator_player": -1,
        "manipulator_prefers_A": 1,
        "manipulator_bonus": 0.0,
        "naive_player": -1,
        "insider_player": -1,
    },
)


def _build_game_info(num_actions: int, num_rounds: int) -> pyspiel.GameInfo:
    return pyspiel.GameInfo(
        num_distinct_actions=NUM_POLICIES * num_actions,
        max_chance_outcomes=len(STATES),
        num_players=3,
        # CFMM payoff can be larger in magnitude than LMSR; bound loosely.
        min_utility=-10.0 * num_rounds,
        max_utility=10.0 * num_rounds,
        utility_sum=None,
        max_game_length=num_rounds + 1,
    )


class MetaDAOGame(pyspiel.Game):
    def __init__(self, params: Optional[dict] = None):
        params = dict(params or {})
        self.num_rounds: int = int(params.get("num_rounds", 3))
        self.num_actions: int = int(params.get("num_actions", 7))
        self.K: float = float(params.get("K", 1.0))
        self.initial_price: float = float(params.get("initial_price", 0.5))
        self.manipulator_player: int = int(params.get("manipulator_player", -1))
        self.manipulator_prefers_A: int = int(params.get("manipulator_prefers_A", 1))
        self.manipulator_bonus: float = float(params.get("manipulator_bonus", 0.0))
        self.naive_player: int = int(params.get("naive_player", -1))
        self.insider_player: int = int(params.get("insider_player", -1))

        if self.num_rounds not in (3, 6, 9):
            raise ValueError("num_rounds must be 3, 6, or 9")
        if not 0.0 < self.initial_price < 1.0:
            raise ValueError("initial_price must lie strictly in (0, 1)")
        if self.manipulator_player not in (-1, 0, 1, 2):
            raise ValueError("manipulator_player must be -1, 0, 1, or 2")
        if self.manipulator_prefers_A not in (0, 1):
            raise ValueError("manipulator_prefers_A must be 0 or 1")
        if self.naive_player not in (-1, 0, 1, 2):
            raise ValueError("naive_player must be -1, 0, 1, or 2")
        if self.insider_player not in (-1, 0, 1, 2):
            raise ValueError("insider_player must be -1, 0, 1, or 2")

        self.price_grid: List[float] = _default_price_grid(self.num_actions)
        self.cfmm: BinaryCFMM = BinaryCFMM(K=self.K)
        super().__init__(_GAME_TYPE, _build_game_info(self.num_actions, self.num_rounds), params)

    def new_initial_state(self) -> "MetaDAOState":
        return MetaDAOState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return MetaDAOObserver(self)


class MetaDAOState(pyspiel.State):
    def __init__(self, game: MetaDAOGame):
        super().__init__(game)
        self._num_rounds = game.num_rounds
        self._num_actions = game.num_actions
        self._price_grid = list(game.price_grid)
        self._cfmm = game.cfmm
        self._initial_price = game.initial_price
        self._manipulator_player = game.manipulator_player
        self._manipulator_prefers_A = game.manipulator_prefers_A
        self._manipulator_bonus = game.manipulator_bonus
        self._naive_player = game.naive_player
        self._insider_player = game.insider_player
        mid_idx = min(
            range(len(game.price_grid)),
            key=lambda i: abs(game.price_grid[i] - 0.5),
        )
        self._naive_actions = [mid_idx, game.num_actions + mid_idx]

        self._omega_idx: Optional[int] = None
        self._market_prices: List[List[float]] = [
            [game.initial_price], [game.initial_price]
        ]
        self._action_history: List[Tuple[int, int]] = []
        # Per-trader, per-market accumulated (shares_held, no_tokens_paid).
        self._holdings: List[List[Tuple[float, float]]] = [
            [(0.0, 0.0), (0.0, 0.0)] for _ in range(3)
        ]

    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._omega_idx is None:
            return pyspiel.PlayerId.CHANCE
        return len(self._action_history) % 3

    def is_terminal(self) -> bool:
        return (
            self._omega_idx is not None
            and len(self._action_history) >= self._num_rounds
        )

    def chance_outcomes(self):
        assert self.is_chance_node()
        prob = 1.0 / len(STATES)
        return [(i, prob) for i in range(len(STATES))]

    def _legal_actions(self, player: int) -> List[int]:
        assert player >= 0
        if player == self._naive_player:
            return list(self._naive_actions)
        return list(range(self._num_actions * NUM_POLICIES))

    def _decode_action(self, action: int) -> Tuple[int, int]:
        return action // self._num_actions, action % self._num_actions

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self._omega_idx = int(action)
            return
        active = self.current_player()
        market, price_idx = self._decode_action(action)
        p_from = self._market_prices[market][-1]
        p_to = self._price_grid[price_idx]
        shares = self._cfmm.shares_to_move(p_from, p_to)
        cost = self._cfmm.cost_to_move(p_from, p_to)
        prev_shares, prev_cost = self._holdings[active][market]
        self._holdings[active][market] = (
            prev_shares + shares, prev_cost + cost,
        )
        self._market_prices[market].append(p_to)
        self._action_history.append((market, price_idx))

    def _action_to_string(self, player: int, action: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            return f"omega={action}"
        market, price_idx = self._decode_action(action)
        market_label = "A" if market == 0 else "B"
        return f"pool={market_label},p={self._price_grid[price_idx]:.4f}"

    def _winning_market(self) -> int:
        p_a = self._market_prices[0][-1]
        p_b = self._market_prices[1][-1]
        return 0 if p_a >= p_b else 1

    def returns(self) -> List[float]:
        if not self.is_terminal():
            return [0.0, 0.0, 0.0]
        assert self._omega_idx is not None
        winning = self._winning_market()
        m = metric_under_policy(winning, self._omega_idx)
        out: List[float] = []
        for trader in range(3):
            shares, cost = self._holdings[trader][winning]
            # CFMM: in the winning market, trader's net shares are
            # `shares` (signed). Their cost in No-tokens is `cost`. At
            # resolution, Yes-tokens pay 1 iff M=1; No-tokens pay 1
            # iff M=0. Since the trader gave up `cost` No-tokens to
            # acquire `shares` Yes-tokens (signs flipped if they did
            # the reverse trade), payoff is:
            #   m == 1:  +shares (Yes pays out)  ;  No-tokens are 0.
            #   m == 0:  -cost (they owe back the No-tokens they took
            #             out, or get the surrendered ones back as 0
            #             value).
            # Combined into a single expression:
            payoff = shares if m == 1 else -cost
            out.append(payoff)
        if 0 <= self._manipulator_player <= 2 and self._manipulator_bonus != 0.0:
            preferred_market = 0 if self._manipulator_prefers_A == 1 else 1
            if winning == preferred_market:
                out[self._manipulator_player] += self._manipulator_bonus
        return out

    def information_state_string(self, player: Optional[int] = None) -> str:
        if player is None:
            player = self.current_player()
        if self._omega_idx is None:
            cell = "?"
        elif player == self._insider_player:
            cell = f"omega{self._omega_idx}"
        else:
            cell = str(STATES[self._omega_idx][player])
        public = ",".join(f"{m}:{p}" for m, p in self._action_history)
        return f"p{player}|d={cell}|hist=[{public}]"

    def observation_string(self, player: Optional[int] = None) -> str:
        return self.information_state_string(player)

    def market_prices(self, market: int) -> List[float]:
        return list(self._market_prices[market])

    def final_market_prices(self) -> Tuple[float, float]:
        return self._market_prices[0][-1], self._market_prices[1][-1]


class MetaDAOObserver:
    def __init__(self, game: MetaDAOGame):
        self._game = game
        self.tensor = np.zeros(1, dtype=np.float32)
        self.dict = {"placeholder": self.tensor}

    def set_from(self, state: MetaDAOState, player: int) -> None:
        self.tensor.fill(0)

    def string_from(self, state: MetaDAOState, player: int) -> str:
        return state.information_state_string(player)


def register() -> None:
    try:
        pyspiel.register_game(_GAME_TYPE, MetaDAOGame)
    except pyspiel.SpielError:
        pass


register()


__all__ = [
    "MetaDAOGame",
    "MetaDAOState",
    "MetaDAOObserver",
    "register",
    "STATES",
    "STATE_LABELS",
    "metric_under_policy",
]
