"""OpenSpiel implementation of Hanson outcome-conditional prediction markets.

The canonical Hanson futarchy setup: there are K policies and one metric M.
For each policy k, we run a separate prediction market on
``M | policy k is implemented``. The policy with the higher final
implied probability of M is chosen, and only that market resolves
(traders in the other market are refunded their cost).

Minimal version implemented here (configurable extension below):

* K = 2 policies, A and B.
* Metric M is binary.
* Three traders, each privately observing one bit d_a, d_b, d_c
  (same partition as the Galanis Easy structure).
* M(policy A, omega) = 1 iff sum(omega) >= 2  (Galanis Easy rule).
* M(policy B, omega) = 1 iff sum(omega) >= 1  (less restrictive --
  policy B almost always succeeds on the metric).
* Decision: policy with higher implied probability of M after the
  final trading round wins. Ties break in favour of A by convention.
* Trades in the winning market resolve via LMSR; trades in the
  losing market refund cost (zero net payoff for those trades).
* num_rounds rounds, players rotate in fixed order. On each turn the
  active trader chooses (market in {A, B}, target price on the grid).

We share LMSR primitives with the Galanis chapter via the
``galanis_market.lmsr`` module that lives in ``mechanism-design/
galanis-market/``. Install it editable in the same environment.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyspiel

from galanis_market.lmsr import LMSR  # shared primitive


# 8 omega states encoded as (d_a, d_b, d_c).
STATES: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
    (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0),
)
STATE_LABELS: Tuple[str, ...] = tuple("abcdefgh")

NUM_POLICIES = 2  # A and B


def metric_under_policy(policy: int, omega_idx: int) -> int:
    state = STATES[omega_idx]
    if policy == 0:  # policy A: at-least-two-yes (Galanis Easy)
        return int(sum(state) >= 2)
    return int(sum(state) >= 1)  # policy B: at-least-one-yes


def _default_price_grid(num_actions: int) -> List[float]:
    if num_actions < 2:
        raise ValueError("num_actions must be at least 2")
    step = 1.0 / (num_actions + 1)
    return [round((i + 1) * step, 6) for i in range(num_actions)]


_GAME_TYPE = pyspiel.GameType(
    short_name="python_hanson_conditional",
    long_name="Python Hanson Conditional Market",
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
        "b": 0.01,
        "initial_price": 0.5,
        "manipulator_player": -1,
        "manipulator_prefers_A": 1,
        "manipulator_bonus": 0.0,
        "naive_player": -1,
        "insider_player": -1,
    },
)


def _build_game_info(num_actions: int, num_rounds: int) -> pyspiel.GameInfo:
    # Combined action = (market, price target). Total = 2 * num_actions.
    total_actions = NUM_POLICIES * num_actions
    return pyspiel.GameInfo(
        num_distinct_actions=total_actions,
        max_chance_outcomes=len(STATES),
        num_players=3,
        min_utility=-float(num_rounds),
        max_utility=float(num_rounds),
        utility_sum=None,
        max_game_length=num_rounds + 1,
    )


class HansonConditionalGame(pyspiel.Game):
    def __init__(self, params: Optional[dict] = None):
        params = dict(params or {})
        self.num_rounds: int = int(params.get("num_rounds", 3))
        self.num_actions: int = int(params.get("num_actions", 7))
        self.b: float = float(params.get("b", 0.01))
        self.initial_price: float = float(params.get("initial_price", 0.5))
        # Manipulator: if `manipulator_player` in {0,1,2}, that player's
        # utility gets a bonus of `manipulator_bonus` whenever the
        # implemented decision matches their preference. The preference
        # is encoded as policy A (1) or policy B (0).
        self.manipulator_player: int = int(params.get("manipulator_player", -1))
        self.manipulator_prefers_A: int = int(params.get("manipulator_prefers_A", 1))
        self.manipulator_bonus: float = float(params.get("manipulator_bonus", 0.0))
        # Naive: action restricted to (any market, price = 0.5).
        # Insider: information state includes the full omega.
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
        self.lmsr: LMSR = LMSR(b=self.b)
        super().__init__(_GAME_TYPE, _build_game_info(self.num_actions, self.num_rounds), params)

    def new_initial_state(self) -> "HansonConditionalState":
        return HansonConditionalState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return HansonObserver(self)


class HansonConditionalState(pyspiel.State):
    def __init__(self, game: HansonConditionalGame):
        super().__init__(game)
        self._num_rounds = game.num_rounds
        self._num_actions = game.num_actions
        self._price_grid = list(game.price_grid)
        self._lmsr = game.lmsr
        self._initial_price = game.initial_price
        self._manipulator_player = game.manipulator_player
        self._manipulator_prefers_A = game.manipulator_prefers_A
        self._manipulator_bonus = game.manipulator_bonus
        self._naive_player = game.naive_player
        self._insider_player = game.insider_player
        # Naive: actions restricted to {pool A @ 0.5, pool B @ 0.5}.
        mid_idx = min(
            range(len(game.price_grid)),
            key=lambda i: abs(game.price_grid[i] - 0.5),
        )
        self._naive_actions = [mid_idx, game.num_actions + mid_idx]

        self._omega_idx: Optional[int] = None
        # Per-market price history. Market 0 = policy A, market 1 = policy B.
        self._market_prices: List[List[float]] = [
            [game.initial_price], [game.initial_price]
        ]
        # Public action history: (market, price_idx) per turn.
        self._action_history: List[Tuple[int, int]] = []
        # Per-trader, per-market accumulated (shares_held, cost_paid) tuples.
        # Resolved at terminal time: only the winning market's holdings pay
        # out; the losing market refunds cost (net 0 on that side).
        self._holdings: List[List[Tuple[float, float]]] = [
            [(0.0, 0.0), (0.0, 0.0)] for _ in range(3)
        ]

    # ---- Sequencing ----

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

    # ---- Chance ----

    def chance_outcomes(self):
        assert self.is_chance_node()
        prob = 1.0 / len(STATES)
        return [(i, prob) for i in range(len(STATES))]

    # ---- Actions ----

    def _legal_actions(self, player: int) -> List[int]:
        assert player >= 0
        if player == self._naive_player:
            return list(self._naive_actions)
        return list(range(self._num_actions * NUM_POLICIES))

    def _decode_action(self, action: int) -> Tuple[int, int]:
        market = action // self._num_actions
        price_idx = action % self._num_actions
        return market, price_idx

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self._omega_idx = int(action)
            return
        active = self.current_player()
        market, price_idx = self._decode_action(action)
        p_from = self._market_prices[market][-1]
        p_to = self._price_grid[price_idx]
        # Potential profit IF this market resolves. We store payoff under
        # both possible outcomes of M | policy. Decision -> only one market
        # resolves, the other refunds the trader's cost. We track running
        # profit assuming market resolves; if it doesn't, we refund cost
        # at terminal time.
        # To keep state compact we store: payoff_if_M0 and payoff_if_M1
        # per trader per market.
        # On 2nd thought, simpler: store (shares_acquired, cost) per
        # trader per market; resolve at terminal time.
        shares = self._lmsr.shares_to_move(p_from, p_to)
        cost = self._lmsr.cost_to_move(p_from, p_to)
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
        return f"mkt={market_label},p={self._price_grid[price_idx]:.4f}"

    # ---- Payoffs ----

    def _winning_market(self) -> int:
        # Pick the policy whose conditional market closes at the higher
        # implied probability. Ties -> policy A.
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
            payout = shares if m == 1 else 0.0
            # Losing-market trades refund cost -> net zero on that side.
            out.append(payout - cost)
        if 0 <= self._manipulator_player <= 2 and self._manipulator_bonus != 0.0:
            preferred_market = 0 if self._manipulator_prefers_A == 1 else 1
            if winning == preferred_market:
                out[self._manipulator_player] += self._manipulator_bonus
        return out

    # ---- Information state ----

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

    # ---- Helpers ----

    def market_prices(self, market: int) -> List[float]:
        return list(self._market_prices[market])

    def final_market_prices(self) -> Tuple[float, float]:
        return self._market_prices[0][-1], self._market_prices[1][-1]

    def __str__(self) -> str:
        return (
            f"HansonConditional(omega={self._omega_idx}, "
            f"pA={['%.3f' % p for p in self._market_prices[0]]}, "
            f"pB={['%.3f' % p for p in self._market_prices[1]]})"
        )


class HansonObserver:
    def __init__(self, game: HansonConditionalGame):
        self._game = game
        self.tensor = np.zeros(1, dtype=np.float32)
        self.dict = {"placeholder": self.tensor}

    def set_from(self, state: HansonConditionalState, player: int) -> None:
        self.tensor.fill(0)

    def string_from(self, state: HansonConditionalState, player: int) -> str:
        return state.information_state_string(player)


def register() -> None:
    try:
        pyspiel.register_game(_GAME_TYPE, HansonConditionalGame)
    except pyspiel.SpielError:
        pass


register()


__all__ = [
    "HansonConditionalGame",
    "HansonConditionalState",
    "HansonObserver",
    "register",
    "STATES",
    "STATE_LABELS",
    "metric_under_policy",
]
