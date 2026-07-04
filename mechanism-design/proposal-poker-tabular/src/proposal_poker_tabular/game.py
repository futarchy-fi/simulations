"""Tabular OpenSpiel encoding of a minimal Proposal Poker game.

This is the simplest non-trivial slice of the model specified in
``mechanism-design/proposal-evaluation/MODEL.md``. We drop the
continuous (x, y) distribution, log-utility, and costly oracle in
favour of a finite, sequentially-acting binary staking market that
fits tabular CFR+.

Game flow:
1. Chance draws the true quality ``x in {0, 1}`` (uniform prior).
2. Chance draws each trader's private signal ``s_i in {0, 1}`` where
   ``Pr(s_i = x) = signal_precision`` (default 0.7), independent
   across traders.
3. Three traders act in fixed rotation. Each picks one of three
   actions: STAKE_YES (stake 1 unit on approval),
   STAKE_NO (stake 1 unit on rejection), or ABSTAIN (no stake).
4. After all traders act, the mechanism computes total stake on each
   side. The decision is APPROVE if YES stake > NO stake (ties break
   to REJECT by convention).
5. Payoffs:
   - Trader who staked on the WINNING side recovers their stake plus
     a proportional share of the losing-side pool.
   - Trader who staked on the LOSING side loses their stake.
   - Trader who abstained earns/loses 0.
   - Social welfare bonus: if the decision matches the true quality
     (approve iff x = 1), each player additionally receives
     ``welfare_bonus`` (default 0); for analysis we will compute this
     externally.

Manipulator and naive support symmetric to the other mechanisms.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyspiel


NUM_PLAYERS = 3
NUM_ACTIONS = 3  # STAKE_YES, STAKE_NO, ABSTAIN
ACTION_YES = 0
ACTION_NO = 1
ACTION_ABSTAIN = 2

_CHANCE_NODES = 1 + NUM_PLAYERS  # 1 quality draw + 3 signal draws


_GAME_TYPE = pyspiel.GameType(
    short_name="python_proposal_poker_tabular",
    long_name="Python Proposal Poker (tabular)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={
        "signal_precision": 0.7,
        "stake_amount": 1.0,
        "manipulator_player": -1,
        "manipulator_prefers_approve": 1,
        "manipulator_bonus": 0.0,
        "naive_player": -1,
        "insider_player": -1,
    },
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_ACTIONS,
    max_chance_outcomes=2,  # both x and s_i have 2 outcomes
    num_players=NUM_PLAYERS,
    min_utility=-2.0,
    max_utility=2.0,
    utility_sum=None,
    max_game_length=NUM_PLAYERS + _CHANCE_NODES,
)


class ProposalPokerGame(pyspiel.Game):
    def __init__(self, params: Optional[dict] = None):
        params = dict(params or {})
        self.signal_precision: float = float(params.get("signal_precision", 0.7))
        self.stake_amount: float = float(params.get("stake_amount", 1.0))
        self.manipulator_player: int = int(params.get("manipulator_player", -1))
        self.manipulator_prefers_approve: int = int(params.get("manipulator_prefers_approve", 1))
        self.manipulator_bonus: float = float(params.get("manipulator_bonus", 0.0))
        self.naive_player: int = int(params.get("naive_player", -1))
        self.insider_player: int = int(params.get("insider_player", -1))

        if not 0.5 <= self.signal_precision < 1.0:
            raise ValueError("signal_precision must lie in [0.5, 1.0)")
        if self.stake_amount <= 0:
            raise ValueError("stake_amount must be positive")

        super().__init__(_GAME_TYPE, _GAME_INFO, params)

    def new_initial_state(self) -> "ProposalPokerState":
        return ProposalPokerState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return ProposalPokerObserver(self)


class ProposalPokerState(pyspiel.State):
    def __init__(self, game: ProposalPokerGame):
        super().__init__(game)
        self._signal_precision = game.signal_precision
        self._stake = game.stake_amount
        self._manipulator_player = game.manipulator_player
        self._manipulator_prefers_approve = game.manipulator_prefers_approve
        self._manipulator_bonus = game.manipulator_bonus
        self._naive_player = game.naive_player
        self._insider_player = game.insider_player

        # _x is filled by the first chance node; _signals[i] by the
        # i+1-th chance node; _actions tracks player choices.
        self._x: Optional[int] = None
        self._signals: List[Optional[int]] = [None, None, None]
        self._actions: List[Optional[int]] = [None, None, None]

    def _num_chance_resolved(self) -> int:
        n = 0
        if self._x is not None:
            n += 1
        for s in self._signals:
            if s is not None:
                n += 1
        return n

    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._num_chance_resolved() < _CHANCE_NODES:
            return pyspiel.PlayerId.CHANCE
        # All chance resolved, players act in order.
        for i in range(NUM_PLAYERS):
            if self._actions[i] is None:
                return i
        return pyspiel.PlayerId.TERMINAL  # should not reach

    def is_terminal(self) -> bool:
        return (
            self._num_chance_resolved() == _CHANCE_NODES
            and all(a is not None for a in self._actions)
        )

    def chance_outcomes(self):
        assert self.is_chance_node()
        if self._x is None:
            # Quality x ~ Bernoulli(0.5)
            return [(0, 0.5), (1, 0.5)]
        # Signal for trader i where i = num signals resolved so far.
        # s_i = x with prob signal_precision, else 1 - x.
        for i in range(NUM_PLAYERS):
            if self._signals[i] is None:
                p = self._signal_precision
                if self._x == 1:
                    return [(0, 1 - p), (1, p)]
                else:
                    return [(0, p), (1, 1 - p)]
        raise RuntimeError("no chance to draw")

    def _legal_actions(self, player: int) -> List[int]:
        assert player >= 0
        if player == self._naive_player:
            # Naive: always abstain.
            return [ACTION_ABSTAIN]
        return [ACTION_YES, ACTION_NO, ACTION_ABSTAIN]

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            if self._x is None:
                self._x = int(action)
                return
            for i in range(NUM_PLAYERS):
                if self._signals[i] is None:
                    self._signals[i] = int(action)
                    return
            raise RuntimeError("unexpected chance action")
        active = self.current_player()
        self._actions[active] = int(action)

    def _action_to_string(self, player: int, action: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            if self._x is None:
                return f"x={action}"
            for i in range(NUM_PLAYERS):
                if self._signals[i] is None:
                    return f"s{i}={action}"
            return f"?={action}"
        return {0: "YES", 1: "NO", 2: "ABSTAIN"}.get(action, str(action))

    def returns(self) -> List[float]:
        if not self.is_terminal():
            return [0.0] * NUM_PLAYERS
        yes_stakers = [i for i, a in enumerate(self._actions) if a == ACTION_YES]
        no_stakers = [i for i, a in enumerate(self._actions) if a == ACTION_NO]
        yes_stake = self._stake * len(yes_stakers)
        no_stake = self._stake * len(no_stakers)
        approve = yes_stake > no_stake
        # Winner's pool = winner's own stake recovered + share of loser pool.
        out = [0.0] * NUM_PLAYERS
        if approve:
            winners, losers = yes_stakers, no_stakers
        elif no_stake > yes_stake:
            winners, losers = no_stakers, yes_stakers
        else:
            # Tie: stakes refunded, decision = REJECT.
            winners = []
            losers = []
            approve = False
        if winners:
            loser_pool = self._stake * len(losers)
            share = loser_pool / len(winners)
            for w in winners:
                out[w] = share  # net profit per winner
            for l_ in losers:
                out[l_] = -self._stake
        # Manipulator bonus: paid if implemented decision matches preference.
        if 0 <= self._manipulator_player <= 2 and self._manipulator_bonus != 0.0:
            wants_approve = self._manipulator_prefers_approve == 1
            if (approve and wants_approve) or (not approve and not wants_approve):
                out[self._manipulator_player] += self._manipulator_bonus
        return out

    def information_state_string(self, player: Optional[int] = None) -> str:
        if player is None:
            player = self.current_player()
        if player == self._insider_player and self._x is not None:
            own = f"x={self._x}"
        elif self._signals[player] is None:
            own = "s=?"
        else:
            own = f"s={self._signals[player]}"
        public = ",".join(
            "_" if a is None else {0: "Y", 1: "N", 2: "A"}[a]
            for a in self._actions
        )
        return f"p{player}|{own}|hist={public}"

    def observation_string(self, player: Optional[int] = None) -> str:
        return self.information_state_string(player)

    def decision_approve(self) -> Optional[bool]:
        if not self.is_terminal():
            return None
        yes = sum(1 for a in self._actions if a == ACTION_YES)
        no = sum(1 for a in self._actions if a == ACTION_NO)
        return yes > no

    def true_quality(self) -> Optional[int]:
        return self._x

    def __str__(self) -> str:
        return (
            f"ProposalPoker(x={self._x}, "
            f"signals={self._signals}, actions={self._actions})"
        )


class ProposalPokerObserver:
    def __init__(self, game: ProposalPokerGame):
        self._game = game
        self.tensor = np.zeros(1, dtype=np.float32)
        self.dict = {"placeholder": self.tensor}

    def set_from(self, state: ProposalPokerState, player: int) -> None:
        self.tensor.fill(0)

    def string_from(self, state: ProposalPokerState, player: int) -> str:
        return state.information_state_string(player)


def register() -> None:
    try:
        pyspiel.register_game(_GAME_TYPE, ProposalPokerGame)
    except pyspiel.SpielError:
        pass


register()


__all__ = [
    "ProposalPokerGame",
    "ProposalPokerState",
    "ProposalPokerObserver",
    "register",
    "ACTION_YES",
    "ACTION_NO",
    "ACTION_ABSTAIN",
]
