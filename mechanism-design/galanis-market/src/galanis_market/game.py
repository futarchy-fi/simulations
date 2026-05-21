"""OpenSpiel implementation of the Galanis (2026) prediction-market game.

The game is sequential and imperfect-information:

  1. Nature draws a state omega uniformly from the 8 possible signal
     triples (d_a, d_b, d_c) in {0,1}^3.
  2. Players 0, 1, 2 take turns in fixed rotation for `num_rounds` rounds.
     With num_rounds = 9, each player acts three times.
  3. Each player observes only their partition cell of omega plus the
     full public price history.
  4. On every turn, the active player picks a target price from a finite
     grid. The market maker (LMSR with liquidity `b`) takes the other
     side of the trade and the price snaps to the chosen target.
  5. At the terminal node the security pays X(omega) in {0, 1} and each
     player's utility is the sum of their per-trade LMSR profits.

The discretisation lets us run tabular CFR/CFR+ directly. Set
`num_actions` and `num_rounds` carefully -- info-state count grows like
`cells_per_trader * num_actions ** (moves_before_player_acts_again)`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyspiel

from galanis_market.lmsr import LMSR
from galanis_market.structures import STATES, STRUCTURES, Structure


def _default_price_grid(num_actions: int) -> List[float]:
    """Equally spaced grid on (0, 1), excluding the singular endpoints.

    For num_actions = 9 -> {0.1, 0.2, ..., 0.9}.
    For num_actions = 19 -> {0.05, 0.10, ..., 0.95}.
    """
    if num_actions < 2:
        raise ValueError("num_actions must be at least 2")
    step = 1.0 / (num_actions + 1)
    return [round((i + 1) * step, 6) for i in range(num_actions)]


def _build_game_type() -> pyspiel.GameType:
    return pyspiel.GameType(
        short_name="python_galanis_market",
        long_name="Python Galanis Market",
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
            "structure": "t3s111y2",
            "num_rounds": 3,
            "num_actions": 9,
            "b": 0.01,
            "initial_price": 0.5,
        },
    )


_DEFAULT_GAME_TYPE = _build_game_type()


def _build_game_info(num_actions: int, num_rounds: int) -> pyspiel.GameInfo:
    # min/max utility bounds: a trader's worst case is losing the full
    # b * log 2 maximum subsidy per trade times their number of trades;
    # we use a loose but safe bound of `num_rounds * 1.0`.
    bound = float(num_rounds)
    return pyspiel.GameInfo(
        num_distinct_actions=num_actions,
        max_chance_outcomes=len(STATES),
        num_players=3,
        min_utility=-bound,
        max_utility=bound,
        utility_sum=None,  # general sum
        max_game_length=num_rounds + 1,  # +1 for the chance node
    )


class GalanisMarketGame(pyspiel.Game):
    """pyspiel.Game wrapper for the Galanis prediction-market game."""

    def __init__(self, params: Optional[dict] = None):
        params = dict(params or {})
        self.structure_name: str = params.get("structure", "t3s111y2")
        self.num_rounds: int = int(params.get("num_rounds", 3))
        self.num_actions: int = int(params.get("num_actions", 9))
        self.b: float = float(params.get("b", 0.01))
        self.initial_price: float = float(params.get("initial_price", 0.5))

        if self.structure_name not in STRUCTURES:
            raise ValueError(
                f"Unknown structure {self.structure_name}; "
                f"choose from {list(STRUCTURES)}"
            )
        if self.num_rounds not in (3, 6, 9):
            raise ValueError("num_rounds must be 3, 6, or 9")
        if not 0.0 < self.initial_price < 1.0:
            raise ValueError("initial_price must lie strictly in (0, 1)")

        self.structure: Structure = STRUCTURES[self.structure_name]
        self.price_grid: List[float] = _default_price_grid(self.num_actions)
        self.lmsr: LMSR = LMSR(b=self.b)

        game_type = _build_game_type()
        game_info = _build_game_info(self.num_actions, self.num_rounds)
        super().__init__(game_type, game_info, params)

    def new_initial_state(self) -> "GalanisMarketState":
        return GalanisMarketState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return GalanisObserver(self)


class GalanisMarketState(pyspiel.State):
    """Sequential, imperfect-information state for the Galanis market.

    We copy all game configuration into the state at construction (rather
    than keeping a reference to the Game object) so that C++-side state
    cloning preserves everything we need to run is_terminal / returns /
    legal_actions without round-tripping through the Python game wrapper.
    """

    def __init__(self, game: GalanisMarketGame):
        super().__init__(game)
        # Snapshot of game configuration.
        self._structure = game.structure
        self._num_rounds = game.num_rounds
        self._num_actions = game.num_actions
        self._price_grid = list(game.price_grid)
        self._lmsr = game.lmsr
        # Mutable state.
        self._omega_idx: Optional[int] = None
        self._price_history: List[float] = [game.initial_price]
        self._action_history: List[int] = []
        self._trader_profit: List[float] = [0.0, 0.0, 0.0]

    # ---- Sequencing -----------------------------------------------------

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

    # ---- Chance ---------------------------------------------------------

    def chance_outcomes(self):
        assert self.is_chance_node()
        prob = 1.0 / len(STATES)
        return [(i, prob) for i in range(len(STATES))]

    # ---- Actions --------------------------------------------------------

    def _legal_actions(self, player: int) -> List[int]:
        assert player >= 0
        return list(range(self._num_actions))

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self._omega_idx = int(action)
            return

        active = self.current_player()
        p_from = self._price_history[-1]
        p_to = self._price_grid[action]
        outcome_yes = bool(
            self._structure.x_of(STATES[self._omega_idx])
        )
        self._trader_profit[active] += self._lmsr.trade_payoff(
            p_from, p_to, outcome_yes
        )
        self._price_history.append(p_to)
        self._action_history.append(action)

    def _action_to_string(self, player: int, action: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            return f"omega={action}"
        return f"p_target={self._price_grid[action]:.4f}"

    # ---- Payoffs --------------------------------------------------------

    def returns(self) -> List[float]:
        if not self.is_terminal():
            return [0.0, 0.0, 0.0]
        return list(self._trader_profit)

    # ---- Information state ---------------------------------------------

    def information_state_string(self, player: Optional[int] = None) -> str:
        if player is None:
            player = self.current_player()
        if self._omega_idx is None:
            cell = "?"
        else:
            cell = str(
                self._structure.cell_of(player, STATES[self._omega_idx])
            )
        public = ",".join(str(a) for a in self._action_history)
        return f"p{player}|cell={cell}|hist=[{public}]"

    def observation_string(self, player: Optional[int] = None) -> str:
        # For now identical to information_state_string; we don't expose
        # any additional public-only observation distinct from history.
        return self.information_state_string(player)

    # ---- Helpers --------------------------------------------------------

    @property
    def price_history(self) -> List[float]:
        return list(self._price_history)

    def final_price(self) -> float:
        return self._price_history[-1]

    def true_outcome(self) -> Optional[int]:
        if self._omega_idx is None:
            return None
        return self._game.structure.x_of(STATES[self._omega_idx])

    def __str__(self) -> str:
        omega = self._omega_idx
        return (
            f"GalanisMarket(omega={omega}, "
            f"prices={['%.3f' % p for p in self._price_history]}, "
            f"profits={['%.5f' % p for p in self._trader_profit]})"
        )


class GalanisObserver:
    """Minimal observer implementing the OpenSpiel observer interface.

    We do not expose a tensor (CFR is fine without one); we only support
    the string form. This is sufficient for CFR+, exploitability, and
    best-response computation.
    """

    def __init__(self, game: GalanisMarketGame):
        self._game = game
        # OpenSpiel checks for a `tensor` attribute on observers used in
        # tensor-consuming algorithms; we leave it as a single-element
        # placeholder so tensor-free algorithms ignore it gracefully.
        self.tensor = np.zeros(1, dtype=np.float32)
        self.dict = {"placeholder": self.tensor}

    def set_from(self, state: GalanisMarketState, player: int) -> None:
        self.tensor.fill(0)

    def string_from(self, state: GalanisMarketState, player: int) -> str:
        return state.information_state_string(player)


# Register with pyspiel so `pyspiel.load_game("python_galanis_market")` works.
def register() -> None:
    try:
        pyspiel.register_game(_DEFAULT_GAME_TYPE, GalanisMarketGame)
    except pyspiel.SpielError:
        # Already registered.
        pass


register()


__all__ = [
    "GalanisMarketGame",
    "GalanisMarketState",
    "GalanisObserver",
    "register",
]
