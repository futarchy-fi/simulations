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
        min_num_players=2,
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


def _build_game_info(
    num_actions: int, num_rounds: int, num_players: int = 3
) -> pyspiel.GameInfo:
    # min/max utility bounds: a trader's worst case is losing the full
    # b * log 2 maximum subsidy per trade times their number of trades;
    # we use a loose but safe bound of `num_rounds * 1.0`.
    bound = float(num_rounds)
    return pyspiel.GameInfo(
        num_distinct_actions=num_actions,
        max_chance_outcomes=len(STATES),
        num_players=num_players,
        min_utility=-bound,
        max_utility=bound,
        utility_sum=None,  # general sum
        max_game_length=num_rounds + 1,  # +1 for the chance node
    )


class GalanisMarketGame(pyspiel.Game):
    """pyspiel.Game wrapper for the Galanis prediction-market game.

    Optional manipulator support. If ``manipulator_player`` is set
    (default ``-1`` = none), that player's utility becomes

        u_manipulator = lmsr_profit
                      + manipulator_bonus * (final_price - 0.5)
                          (if manipulator_direction == +1, push price up)
                      - manipulator_bonus * (final_price - 0.5)
                          (if manipulator_direction == -1, push price down)

    Bayesian best-responders see the manipulator's actions and may adjust;
    CFR finds the equilibrium where the manipulator's strategy maximises
    its biased utility. ``manipulator_bonus`` is the per-unit price-shift
    payoff to the manipulator -- e.g., 1.0 means moving the final price
    by 0.1 yields a bias gain of 0.1 utility units, on top of any LMSR
    profit/loss incurred.
    """

    def __init__(self, params: Optional[dict] = None):
        params = dict(params or {})
        self.structure_name: str = params.get("structure", "t3s111y2")
        self.num_rounds: int = int(params.get("num_rounds", 3))
        self.num_actions: int = int(params.get("num_actions", 9))
        self.b: float = float(params.get("b", 0.01))
        self.initial_price: float = float(params.get("initial_price", 0.5))
        # Number of trading seats (2 or 3). The chance state is always a
        # full triple (d_a, d_b, d_c); with 2 players one bit may simply
        # be unobserved by everyone (see `signals`).
        self.n_players: int = int(params.get("num_players", 3))
        # Per-seat observation override, comma-separated, one entry per
        # player, each in {"a","b","c","none","all"}. Empty string
        # (default) falls back to the structure's own partition.
        # Example: "b,c,none" -- players 0/1 observe d_b/d_c, player 2
        # observes nothing (an uninformed entrant).
        signals_spec: str = str(params.get("signals", "")).strip()
        self.signals: Optional[List[str]] = (
            [s.strip() for s in signals_spec.split(",")] if signals_spec else None
        )
        # Decision statistic read from the market: "final" (the raw last
        # price) or "twap" (time-average of the post-trade prices). The
        # manipulator's price-target bonus is computed on this statistic,
        # so changing it changes payoffs and requires a fresh solve.
        self.decision_rule: str = str(params.get("decision_rule", "final"))
        self.manipulator_player: int = int(params.get("manipulator_player", -1))
        self.manipulator_direction: int = int(params.get("manipulator_direction", 1))
        self.manipulator_bonus: float = float(params.get("manipulator_bonus", 0.0))
        # Type uncertainty: with probability `manipulator_prob` the
        # designated player is actually bribed (receives the bonus);
        # with 1-p they are an ordinary honest trader. The realised type
        # is drawn by a second chance node after omega and is PRIVATE to
        # the designated player. p = 1.0 (default) recovers the
        # common-knowledge manipulator and adds no chance node.
        self.manipulator_prob: float = float(params.get("manipulator_prob", 1.0))
        # Player types: 'bayesian' (default, CFR best-response), 'naive'
        # (action restricted to play price = 0.5), 'insider' (info state
        # includes ALL signals -- god-mode trader who sees omega exactly).
        # Manipulator is configured via the manipulator_* params above.
        self.naive_player: int = int(params.get("naive_player", -1))
        self.insider_player: int = int(params.get("insider_player", -1))

        if self.structure_name not in STRUCTURES:
            raise ValueError(
                f"Unknown structure {self.structure_name}; "
                f"choose from {list(STRUCTURES)}"
            )
        if self.n_players not in (2, 3):
            raise ValueError("num_players must be 2 or 3")
        if self.decision_rule not in ("final", "twap"):
            raise ValueError("decision_rule must be 'final' or 'twap'")
        valid_rounds = (3, 6, 9) if self.n_players == 3 else (2, 4, 6)
        if self.num_rounds not in valid_rounds:
            raise ValueError(
                f"num_rounds must be one of {valid_rounds} for "
                f"{self.n_players} players"
            )
        if self.signals is not None:
            if len(self.signals) != self.n_players:
                raise ValueError(
                    "signals must have one entry per player "
                    f"({self.n_players}), got {self.signals}"
                )
            for s in self.signals:
                if s not in ("a", "b", "c", "none", "all"):
                    raise ValueError(f"invalid signal spec {s!r}")
        if not 0.0 < self.initial_price < 1.0:
            raise ValueError("initial_price must lie strictly in (0, 1)")
        seat_range = tuple(range(-1, self.n_players))
        if self.manipulator_player not in seat_range:
            raise ValueError(f"manipulator_player must be in {seat_range}")
        if self.manipulator_direction not in (-1, 1):
            raise ValueError("manipulator_direction must be -1 or +1")
        if not 0.0 < self.manipulator_prob <= 1.0:
            raise ValueError("manipulator_prob must lie in (0, 1]")
        if self.naive_player not in seat_range:
            raise ValueError(f"naive_player must be in {seat_range}")
        if self.insider_player not in seat_range:
            raise ValueError(f"insider_player must be in {seat_range}")

        self.structure: Structure = STRUCTURES[self.structure_name]
        self.price_grid: List[float] = _default_price_grid(self.num_actions)
        self.lmsr: LMSR = LMSR(b=self.b)

        game_type = _build_game_type()
        game_info = _build_game_info(
            self.num_actions, self.num_rounds, self.n_players
        )
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
        self._num_players = game.n_players
        self._signals = list(game.signals) if game.signals is not None else None
        self._decision_rule = game.decision_rule
        self._price_grid = list(game.price_grid)
        self._lmsr = game.lmsr
        self._manipulator_player = game.manipulator_player
        self._manipulator_direction = game.manipulator_direction
        self._manipulator_bonus = game.manipulator_bonus
        self._naive_player = game.naive_player
        self._insider_player = game.insider_player
        # Compute the action index whose target price is closest to 0.5;
        # this is the naive player's only legal action.
        self._naive_action = min(
            range(len(game.price_grid)),
            key=lambda i: abs(game.price_grid[i] - 0.5),
        )
        # Mutable state.
        self._omega_idx: Optional[int] = None
        self._price_history: List[float] = [game.initial_price]
        self._action_history: List[int] = []
        self._trader_profit: List[float] = [0.0] * self._num_players

    # ---- Sequencing -----------------------------------------------------

    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._omega_idx is None:
            return pyspiel.PlayerId.CHANCE
        return len(self._action_history) % self._num_players

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
        # Naive players have a degenerate action set: only "play 0.5".
        if player == self._naive_player:
            return [self._naive_action]
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
            return [0.0] * self._num_players
        out = list(self._trader_profit)
        if (
            0 <= self._manipulator_player < self._num_players
            and self._manipulator_bonus != 0.0
        ):
            shift = (self.decision_price() - 0.5) * self._manipulator_direction
            out[self._manipulator_player] += self._manipulator_bonus * shift
        return out

    # ---- Information state ---------------------------------------------

    def information_state_string(self, player: Optional[int] = None) -> str:
        if player is None:
            player = self.current_player()
        if self._omega_idx is None:
            cell = "?"
        elif player == self._insider_player:
            # Insider sees all three signals -- the full omega index.
            cell = f"omega{self._omega_idx}"
        elif self._signals is not None:
            spec = self._signals[player]
            if spec == "none":
                cell = "n"
            elif spec == "all":
                cell = f"omega{self._omega_idx}"
            else:
                bit = {"a": 0, "b": 1, "c": 2}[spec]
                cell = str(STATES[self._omega_idx][bit])
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

    def decision_price(self) -> float:
        """The statistic the decision rule (and any manipulator price
        bonus) reads: the raw final price, or the time-average of the
        post-trade prices (TWAP) if decision_rule == 'twap'."""
        if self._decision_rule == "twap":
            post = self._price_history[1:]
            if not post:
                return self._price_history[0]
            return sum(post) / len(post)
        return self._price_history[-1]

    def true_outcome(self) -> Optional[int]:
        if self._omega_idx is None:
            return None
        return self._structure.x_of(STATES[self._omega_idx])

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
