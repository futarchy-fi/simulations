"""Sanity tests for the GalanisMarketGame pyspiel wrapper."""

import math
import pytest
import pyspiel

from galanis_market.game import GalanisMarketGame
from galanis_market.lmsr import LMSR


# --------------------------------------------------------------------------- #
# Basic shape
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_game() -> GalanisMarketGame:
    return GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 3, "num_actions": 5}
    )


def test_game_shape(small_game: GalanisMarketGame) -> None:
    assert small_game.num_players() == 3
    assert small_game.num_distinct_actions() == 5
    assert small_game.max_chance_outcomes() == 8


def test_initial_state_is_chance(small_game: GalanisMarketGame) -> None:
    state = small_game.new_initial_state()
    assert state.is_chance_node()
    assert state.current_player() == pyspiel.PlayerId.CHANCE
    outcomes = state.chance_outcomes()
    assert len(outcomes) == 8
    assert all(prob == pytest.approx(1.0 / 8.0) for _, prob in outcomes)


def test_chance_then_three_players(small_game: GalanisMarketGame) -> None:
    state = small_game.new_initial_state()
    state.apply_action(0)
    assert state.current_player() == 0
    state.apply_action(2)
    assert state.current_player() == 1
    state.apply_action(2)
    assert state.current_player() == 2
    state.apply_action(2)
    assert state.is_terminal()


def test_rejects_invalid_structure() -> None:
    with pytest.raises(ValueError):
        GalanisMarketGame({"structure": "not_a_structure"})


def test_rejects_invalid_rounds() -> None:
    with pytest.raises(ValueError):
        GalanisMarketGame({"num_rounds": 4})


# --------------------------------------------------------------------------- #
# Pricing: returns must equal closed-form LMSR profits.
# --------------------------------------------------------------------------- #


def _play_to_terminal(game, omega: int, action_seq):
    state = game.new_initial_state()
    state.apply_action(omega)
    for a in action_seq:
        state.apply_action(a)
    return state


def test_returns_match_closed_form_yes_outcome() -> None:
    # Structure t3s111y2, omega = a = (1,1,1) -> X = 1 (Yes wins).
    # Use 5-action grid: {0.167, 0.333, 0.5, 0.667, 0.833}; b = 0.01.
    game = GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 3, "num_actions": 5}
    )
    grid = game.price_grid
    actions = [3, 4, 4]  # 0.5 -> 0.667 -> 0.833 -> 0.833 (no change last)
    state = _play_to_terminal(game, omega=0, action_seq=actions)
    assert state.is_terminal()
    mm = LMSR(b=0.01)
    p = [0.5] + [grid[a] for a in actions]
    expected = [
        mm.trade_payoff(p[i], p[i + 1], outcome_yes=True) for i in range(3)
    ]
    got = state.returns()
    assert got == pytest.approx(expected)
    # total trader profit equals b*log(2 * p_final) -- LMSR closed form.
    total_expected = mm.b * math.log(2 * p[-1])
    assert sum(got) == pytest.approx(total_expected, rel=1e-9)


def test_returns_zero_if_price_never_moves() -> None:
    game = GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 3, "num_actions": 5,
         "initial_price": 0.5}
    )
    # Pick the action that targets 0.5 exactly: idx 2 in a 5-grid.
    state = _play_to_terminal(game, omega=0, action_seq=[2, 2, 2])
    assert state.returns() == [0.0, 0.0, 0.0]


def test_total_payoff_bounded_by_lmsr_subsidy() -> None:
    # Sum of trader profits must not exceed b * log(2) -- the LMSR max
    # subsidy. Tested for many random playouts.
    import random
    rng = random.Random(0)
    game = GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 9, "num_actions": 9}
    )
    bound = 0.01 * math.log(2.0) + 1e-12
    for _ in range(200):
        state = game.new_initial_state()
        state.apply_action(rng.randint(0, 7))
        while not state.is_terminal():
            state.apply_action(rng.randint(0, 8))
        assert sum(state.returns()) <= bound


# --------------------------------------------------------------------------- #
# Information state: same private info + same public history -> same string.
# --------------------------------------------------------------------------- #


def test_information_state_invariant_under_omega_within_cell() -> None:
    # In t3s111y2 trader 0 only sees d_a. Omega indices 0..3 all have
    # d_a = 1, so trader 0's info_state at the root must coincide
    # regardless of which of those four omegas was drawn.
    game = GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 3, "num_actions": 5}
    )
    seen = set()
    for omega in (0, 1, 2, 3):  # all states with d_a = 1
        state = game.new_initial_state()
        state.apply_action(omega)
        seen.add(state.information_state_string(0))
    assert len(seen) == 1


def test_information_state_changes_across_cells() -> None:
    # Different cells -> different info-state strings for trader 0.
    game = GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 3, "num_actions": 5}
    )
    s_da1 = game.new_initial_state()
    s_da1.apply_action(0)  # d_a = 1
    s_da0 = game.new_initial_state()
    s_da0.apply_action(4)  # d_a = 0
    assert (
        s_da1.information_state_string(0)
        != s_da0.information_state_string(0)
    )


# --------------------------------------------------------------------------- #
# Loading via pyspiel.load_game also works (registered).
# --------------------------------------------------------------------------- #


def test_pyspiel_load_game_via_registry() -> None:
    g = pyspiel.load_game(
        "python_galanis_market",
        {"structure": "t3s111", "num_rounds": 3, "num_actions": 5},
    )
    assert g.num_players() == 3
