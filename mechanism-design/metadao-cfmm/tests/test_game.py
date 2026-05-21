"""Sanity tests for the MetaDAO CFMM pyspiel game."""

import pytest
import pyspiel

from metadao_cfmm.game import MetaDAOGame, NUM_POLICIES


@pytest.fixture
def small_game() -> MetaDAOGame:
    return MetaDAOGame({"num_rounds": 3, "num_actions": 5, "K": 0.001})


def test_game_shape(small_game: MetaDAOGame) -> None:
    assert small_game.num_players() == 3
    assert small_game.num_distinct_actions() == NUM_POLICIES * 5
    assert small_game.max_chance_outcomes() == 8


def test_chance_first(small_game: MetaDAOGame) -> None:
    state = small_game.new_initial_state()
    assert state.is_chance_node()
    outcomes = state.chance_outcomes()
    assert len(outcomes) == 8
    assert all(prob == pytest.approx(1.0 / 8.0) for _, prob in outcomes)


def test_three_round_terminal(small_game: MetaDAOGame) -> None:
    state = small_game.new_initial_state()
    state.apply_action(0)
    state.apply_action(2)
    state.apply_action(2)
    state.apply_action(2)
    assert state.is_terminal()


def test_no_movement_returns_zero(small_game: MetaDAOGame) -> None:
    state = small_game.new_initial_state()
    state.apply_action(0)
    state.apply_action(2)  # idx 2 = grid point 3/6 = 0.5 (no move)
    state.apply_action(2)
    state.apply_action(2)
    assert state.returns() == [0.0, 0.0, 0.0]


def test_pyspiel_load() -> None:
    g = pyspiel.load_game(
        "python_metadao_cfmm",
        {"num_rounds": 3, "num_actions": 5, "K": 0.001},
    )
    assert g.num_players() == 3
