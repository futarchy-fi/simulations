"""Sanity tests for the ProposalPoker tabular OpenSpiel game."""

import pytest
import pyspiel

from proposal_poker_tabular.game import (
    ProposalPokerGame, ACTION_YES, ACTION_NO, ACTION_ABSTAIN
)


@pytest.fixture
def game() -> ProposalPokerGame:
    return ProposalPokerGame({"signal_precision": 0.7})


def test_shape(game: ProposalPokerGame) -> None:
    assert game.num_players() == 3
    assert game.num_distinct_actions() == 3


def test_chance_sequence_then_players(game: ProposalPokerGame) -> None:
    state = game.new_initial_state()
    # 4 chance nodes: x, s0, s1, s2.
    for _ in range(4):
        assert state.is_chance_node()
        state.apply_action(1)
    assert state.current_player() == 0
    state.apply_action(ACTION_YES)
    assert state.current_player() == 1
    state.apply_action(ACTION_NO)
    assert state.current_player() == 2
    state.apply_action(ACTION_ABSTAIN)
    assert state.is_terminal()


def test_signal_precision_chance_outcomes(game: ProposalPokerGame) -> None:
    state = game.new_initial_state()
    state.apply_action(1)  # x = 1
    outcomes = state.chance_outcomes()
    # signal_precision = 0.7. Given x = 1, P(s_0 = 1) = 0.7.
    probs = dict(outcomes)
    assert probs[1] == pytest.approx(0.7)
    assert probs[0] == pytest.approx(0.3)


def test_yes_majority_approves() -> None:
    g = ProposalPokerGame({"signal_precision": 0.7})
    state = g.new_initial_state()
    state.apply_action(1)  # x
    for _ in range(3):
        state.apply_action(1)  # s_i
    state.apply_action(ACTION_YES)
    state.apply_action(ACTION_YES)
    state.apply_action(ACTION_NO)
    assert state.decision_approve() is True
    rets = state.returns()
    assert rets[0] > 0 and rets[1] > 0 and rets[2] < 0


def test_tie_breaks_to_reject() -> None:
    g = ProposalPokerGame({"signal_precision": 0.7})
    state = g.new_initial_state()
    state.apply_action(1)
    for _ in range(3):
        state.apply_action(1)
    state.apply_action(ACTION_YES)
    state.apply_action(ACTION_NO)
    state.apply_action(ACTION_ABSTAIN)
    assert state.decision_approve() is False
    # In a tie, stakes are refunded -> all returns are 0.
    assert state.returns() == [0.0, 0.0, 0.0]


def test_abstain_pays_zero() -> None:
    g = ProposalPokerGame({"signal_precision": 0.7})
    state = g.new_initial_state()
    state.apply_action(1)
    for _ in range(3):
        state.apply_action(1)
    state.apply_action(ACTION_YES)
    state.apply_action(ACTION_YES)
    state.apply_action(ACTION_ABSTAIN)
    rets = state.returns()
    assert rets[2] == 0.0  # abstainer earns/loses 0


def test_naive_player_only_abstains() -> None:
    g = ProposalPokerGame({"signal_precision": 0.7, "naive_player": 0})
    state = g.new_initial_state()
    state.apply_action(1)
    for _ in range(3):
        state.apply_action(1)
    assert state.legal_actions() == [ACTION_ABSTAIN]


def test_insider_sees_quality(game: ProposalPokerGame) -> None:
    g = ProposalPokerGame({"signal_precision": 0.7, "insider_player": 0})
    state = g.new_initial_state()
    state.apply_action(1)  # x = 1
    state.apply_action(0)  # s0 = 0 (wrong signal)
    state.apply_action(1)
    state.apply_action(1)
    # Insider's info-state should mention x=1, not s=0.
    info = state.information_state_string(0)
    assert "x=1" in info
    # Non-insider sees s only.
    info_p1 = state.information_state_string(1)
    assert "s=" in info_p1


def test_pyspiel_load_via_registry() -> None:
    g = pyspiel.load_game("python_proposal_poker_tabular",
                           {"signal_precision": 0.6})
    assert g.num_players() == 3
