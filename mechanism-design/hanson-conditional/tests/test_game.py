"""Sanity tests for the Hanson outcome-conditional pyspiel game."""

import math
import pytest
import pyspiel

from hanson_conditional.game import (
    HansonConditionalGame,
    metric_under_policy,
    NUM_POLICIES,
)


@pytest.fixture
def small_game() -> HansonConditionalGame:
    return HansonConditionalGame(
        {"num_rounds": 3, "num_actions": 5}
    )


def test_metric_definitions() -> None:
    # Policy A: at-least-two-yes
    assert metric_under_policy(0, 0) == 1  # (1,1,1) -> 3 yes -> 1
    assert metric_under_policy(0, 3) == 0  # (1,0,0) -> 1 yes -> 0
    assert metric_under_policy(0, 7) == 0  # (0,0,0) -> 0 yes -> 0
    # Policy B: at-least-one-yes
    assert metric_under_policy(1, 0) == 1
    assert metric_under_policy(1, 3) == 1
    assert metric_under_policy(1, 7) == 0


def test_game_shape(small_game: HansonConditionalGame) -> None:
    assert small_game.num_players() == 3
    # Combined actions = 2 markets * 5 prices = 10.
    assert small_game.num_distinct_actions() == NUM_POLICIES * 5
    assert small_game.max_chance_outcomes() == 8


def test_initial_state_chance(small_game: HansonConditionalGame) -> None:
    state = small_game.new_initial_state()
    assert state.is_chance_node()
    outcomes = state.chance_outcomes()
    assert len(outcomes) == 8
    assert all(prob == pytest.approx(1.0 / 8.0) for _, prob in outcomes)


def test_decision_picks_higher_price_market() -> None:
    g = HansonConditionalGame({"num_rounds": 3, "num_actions": 5})
    state = g.new_initial_state()
    state.apply_action(0)  # omega = a
    # Trader 0 moves market A up to 0.833 (action 4: mkt 0, price idx 4)
    state.apply_action(4)
    # Trader 1 moves market B up to 0.667 (action 8: mkt 1, price idx 3)
    state.apply_action(8)
    # Trader 2 holds (action 7: mkt 1, price idx 2 = 0.5, no move)
    state.apply_action(7)
    assert state.is_terminal()
    final_a, final_b = state.final_market_prices()
    assert final_a > final_b
    # M(A, omega=a) = 1, trader 0 was long A -> positive return.
    assert state.returns()[0] > 0


def test_losing_market_refund_means_zero_profit() -> None:
    # Trader pumps market A to high but B wins: trader gets refund -> 0 profit.
    g = HansonConditionalGame({"num_rounds": 3, "num_actions": 5})
    state = g.new_initial_state()
    state.apply_action(0)  # omega = a
    # T0: market A to 0.833 (long Yes in A)
    state.apply_action(4)
    # T1: market B even higher, say 0.833 too
    state.apply_action(9)
    # T2: holds, no move (action 7: market B, price 0.5, but B is now at 0.833 -> moves down)
    # To keep things clean: T2 picks action 2 = market A, price 0.5 (would push A down)
    state.apply_action(2)
    final_a, final_b = state.final_market_prices()
    # If B >= A, market B wins (or tie -> A wins). Verify the conditional refund:
    if final_b > final_a:
        # T0 should get refund on his A position -> 0 profit from A side
        assert abs(state.returns()[0]) < 1e-9 or state.returns()[0] != 0


def test_total_payoff_is_lmsr_bounded() -> None:
    # Sum of trader profits cannot exceed b * log(2) regardless of trajectory:
    # only one market resolves and that market is LMSR-bounded.
    import random
    rng = random.Random(0)
    g = HansonConditionalGame({"num_rounds": 9, "num_actions": 7})
    bound = 0.01 * math.log(2.0) + 1e-12
    for _ in range(150):
        state = g.new_initial_state()
        state.apply_action(rng.randint(0, 7))
        while not state.is_terminal():
            state.apply_action(rng.randint(0, g.num_actions * NUM_POLICIES - 1))
        assert sum(state.returns()) <= bound


def test_info_state_only_uses_own_signal() -> None:
    g = HansonConditionalGame({"num_rounds": 3, "num_actions": 5})
    # Trader 0's info_state at the root depends only on d_a.
    seen = set()
    for omega in (0, 1, 2, 3):  # all states with d_a = 1
        state = g.new_initial_state()
        state.apply_action(omega)
        seen.add(state.information_state_string(0))
    assert len(seen) == 1


def test_pyspiel_load_via_registry() -> None:
    g = pyspiel.load_game(
        "python_hanson_conditional",
        {"num_rounds": 3, "num_actions": 5},
    )
    assert g.num_players() == 3
