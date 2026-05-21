"""Tests for the price-distribution utilities used by the solver."""

import pytest

from galanis_market.solve import _final_price_distribution, _weighted_median


def test_weighted_median_uniform() -> None:
    items = [(0.1, 1.0), (0.5, 1.0), (0.9, 1.0)]
    assert _weighted_median(items) == 0.5


def test_weighted_median_skewed() -> None:
    items = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.8)]
    # Cumulative: 0.1, 0.2, 1.0. Halfway is 0.5; cum reaches 0.5 at the
    # third entry, so weighted median = 0.9.
    assert _weighted_median(items) == 0.9


def test_weighted_median_two_modes() -> None:
    items = [(0.0, 0.5), (1.0, 0.5)]
    # Tie at the midpoint: implementation picks the first value where
    # cumulative weight >= total/2.
    assert _weighted_median(items) == 0.0


def test_weighted_median_handles_unsorted_input() -> None:
    items = [(0.9, 0.5), (0.1, 0.5)]
    assert _weighted_median(items) == 0.1


def test_weighted_median_empty() -> None:
    import math
    assert math.isnan(_weighted_median([]))


def test_final_price_distribution_terminal() -> None:
    from galanis_market.game import GalanisMarketGame
    from open_spiel.python.algorithms import cfr

    game = GalanisMarketGame(
        {"structure": "t3s111y2", "num_rounds": 3, "num_actions": 5}
    )
    solver = cfr.CFRPlusSolver(game)
    for _ in range(5):
        solver.evaluate_and_update_policy()
    policy = solver.average_policy()

    state = game.new_initial_state()
    state.apply_action(0)  # omega = a
    dist = _final_price_distribution(state, policy)
    # All probabilities must sum to 1.
    total = sum(w for _, w in dist)
    assert total == pytest.approx(1.0, abs=1e-9)
    # All final prices must lie in the action grid.
    grid = set(game.price_grid)
    for p, _ in dist:
        assert p in grid
