"""Tests for the binary CFMM primitives."""

import pytest

from metadao_cfmm.cfmm import BinaryCFMM


@pytest.fixture
def mm() -> BinaryCFMM:
    return BinaryCFMM(K=1.0)


def test_reserves_at_half(mm: BinaryCFMM) -> None:
    # At p = 0.5, y_yes = y_no = sqrt(K) = 1.
    y_yes, y_no = mm.reserves_at_price(0.5)
    assert y_yes == pytest.approx(1.0)
    assert y_no == pytest.approx(1.0)


def test_reserves_invariant(mm: BinaryCFMM) -> None:
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        y_yes, y_no = mm.reserves_at_price(p)
        assert y_yes * y_no == pytest.approx(1.0)
        # Marginal price check: p = y_no / (y_yes + y_no)
        assert y_no / (y_yes + y_no) == pytest.approx(p)


def test_cost_to_move_zero_if_no_move(mm: BinaryCFMM) -> None:
    for p in (0.2, 0.5, 0.8):
        assert mm.cost_to_move(p, p) == pytest.approx(0.0)


def test_cost_to_move_positive_when_pushing_up(mm: BinaryCFMM) -> None:
    # Pushing price from 0.5 -> 0.7 should cost positive No tokens.
    assert mm.cost_to_move(0.5, 0.7) > 0


def test_cost_to_move_negative_when_pushing_down(mm: BinaryCFMM) -> None:
    assert mm.cost_to_move(0.5, 0.3) < 0


def test_payoff_positive_for_correct_buy(mm: BinaryCFMM) -> None:
    # Bought Yes (price up), Yes wins: profit > 0.
    profit = mm.trade_payoff(0.5, 0.7, outcome_yes=True)
    assert profit > 0


def test_payoff_negative_for_wrong_buy(mm: BinaryCFMM) -> None:
    # Bought Yes (price up), No wins: profit < 0.
    profit = mm.trade_payoff(0.5, 0.7, outcome_yes=False)
    assert profit < 0


def test_cfmm_cost_grows_with_distance(mm: BinaryCFMM) -> None:
    # Cost to push price further should be larger. Both LMSR and CFMM
    # have unbounded cost near endpoints, but the growth rates differ:
    # LMSR is logarithmic (b * log((1-p_from)/(1-p_to))), CFMM is
    # power-law (sqrt(p/(1-p)) - 1). Both should be monotonically
    # increasing in p_to - p_from.
    c_to_0_7 = mm.cost_to_move(0.5, 0.7)
    c_to_0_9 = mm.cost_to_move(0.5, 0.9)
    c_to_0_99 = mm.cost_to_move(0.5, 0.99)
    assert 0 < c_to_0_7 < c_to_0_9 < c_to_0_99
    # The CFMM cost from 0.9 -> 0.99 is much larger than from 0.5 -> 0.7
    # despite the same delta-p, reflecting the power-law geometry.
    extra_cost = c_to_0_99 - c_to_0_9
    assert extra_cost > c_to_0_7
