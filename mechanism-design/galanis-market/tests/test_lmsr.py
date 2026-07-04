"""Sanity tests for LMSR primitives."""

import math
import pytest

from galanis_market.lmsr import LMSR, logit, sigmoid


@pytest.fixture
def mm() -> LMSR:
    return LMSR(b=0.01)


def test_cost_at_neutral(mm: LMSR) -> None:
    # p = 0.5 -> C = -b log(0.5) = b log 2
    assert mm.cost_at_price(0.5) == pytest.approx(mm.b * math.log(2.0))


def test_path_independence(mm: LMSR) -> None:
    # Moving 0.5 -> 0.8 directly costs the same as 0.5 -> 0.6 -> 0.8.
    direct = mm.cost_to_move(0.5, 0.8)
    via = mm.cost_to_move(0.5, 0.6) + mm.cost_to_move(0.6, 0.8)
    assert direct == pytest.approx(via)


def test_shares_path_independence(mm: LMSR) -> None:
    # Share-count change is also path-independent (sum telescopes).
    direct = mm.shares_to_move(0.5, 0.8)
    via = mm.shares_to_move(0.5, 0.6) + mm.shares_to_move(0.6, 0.8)
    assert direct == pytest.approx(via)


def test_max_mm_loss_is_b_log_2(mm: LMSR) -> None:
    # If price ends at 1 and Yes wins, trader's profit -> b log 2.
    # We test the limit by going to p close to 1.
    p_close_to_1 = 1.0 - 1e-9
    profit = mm.trade_payoff(0.5, p_close_to_1, outcome_yes=True)
    # The exact closed-form for one trader moving 0.5 -> p with Yes win:
    #   shares = b * (logit(p) - logit(0.5)) = b * logit(p)
    #   cost   = -b * log(1 - p) + b * log(0.5)
    #   profit = shares - cost = b*logit(p) + b*log(1-p) - b*log(0.5)
    #          = b*(log(p) - log(1-p)) + b*log(1-p) - b*log(0.5)
    #          = b*log(p) - b*log(0.5) = b*log(2p)
    # at p=1: profit = b*log 2.
    assert profit == pytest.approx(mm.b * math.log(2.0), rel=1e-6)


def test_mm_breaks_even_if_no_movement(mm: LMSR) -> None:
    # If no one moves the price, MM loss = 0 regardless of outcome.
    for outcome in (True, False):
        assert mm.trade_payoff(0.5, 0.5, outcome) == 0.0


def test_buying_no_when_outcome_yes_loses(mm: LMSR) -> None:
    # Move 0.5 -> 0.3 (push toward No). If Yes wins, trader loses money.
    profit = mm.trade_payoff(0.5, 0.3, outcome_yes=True)
    assert profit < 0


def test_buying_no_when_outcome_no_wins(mm: LMSR) -> None:
    profit = mm.trade_payoff(0.5, 0.3, outcome_yes=False)
    assert profit > 0


def test_total_trader_profit_path_independent(mm: LMSR) -> None:
    # Two trajectories ending at the same price should give the SAME total
    # profit summed across traders (in our q_N=0 normalisation, total
    # shares-acquired is also path-independent, since each "shares to move"
    # telescopes).
    # Path A: 0.5 -> 0.7 -> 0.9
    a1 = mm.trade_payoff(0.5, 0.7, True)
    a2 = mm.trade_payoff(0.7, 0.9, True)
    # Path B: 0.5 -> 0.6 -> 0.9
    b1 = mm.trade_payoff(0.5, 0.6, True)
    b2 = mm.trade_payoff(0.6, 0.9, True)
    assert (a1 + a2) == pytest.approx(b1 + b2)


def test_logit_sigmoid_roundtrip() -> None:
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert sigmoid(logit(p)) == pytest.approx(p)
