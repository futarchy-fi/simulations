"""Q4 (TWAP dynamics) and Q5 (fixed-impact AMM) checks."""

import numpy as np
import pytest

from kyle_batch.closed_forms import baseline
from kyle_batch.onebatch import Params, solve_equilibrium, metrics
from kyle_batch.twap import (TwapParams, solve_honest_dynamics, evaluate,
                             solve_manipulator, mc_check)


def test_twap_T1_reduces_to_static():
    """T = 1 must reproduce the static one-shot baseline."""
    p = TwapParams(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, T=1)
    dyn = solve_honest_dynamics(p)
    ref = baseline(3, 1.0, 1.0)
    assert dyn["betas"][0] == pytest.approx(ref.beta, rel=1e-10)
    assert dyn["lams"][0] == pytest.approx(ref.lam, rel=1e-10)
    ev = evaluate(dyn, np.zeros(1), "twap")
    assert ev["corr_Pv"] == pytest.approx(ref.corr_pv, rel=1e-10)


def test_twap_affine_matches_mc():
    p = TwapParams(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, T=4)
    dyn = solve_honest_dynamics(p)
    al = np.array([0.2, 0.1, 0.0, 0.3])
    for stat in ("twap", "last"):
        ana = evaluate(dyn, al, stat)
        mc = mc_check(dyn, al, stat, n=600_000, seed=5)
        assert mc["stat_bias"] == pytest.approx(ana["stat_bias"], abs=3e-3)
        assert mc["decision_quality"] == pytest.approx(
            ana["decision_quality"], abs=4 * mc["decision_quality_se"])
        assert mc["corr_Pv"] == pytest.approx(ana["corr_Pv"], abs=3e-3)


def test_twap_last_price_more_informative_at_baseline():
    """With info revealed over rounds, the last price dominates the TWAP in
    baseline informativeness (early prices are less informed)."""
    p = TwapParams(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, T=4)
    dyn = solve_honest_dynamics(p)
    evt = evaluate(dyn, np.zeros(4), "twap")
    evl = evaluate(dyn, np.zeros(4), "last")
    assert evl["corr_Pv"] > evt["corr_Pv"]
    assert evl["decision_quality"] > evt["decision_quality"]


def test_twap_manipulator_timing():
    """Optimal pushes concentrate late under the last-price rule and early
    under TWAP (early pushes decay through honest correction but enter more
    average terms)."""
    p = TwapParams(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, T=4)
    dyn = solve_honest_dynamics(p)
    a_twap = solve_manipulator(dyn, B=2.0, statistic="twap")
    a_last = solve_manipulator(dyn, B=2.0, statistic="last")
    assert a_last[-1] > a_last[:-1].max()
    assert a_twap[0] > a_twap[-1]


def test_amm_baseline_and_pushback():
    """kappa = Kyle lambda reproduces the Kyle baseline at B = 0; with a
    bounty, honest traders absorb a_h = -a_m/(N+1) of the push and honest
    profits RISE (Hanson subsidy transfer works under a non-updating curve)."""
    ref = baseline(4, 1.0, 1.0)
    p0 = Params(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, B=0.0,
                manip="informed", rho=1.0, mm="fixed", kappa=ref.lam)
    pB = Params(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, B=2.0,
                manip="informed", rho=1.0, mm="fixed", kappa=ref.lam)
    prof0, profB = solve_equilibrium(p0), solve_equilibrium(pB)
    m0, mB = metrics(prof0, p0, True), metrics(profB, pB, True)
    assert prof0.b_h == pytest.approx(ref.beta, abs=1e-8)
    assert abs(m0["price_bias"]) < 1e-9
    assert abs(m0["profit_mm"]) < 1e-9
    assert profB.a_h == pytest.approx(-profB.a_m / (p0.N + 1), rel=1e-6)
    assert mB["price_bias"] > 0
    assert mB["profit_honest_each"] > m0["profit_honest_each"]
    assert mB["manip_trading_pnl"] < m0["manip_trading_pnl"]
