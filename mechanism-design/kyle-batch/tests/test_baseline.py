"""Q1: baseline closed form vs SymPy, vs fixed-point solver, vs Monte Carlo."""

import numpy as np
import pytest

from kyle_batch.closed_forms import baseline, sympy_derivation
from kyle_batch.onebatch import Params, solve_equilibrium, metrics
from kyle_batch.mc import simulate, deviation_report


def test_kyle_1985_single_trader_limit():
    """N=1, sigma_eps=0 must reproduce Kyle (1985): lam = 1/(2 su), beta = su."""
    for su in (0.5, 1.0, 2.0):
        b = baseline(1, 0.0, su)
        assert b.lam == pytest.approx(1.0 / (2 * su), rel=1e-12)
        assert b.beta == pytest.approx(su, rel=1e-12)
        assert b.corr_pv == pytest.approx(np.sqrt(0.5), rel=1e-12)


def test_sympy_rederivation():
    sympy_derivation(N_val=3, se2_val=0.5)
    sympy_derivation(N_val=5, se2_val=2.0)


def test_solver_matches_closed_form():
    for (N, se, su) in [(1, 0.0, 1.0), (3, 1.0, 1.0), (5, 0.5, 2.0), (2, 2.0, 0.3)]:
        ref = baseline(N, se, su)
        prof = solve_equilibrium(Params(N=N, sigma_eps=se, sigma_u=su, manip="none"))
        assert prof.b_h == pytest.approx(ref.beta, rel=1e-8)
        assert prof.lam == pytest.approx(ref.lam, rel=1e-8)
        assert abs(prof.a_h) < 1e-9
        assert abs(prof.mu) < 1e-9


def test_price_distribution_invariant_to_sigma_u():
    """Var(p) = Cov(v,p) = cN independent of sigma_u."""
    b1 = baseline(4, 1.0, 0.2)
    b2 = baseline(4, 1.0, 5.0)
    assert b1.var_p == pytest.approx(b2.var_p, rel=1e-12)
    assert b1.corr_pv == pytest.approx(b2.corr_pv, rel=1e-12)
    # but profits scale with sigma_u
    assert b2.informed_profit_total == pytest.approx(
        b1.informed_profit_total * 5.0 / 0.2, rel=1e-12)


def test_monte_carlo_agreement():
    p = Params(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, manip="none")
    prof = solve_equilibrium(p)
    ana = metrics(prof, p, present=False)
    mc = simulate(prof, p, present=False, n=2_000_000, seed=7)
    assert mc["corr_pv"] == pytest.approx(ana["corr_pv"], abs=2e-3)
    assert mc["var_p"] == pytest.approx(ana["var_p"], abs=5e-3)
    assert mc["decision_quality"] == pytest.approx(
        ana["decision_quality"], abs=4 * mc["decision_quality_se"])
    assert mc["profit_honest_each"] == pytest.approx(
        ana["profit_honest_each"], abs=4 * mc["profit_honest_each_se"])
    # MM zero profit at equilibrium
    assert abs(ana["profit_mm"]) < 1e-9


def test_baseline_deviation_certificate():
    p = Params(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, manip="none")
    prof = solve_equilibrium(p)
    rep = deviation_report(prof, p, n=200_000)
    # sup gain (exact) essentially zero; grid MC gain within noise
    assert rep["honest_sup_gain"] < 1e-9
    assert rep["honest_grid"]["max_gain"] < 4 * rep["honest_grid"]["se"] + 1e-6
