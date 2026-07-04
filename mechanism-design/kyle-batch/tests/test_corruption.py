"""Q2/Q3: corruption equilibrium identities, neutralization theorem, T2u analog."""

import numpy as np
import pytest

from kyle_batch.closed_forms import baseline
from kyle_batch.decision import E_qprime
from kyle_batch.onebatch import Params, Profile, solve_equilibrium, metrics, bayes_mm_metrics
from kyle_batch.mc import simulate, deviation_report


BASE = dict(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3)


def test_known_manipulator_full_neutralization():
    """rho=1 (presence common knowledge), affine strategies, linear MM:
    the bounty is fully neutralized -- price bias 0, information weights and
    lambda equal the (N+1)-honest baseline, decision quality unchanged."""
    ref = baseline(4, 1.0, 1.0)
    p0 = Params(**BASE, B=0.0, manip="informed", rho=1.0)
    pB = Params(**BASE, B=2.0, manip="informed", rho=1.0)
    prof0, profB = solve_equilibrium(p0), solve_equilibrium(pB)
    m0, mB = metrics(prof0, p0, True), metrics(profB, pB, True)
    assert abs(mB["price_bias"]) < 1e-8
    assert profB.b_m == pytest.approx(ref.beta, abs=1e-6)
    assert profB.b_h == pytest.approx(ref.beta, abs=1e-6)
    assert profB.lam == pytest.approx(ref.lam, abs=1e-6)
    assert mB["decision_quality"] == pytest.approx(m0["decision_quality"], abs=1e-7)
    assert mB["approval_prob"] == pytest.approx(0.5, abs=1e-8)
    # manipulator pays nothing for the neutralized push
    assert mB["manip_trading_pnl"] == pytest.approx(m0["manip_trading_pnl"], abs=1e-7)


def test_manipulator_foc_identity():
    """a_m (2 - rho) = B E[q'(p)] with p at its present-state equilibrium law."""
    for rho in (1.0, 0.5, 1e-9):
        p = Params(**BASE, B=1.0, manip="informed", rho=rho)
        prof = solve_equilibrium(p)
        m1 = metrics(prof, p, True)
        eqp = E_qprime(m1["price_bias"], np.sqrt(m1["var_p"]), p.tau)
        assert prof.a_m * (2 - rho) == pytest.approx(p.B * eqp, rel=1e-5)


def test_covert_bias_identities():
    """bias_present = lam (1-rho) a_m ; bias_absent = -lam rho a_m."""
    p = Params(**BASE, B=1.0, manip="informed", rho=0.4)
    prof = solve_equilibrium(p)
    m1, m0 = metrics(prof, p, True), metrics(prof, p, False)
    assert m1["price_bias"] == pytest.approx(prof.lam * (1 - 0.4) * prof.a_m, rel=1e-8)
    assert m0["price_bias"] == pytest.approx(-prof.lam * 0.4 * prof.a_m, rel=1e-8)


def test_t2u_analog_symmetric_blur():
    """Type uncertainty (absent='honest'): bribed type keeps +bias, honest
    type pays the mirror-image discount; at rho=0.5 they are symmetric."""
    p = Params(N=2, sigma_eps=1.0, sigma_u=1.0, tau=0.3, B=1.0,
               manip="informed", rho=0.5, absent="honest")
    prof = solve_equilibrium(p)
    mB, mH = metrics(prof, p, True), metrics(prof, p, False)
    assert mB["price_bias"] > 0 > mH["price_bias"]
    assert mB["price_bias"] == pytest.approx(-mH["price_bias"], rel=1e-6)
    # bias identity: lam*(1-rho)*(a_m - a_e) / -lam*rho*(a_m - a_e)
    assert mB["price_bias"] == pytest.approx(
        prof.lam * 0.5 * (prof.a_m - prof.a_e), rel=1e-8)


def test_corrupted_equilibrium_vs_monte_carlo():
    p = Params(**BASE, B=1.0, manip="informed", rho=0.5)
    prof = solve_equilibrium(p)
    for present in (True, False):
        ana = metrics(prof, p, present)
        mc = simulate(prof, p, present, n=2_000_000, seed=11)
        assert mc["price_bias"] == pytest.approx(
            ana["price_bias"], abs=4 * mc["price_bias_se"])
        assert mc["decision_quality"] == pytest.approx(
            ana["decision_quality"], abs=4 * mc["decision_quality_se"])
        if present:
            assert mc["manip_trading_pnl"] == pytest.approx(
                ana["manip_trading_pnl"], abs=4 * mc["manip_trading_pnl_se"])


def test_deviation_certificate_corrupted():
    p = Params(**BASE, B=1.0, manip="informed", rho=1.0)
    prof = solve_equilibrium(p)
    rep = deviation_report(prof, p, n=200_000)
    assert rep["honest_sup_gain"] < 1e-9
    # affine-restriction error: nonlinear sup gain small relative to eq value
    assert rep["manip_sup"]["gain"] < 0.005 * abs(rep["manip_sup"]["eq_value"]) + 1e-3
    assert rep["manip_grid"]["max_gain"] < 4 * rep["manip_grid"]["se"] + 1e-6


def test_bayes_mm_reduces_to_linear_at_rho1():
    p = Params(**BASE, B=1.0, manip="informed", rho=1.0)
    prof = solve_equilibrium(p)
    lin = metrics(prof, p, True)
    bay = bayes_mm_metrics(prof, p, True)
    assert bay["price_bias"] == pytest.approx(lin["price_bias"], abs=1e-6)
    assert bay["decision_quality"] == pytest.approx(lin["decision_quality"], abs=1e-6)
