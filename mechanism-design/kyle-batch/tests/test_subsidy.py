"""Q6 subsidy-comparison machinery: sizing identities and equilibrium checks."""

import numpy as np
import pytest

from kyle_batch.closed_forms import baseline
from kyle_batch.decision import logistic_q
from kyle_batch.onebatch import Params, solve_equilibrium, metrics

N, SE, TAU = 3, 1.0, 0.3
RHO0 = 1e-12
LN2 = float(np.log(2.0))


def test_noise_sizing_identity():
    """sigma_u = S / lambda0(1) delivers expected noise-flow loss = S exactly:
    the covert equilibrium scales with sigma_u, so lambda*sigma_u^2 is linear."""
    lam0 = baseline(N, SE, 1.0).lam
    for S in (0.2, 0.8, 3.2):
        su = S / lam0
        p = Params(N=N, sigma_eps=SE, sigma_u=su, tau=TAU, B=0.0,
                   manip="informed", rho=RHO0)
        prof = solve_equilibrium(p)
        assert prof.lam * su**2 == pytest.approx(S, rel=1e-8)
        # noise flow's expected loss equals the subsidy (MM belief-based lam)
        m = metrics(prof, p, True)
        assert -m["profit_noise"] == pytest.approx(S, rel=1e-8)


def test_noise_baseline_dq_sigma_u_invariant():
    """The covert-informed baseline DQ is sigma_u-invariant (whole equilibrium
    scales in sigma_u units) -- the noise instrument's zero-aggregation-cost
    property, extended from the honest-only Q1 result."""
    dqs = []
    for su in (0.25, 1.0, 4.0):
        p = Params(N=N, sigma_eps=SE, sigma_u=su, tau=TAU, B=0.0,
                   manip="informed", rho=RHO0)
        prof = solve_equilibrium(p)
        dqs.append(metrics(prof, p, True)["decision_quality"])
    assert dqs[0] == pytest.approx(dqs[1], rel=1e-9)
    assert dqs[2] == pytest.approx(dqs[1], rel=1e-9)


def test_lmsr_logit_bridge_exact():
    """LMSR price pi(Q) = logistic(Q/b) read in v-units through the logistic
    decision rule (p = tau*logit(pi)) is EXACTLY the linear curve p=(tau/b)Q."""
    b = 1.7
    Q = np.linspace(-8, 8, 41)
    pi = logistic_q(Q / b * TAU, TAU)          # logistic(Q/b)
    p_v = TAU * np.log(pi / (1 - pi))          # tau * logit(pi)
    assert p_v == pytest.approx((TAU / b) * Q, rel=1e-10)


def test_amm_sizing_and_breakeven_point():
    """kappa = tau*ln2/S; at S* = tau*ln2/lambda*(N+1) the curve equals the
    Kyle-equilibrium impact and (with aware honest traders) the maker breaks
    even at B=0 -- reproducing the Q5 anchor from the budget parametrization."""
    lam_star = baseline(N + 1, SE, 1.0).lam
    S_star = TAU * LN2 / lam_star
    kappa = TAU * LN2 / S_star
    assert kappa == pytest.approx(lam_star, rel=1e-12)
    p0 = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=0.0,
                manip="informed", rho=1.0, mm="fixed", kappa=kappa)
    prof = solve_equilibrium(p0)
    m = metrics(prof, p0, True)
    assert abs(m["profit_mm"]) < 1e-8
    assert abs(m["price_bias"]) < 1e-9
    # worst-case maker loss of the bridged LMSR = b*ln2 = S by construction
    assert (S_star / LN2) * LN2 == pytest.approx(S_star)


def test_amm_deep_curve_subsidizes_and_resists():
    """Deeper curve (larger S, smaller kappa): maker's expected loss grows,
    bias at B=2 shrinks -- the resistance-per-dollar direction of Q6."""
    rows = []
    for S in (0.4, 1.6):
        kappa = TAU * LN2 / S
        pB = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=2.0,
                    manip="informed", rho=RHO0, mm="fixed", kappa=kappa)
        p0 = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=0.0,
                    manip="informed", rho=RHO0, mm="fixed", kappa=kappa)
        profB, prof0 = solve_equilibrium(pB), solve_equilibrium(p0)
        mB, m0 = metrics(profB, pB, True), metrics(prof0, p0, True)
        rows.append({"S": S, "bias": mB["price_bias"],
                     "maker_loss0": -m0["profit_mm"],
                     "damage": m0["decision_quality"] - mB["decision_quality"]})
    assert rows[1]["bias"] < rows[0]["bias"]
    assert rows[1]["damage"] < rows[0]["damage"]
    assert rows[1]["maker_loss0"] > rows[0]["maker_loss0"]
