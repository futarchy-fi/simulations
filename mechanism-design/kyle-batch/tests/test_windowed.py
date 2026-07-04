"""Windowed-TWAP extension: statistic = mean of the last K batch prices,
fast push-response solver, concealed-window mixture."""

import numpy as np
import pytest

from kyle_batch.twap import (TwapParams, solve_honest_dynamics, evaluate,
                             solve_manipulator, solve_manipulator_fast,
                             push_response, stat_weights, evaluate_mixture,
                             mc_check)

P4 = TwapParams(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, T=4)


@pytest.fixture(scope="module")
def dyn4():
    return solve_honest_dynamics(P4)


@pytest.fixture(scope="module")
def pr4(dyn4):
    return push_response(dyn4)


def test_window_endpoints_reduce_to_last_and_twap(dyn4):
    """win:1 == last, win:T == twap, exactly, for arbitrary pushes."""
    al = np.array([0.2, -0.1, 0.05, 0.3])
    for a, b in (("win:1", "last"), ("win:4", "twap")):
        ea, eb = evaluate(dyn4, al, a), evaluate(dyn4, al, b)
        for k in ("stat_bias", "stat_sd", "corr_Pv", "decision_quality",
                  "approval_prob"):
            assert ea[k] == pytest.approx(eb[k], rel=1e-12)


def test_stat_weights():
    assert np.allclose(stat_weights(4, "win:2"), [0, 0, .5, .5])
    assert np.allclose(stat_weights(4, "last"), [0, 0, 0, 1])
    assert np.allclose(stat_weights(4, "twap"), [.25] * 4)


def test_push_response_matches_evaluate(dyn4, pr4):
    """bias = D@alpha, trading pnl = -alpha'D alpha, moments alpha-free —
    the linear structure underlying the fast solver."""
    al = np.array([0.3, 0.1, -0.2, 0.4])
    for stat in ("last", "win:2", "twap"):
        ev = evaluate(dyn4, al, stat)
        w = stat_weights(4, stat)
        assert ev["price_biases"] == pytest.approx(list(pr4["D"] @ al), rel=1e-10)
        assert ev["stat_bias"] == pytest.approx(float(w @ pr4["D"] @ al), rel=1e-10)
        assert ev["manip_trading_pnl"] == pytest.approx(
            -float(al @ pr4["D"] @ al), rel=1e-10)
        sd = np.sqrt(w @ pr4["cov_pp"] @ w)
        assert ev["stat_sd"] == pytest.approx(float(sd), rel=1e-10)


def test_fast_solver_matches_nelder_mead(dyn4, pr4):
    """BFGS + exact gradient vs the original Nelder-Mead full-evaluate solver."""
    for stat in ("last", "twap", "win:2"):
        a_fast = solve_manipulator_fast(pr4, B=2.0, statistics=[stat])
        a_nm = solve_manipulator(dyn4, B=2.0, statistic=stat)
        assert a_fast == pytest.approx(a_nm, abs=2e-5)
        # fast solution weakly better on the exact objective
        u_fast = evaluate(dyn4, a_fast, stat)
        u_nm = evaluate(dyn4, a_nm, stat)
        tot_f = u_fast["manip_trading_pnl"] + 2.0 * u_fast["approval_prob"]
        tot_n = u_nm["manip_trading_pnl"] + 2.0 * u_nm["approval_prob"]
        assert tot_f >= tot_n - 1e-10


def test_fast_solver_local_optimality(pr4, dyn4):
    """Unilateral-deviation check: random perturbations of the optimal push
    never gain (exact objective, 200 directions)."""
    rng = np.random.default_rng(7)
    B = 5.0
    for stat in ("last", "win:2"):
        al = solve_manipulator_fast(pr4, B=B, statistics=[stat])
        ev0 = evaluate(dyn4, al, stat)
        u0 = ev0["manip_trading_pnl"] + B * ev0["approval_prob"]
        for scale in (1e-3, 1e-2, 1e-1):
            for _ in range(200):
                d = rng.standard_normal(4) * scale
                ev = evaluate(dyn4, al + d, stat)
                u = ev["manip_trading_pnl"] + B * ev["approval_prob"]
                assert u <= u0 + 1e-12


def test_window_mc_agreement(dyn4):
    """MC verification of the affine propagation for a windowed statistic."""
    al = np.array([0.2, 0.1, 0.0, 0.3])
    ana = evaluate(dyn4, al, "win:2")
    mc = mc_check(dyn4, al, "win:2", n=600_000, seed=11)
    assert mc["stat_bias"] == pytest.approx(ana["stat_bias"], abs=3e-3)
    assert mc["decision_quality"] == pytest.approx(
        ana["decision_quality"], abs=4 * mc["decision_quality_se"])
    assert mc["corr_Pv"] == pytest.approx(ana["corr_Pv"], abs=3e-3)


def test_concealed_mixture_consistency(dyn4, pr4):
    """Concealed-K: mixture DQ/approval = probability-weighted per-K values;
    the concealed best response does weakly worse against the mixture than
    tailored per-K best responses do against their own K."""
    B = 2.0
    stats = ["win:1", "win:2", "win:4"]
    al_c = solve_manipulator_fast(pr4, B=B, statistics=stats)
    mix = evaluate_mixture(dyn4, al_c, stats, B=B)
    dq_manual = np.mean([mix["per_statistic"][s]["decision_quality"] for s in stats])
    assert mix["decision_quality_mix"] == pytest.approx(dq_manual, rel=1e-12)
    # manipulator value: concealed <= mean of known-K values (info has value)
    u_conc = mix["manip_total"]
    u_known = 0.0
    for s in stats:
        a_k = solve_manipulator_fast(pr4, B=B, statistics=[s])
        ev = evaluate(dyn4, a_k, s)
        u_known += (ev["manip_trading_pnl"] + B * ev["approval_prob"]) / len(stats)
    assert u_conc <= u_known + 1e-10


def test_concealed_local_optimality(dyn4, pr4):
    """Deviation test for the concealed-K push (exact mixture objective)."""
    rng = np.random.default_rng(3)
    B = 5.0
    stats = ["win:1", "win:2", "win:4"]
    al = solve_manipulator_fast(pr4, B=B, statistics=stats)
    u0 = evaluate_mixture(dyn4, al, stats, B=B)["manip_total"]
    for scale in (1e-3, 1e-2, 1e-1):
        for _ in range(200):
            d = rng.standard_normal(4) * scale
            u = evaluate_mixture(dyn4, al + d, stats, B=B)["manip_total"]
            assert u <= u0 + 1e-12
