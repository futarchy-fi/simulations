"""Q7 information-arrival extension: public signal stream + staggered signals.

Checks: exact reduction to the static Q4/Q4b model at phi=0 / all-at-batch-1,
the fixed total-precision parameterization, push linearity (D-matrix
identities), MC agreement for both variants, wash-out monotonicity, and
manipulator-solver optimality (stationarity + random-perturbation deviations).
"""

import numpy as np
import pytest

from kyle_batch import arrival as ar
from kyle_batch import twap as tw
from kyle_batch.twap import solve_manipulator_fast, stat_weights


@pytest.fixture(scope="module")
def dyn0():
    """phi = 0 public-stream dynamics (must equal the static model)."""
    return ar.solve_honest_dynamics_arrival(ar.ArrivalParams(T=4, phi=0.0))


@pytest.fixture(scope="module")
def dyn_half():
    return ar.solve_honest_dynamics_arrival(ar.ArrivalParams(T=4, phi=0.5))


@pytest.fixture(scope="module")
def pr_half(dyn_half):
    return ar.push_response(dyn_half)


def test_phi0_reduces_to_static_twap(dyn0):
    """phi=0 is EXACTLY the Q4/Q4b model: dynamics, D matrix, and full
    evaluate metrics agree with twap.py for an arbitrary push."""
    p_tw = tw.TwapParams(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3, T=4)
    dyn_tw = tw.solve_honest_dynamics(p_tw)
    assert dyn0["p"].sigma_eps == pytest.approx(1.0, rel=1e-14)
    assert dyn0["betas"] == pytest.approx(dyn_tw["betas"], rel=1e-12)
    assert dyn0["lams"] == pytest.approx(dyn_tw["lams"], rel=1e-12)
    pr_a, pr_t = ar.push_response(dyn0), tw.push_response(dyn_tw)
    assert pr_a["D"] == pytest.approx(pr_t["D"], abs=1e-12)
    assert pr_a["cov_pp"] == pytest.approx(pr_t["cov_pp"], abs=1e-12)
    al = np.array([0.2, -0.1, 0.05, 0.3])
    for stat in ("last", "win:2", "twap"):
        ea, et = ar.evaluate(dyn0, al, stat), tw.evaluate(dyn_tw, al, stat)
        for k in ("stat_bias", "stat_sd", "corr_Pv", "decision_quality",
                  "approval_prob", "manip_trading_pnl"):
            assert ea[k] == pytest.approx(et[k], rel=1e-10, abs=1e-12)


def test_total_precision_held_fixed():
    """Fundamental info budget: 1 + N/sig_eps^2 + (T-1)/sig_z^2 = 1 + Pi
    for every phi (the parameterization moves timing, not quantity)."""
    for T in (4, 8):
        for phi in (0.0, 0.25, 0.5, 0.75):
            p = ar.ArrivalParams(T=T, phi=phi)
            prec = p.N / p.sigma_eps**2
            if phi > 0:
                prec += (T - 1) / p.sigma_z**2
            assert prec == pytest.approx(p.Pi, rel=1e-12)


def test_push_linearity_public(dyn_half, pr_half):
    """bias = D@alpha, trading pnl = -alpha'D alpha, moments alpha-free."""
    al = np.array([0.3, 0.1, -0.2, 0.4])
    for stat in ("last", "win:2", "twap"):
        ev = ar.evaluate(dyn_half, al, stat)
        w = stat_weights(4, stat)
        assert ev["price_biases"] == pytest.approx(list(pr_half["D"] @ al), rel=1e-10)
        assert ev["stat_bias"] == pytest.approx(float(w @ pr_half["D"] @ al), rel=1e-10)
        assert ev["manip_trading_pnl"] == pytest.approx(
            -float(al @ pr_half["D"] @ al), rel=1e-10)
        sd = np.sqrt(w @ pr_half["cov_pp"] @ w)
        assert ev["stat_sd"] == pytest.approx(float(sd), rel=1e-10)


def test_mc_agreement_public(dyn_half):
    al = np.array([0.2, 0.1, 0.0, 0.3])
    ana = ar.evaluate(dyn_half, al, "win:2")
    mc = ar.mc_check_arrival(dyn_half, al, "win:2", n=600_000, seed=11)
    assert mc["stat_bias"] == pytest.approx(ana["stat_bias"], abs=3e-3)
    assert mc["decision_quality"] == pytest.approx(
        ana["decision_quality"], abs=4 * mc["decision_quality_se"])
    assert mc["corr_Pv"] == pytest.approx(ana["corr_Pv"], abs=3e-3)
    assert mc["price_biases"] == pytest.approx(ana["price_biases"], abs=4e-3)


def test_washout_monotone_in_phi():
    """The point of the model: an early push decays FASTER the more public
    information arrives (half-life decreasing in phi; end-of-window residual
    bias decreasing in phi)."""
    hl, resid = [], []
    for phi in (0.0, 0.25, 0.5, 0.75):
        dyn = ar.solve_honest_dynamics_arrival(ar.ArrivalParams(T=8, phi=phi))
        pr = ar.push_response(dyn)
        dec = ar.push_decay(pr, push_batch=0)
        hl.append(dec["half_life_batches"])
        resid.append(dec["bias_path"][-1] / dec["impact"])
    assert all(hl[i + 1] < hl[i] for i in range(3)), hl
    assert all(resid[i + 1] < resid[i] for i in range(3)), resid


def test_solver_local_optimality_public(dyn_half, pr_half):
    """Random-perturbation deviation check on the exact objective at the
    fast-solver optimum (public stream, win:2 and last)."""
    rng = np.random.default_rng(7)
    B = 5.0
    for stat in ("last", "win:2"):
        al = solve_manipulator_fast(pr_half, B=B, statistics=[stat])
        ev0 = ar.evaluate(dyn_half, al, stat)
        u0 = ev0["manip_trading_pnl"] + B * ev0["approval_prob"]
        for scale in (1e-3, 1e-2, 1e-1):
            for _ in range(100):
                d = rng.standard_normal(4) * scale
                ev = ar.evaluate(dyn_half, al + d, stat)
                u = ev["manip_trading_pnl"] + B * ev["approval_prob"]
                assert u <= u0 + 1e-12


def test_named_schedules_dominated_by_exact_br(pr_half, dyn_half):
    """Exact open-loop BR weakly beats the restricted named schedules
    (re-time-to-window-start, sustained, last-batch)."""
    B, stat = 5.0, "win:2"
    al = solve_manipulator_fast(pr_half, B=B, statistics=[stat])
    ev = ar.evaluate(dyn_half, al, stat)
    u_star = ev["manip_trading_pnl"] + B * ev["approval_prob"]
    for name, shape in ar.schedule_shapes(4, 2).items():
        _, u = ar.solve_scalar_schedule(pr_half, B=B, statistic=stat, shape=shape)
        assert u <= u_star + 1e-10, (name, u, u_star)


def test_staggered_all_at_batch1_reduces_to_symmetric():
    """All signals at batch 1 == the symmetric static model with the same
    (N, sigma_eps): betas (identical across traders), lams, D agree."""
    p_st = ar.StaggeredParams(N=4, T=4, arrival=(1, 1, 1, 1))
    dyn_st = ar.solve_honest_dynamics_staggered(p_st)
    p_tw = tw.TwapParams(N=4, sigma_eps=p_st.sigma_eps, sigma_u=1.0,
                         tau=0.3, T=4)
    dyn_tw = tw.solve_honest_dynamics(p_tw)
    for t in range(4):
        assert dyn_st["betas"][t] == pytest.approx(
            np.full(4, dyn_tw["betas"][t]), rel=1e-9)
    assert dyn_st["lams"] == pytest.approx(dyn_tw["lams"], rel=1e-9)
    pr_st, pr_tw = ar.push_response(dyn_st), tw.push_response(dyn_tw)
    assert pr_st["D"] == pytest.approx(pr_tw["D"], abs=1e-10)


def test_staggered_times_schedule():
    assert ar.staggered_times(4, 4, 0.0) == (1, 1, 1, 1)
    assert ar.staggered_times(4, 4, 0.25) == (1, 1, 1, 3)
    assert ar.staggered_times(4, 4, 0.5) == (1, 1, 2, 4)
    assert ar.staggered_times(4, 4, 0.75) == (1, 2, 3, 4)
    assert ar.staggered_times(4, 8, 0.5) == (1, 1, 2, 8)


def test_mc_agreement_staggered():
    p = ar.StaggeredParams(N=4, T=4, arrival=(1, 1, 2, 3))
    dyn = ar.solve_honest_dynamics_staggered(p)
    al = np.array([0.2, 0.1, 0.0, 0.3])
    ana = ar.evaluate(dyn, al, "win:2")
    mc = ar.mc_check_staggered(dyn, al, "win:2", n=600_000, seed=13)
    assert mc["stat_bias"] == pytest.approx(ana["stat_bias"], abs=3e-3)
    assert mc["decision_quality"] == pytest.approx(
        ana["decision_quality"], abs=4 * mc["decision_quality_se"])
    assert mc["corr_Pv"] == pytest.approx(ana["corr_Pv"], abs=3e-3)
    assert mc["price_biases"] == pytest.approx(ana["price_biases"], abs=4e-3)


def test_concealed_mixture_consistency_public(dyn_half, pr_half):
    B = 2.0
    stats = ["win:1", "win:2", "win:4"]
    al_c = solve_manipulator_fast(pr_half, B=B, statistics=stats)
    mix = ar.evaluate_mixture(dyn_half, al_c, stats, B=B)
    dq_manual = np.mean([mix["per_statistic"][s]["decision_quality"] for s in stats])
    assert mix["decision_quality_mix"] == pytest.approx(dq_manual, rel=1e-12)
    u_conc = mix["manip_total"]
    u_known = 0.0
    for s in stats:
        a_k = solve_manipulator_fast(pr_half, B=B, statistics=[s])
        ev = ar.evaluate(dyn_half, a_k, s)
        u_known += (ev["manip_trading_pnl"] + B * ev["approval_prob"]) / len(stats)
    assert u_conc <= u_known + 1e-10
