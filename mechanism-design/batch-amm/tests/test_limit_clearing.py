"""Unit tests for limit-order batch clearing (clearing.py + the
batch_lmsr_limit mechanism).

Clearing rule under test (uniform-price call auction against the LMSR):
a buy fills only if the uniform clearing price pi <= its limit, a sell only
if pi >= its limit; pi is the fixed point pi = phi(D(pi)) where phi(X) is
the LMSR average execution price of net flow X and D(pi) the eligible net
demand; at a jump the crossing price is the marginal order's limit and
marginal orders fill pro-rata (standard uniform-price treatment).
"""

import numpy as np
import pytest

from batch_amm import lmsr_np as lmsr
from batch_amm.clearing import avg_price, clear_limit_batch

B = 0.1
INF = np.inf


def _mkt_move(p0, targets, b=B, scale=1.0):
    """Order sizes of the plain batch mechanism: scale*b*(logit t - logit p)."""
    p0 = np.asarray(p0, dtype=float)
    t = np.asarray(targets, dtype=float)
    return scale * b * (lmsr.logit(t) - lmsr.logit(p0)[None, :])


def _conserves(p0, res, b=B, atol=1e-12):
    dc = lmsr.cost_to_move(np.asarray(p0, float), res["p1"], b)
    lhs = (res["pi"][None, :] * res["x_exec"]).sum(axis=0)
    np.testing.assert_allclose(lhs, dc, atol=atol)


def test_avg_price_continuous_at_zero_and_increasing():
    p = np.full(5, 0.5)
    xs = np.array([-0.05, -1e-16, 0.0, 1e-16, 0.05])
    phi = avg_price(p, xs, B)
    assert abs(phi[2] - 0.5) < 1e-15
    assert np.all(np.diff(phi) >= 0)
    # exact endpoints: average price of the move p -> p1
    p1 = lmsr.sigmoid(lmsr.logit(p) + xs / B)
    assert phi[0] > p1[0] and phi[-1] < p1[-1]  # avg between p and p1


def test_infinite_limits_reproduce_plain_netting_bitexact():
    rng = np.random.default_rng(7)
    p0 = rng.uniform(0.2, 0.8, size=64)
    targets = rng.uniform(0.05, 0.95, size=(4, 64))
    x = _mkt_move(p0, targets)
    limits = np.where(x > 0, INF, -INF)
    res = clear_limit_batch(p0, x, limits, B)
    # exact base-engine arithmetic
    lp = lmsr.logit(p0)
    x_net = x.sum(axis=0)
    lmax = float(lmsr.logit(1.0 - 1e-4))
    lp1 = np.clip(lp + x_net / B, -lmax, lmax)
    p1 = lmsr.sigmoid(lp1)
    dc = lmsr.cost_to_move(p0, p1, B)
    x_exec_net = B * (lp1 - lp)
    alpha = x_exec_net / x_net
    pi = dc / x_exec_net
    assert np.array_equal(res["p1"], p1)
    assert np.array_equal(res["x_exec"], x * alpha)
    assert np.array_equal(res["pi"], pi)
    _conserves(p0, res)


def test_single_buyer_tight_limit_never_exceeded():
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.8]])
    limits = np.array([[0.8]])
    res = clear_limit_batch(p0, x, limits, B)
    # phi of the full move < 0.8, so the limit does not bind: full fill
    assert np.allclose(res["fill"], 1.0)
    assert np.allclose(res["p1"], 0.8)
    assert res["pi"][0] <= 0.8 and res["pi"][0] > 0.5
    _conserves(p0, res)


def test_extramarginal_buy_drops_out_entirely():
    """Second buyer's limit (0.55) is below phi of the first buyer's flow
    alone, so they drop out completely and pi is set by the remaining flow."""
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.9], [0.55]])
    limits = np.array([[INF], [0.55]])
    res = clear_limit_batch(p0, x, limits, B)
    phi_alone = avg_price(p0, x[0], B)
    assert phi_alone[0] > 0.55  # premise of the test
    assert res["fill"][0, 0] == 1.0
    assert res["fill"][1, 0] == 0.0
    assert np.allclose(res["pi"], phi_alone)
    assert np.allclose(res["x_exec"][0], x[0]) and np.allclose(res["x_exec"][1], 0.0)
    _conserves(p0, res)


def test_marginal_buys_fill_pro_rata_at_their_limit():
    """Three identical buys with limit 0.6 are marginal: including them all
    pushes phi above 0.6, excluding them leaves phi below 0.6 -> pi pins at
    0.6 exactly and they fill an equal partial fraction."""
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.58], [0.7], [0.7], [0.7]])
    limits = np.array([[INF], [0.6], [0.6], [0.6]])
    assert avg_price(p0, x[0], B)[0] < 0.6 < avg_price(p0, x.sum(axis=0), B)[0]
    res = clear_limit_batch(p0, x, limits, B)
    assert abs(res["pi"][0] - 0.6) < 1e-9
    assert res["fill"][0, 0] == 1.0
    f = res["fill"][1:, 0]
    assert np.allclose(f, f[0]) and 0.0 < f[0] < 1.0
    # executed net flow has average price exactly at the marginal limit
    assert abs(avg_price(p0, res["x_exec"].sum(axis=0), B)[0] - 0.6) < 1e-9
    _conserves(p0, res)


def test_marginal_single_sell_partial_fill():
    """A lone sell with a limit above phi(full order) partially fills so that
    the average execution price equals the limit exactly."""
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.4]])
    limits = np.array([[0.48]])
    assert avg_price(p0, x[0], B)[0] < 0.48  # full fill would violate limit
    res = clear_limit_batch(p0, x, limits, B)
    assert abs(res["pi"][0] - 0.48) < 1e-9
    assert 0.0 < res["fill"][0, 0] < 1.0
    _conserves(p0, res)


def test_sell_side_eligibility():
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.4]])
    limits = np.array([[0.4]])
    res = clear_limit_batch(p0, x, limits, B)
    assert np.allclose(res["fill"], 1.0)  # phi(=0.45) >= 0.4: fills in full
    assert res["pi"][0] >= 0.4 and res["pi"][0] < 0.5
    assert np.allclose(res["p1"], 0.4)
    _conserves(p0, res)


def test_exact_cancellation_fills_fully_at_mid():
    """Orders that net to a pure rounding error (|X| < 1e-14 but != 0) are
    exact cancellation: full fills crossing at mid, no alpha blow-up from
    dividing two rounding errors (regression: manipulated Galanis rounds
    produce exactly-offsetting flow and executed > submitted resulted)."""
    p0 = np.array([0.7])
    x = np.array([[1e-3 + 1e-16], [-0.5e-3], [-0.5e-3]])
    assert 0.0 < abs(x.sum()) < 1e-14
    limits = np.array([[INF], [0.6], [0.6]])
    res = clear_limit_batch(p0, x, limits, B)
    assert np.array_equal(res["x_exec"], x)
    assert np.allclose(res["fill"], 1.0)
    assert np.allclose(res["pi"], 0.7) and np.allclose(res["p1"], 0.7)


def test_offsetting_tight_orders_cross_at_mid():
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.7], [0.3]])
    limits = np.array([[0.7], [0.3]])
    res = clear_limit_batch(p0, x, limits, B)
    assert np.allclose(res["fill"], 1.0)
    assert np.allclose(res["pi"], 0.5) and np.allclose(res["p1"], 0.5)
    _conserves(p0, res)


def test_eligibility_respected_and_conservation_random():
    """Property test: fills in [0,1]; filled orders never breach their limit;
    fully-dropped orders would have breached it; cash conserves."""
    rng = np.random.default_rng(42)
    m = 512
    p0 = rng.uniform(0.15, 0.85, size=m)
    targets = rng.uniform(0.02, 0.98, size=(6, m))
    x = _mkt_move(p0, targets, scale=1.0 / 6)
    slack = rng.choice([0.0, 0.03, 0.1, INF], size=(6, m))
    limits = np.where(x > 0, targets + slack, targets - slack)
    res = clear_limit_batch(p0, x, limits, B)
    assert np.all(res["fill"] >= 0.0) and np.all(res["fill"] <= 1.0)
    pi = np.broadcast_to(res["pi"][None, :], x.shape)
    buys, sells = x > 0, x < 0
    filled = res["fill"] > 0
    tol = 1e-8
    assert np.all(pi[buys & filled] <= limits[buys & filled] + tol)
    assert np.all(pi[sells & filled] >= limits[sells & filled] - tol)
    dropped_b = buys & ~filled
    dropped_s = sells & ~filled
    assert np.all(limits[dropped_b] <= pi[dropped_b] + tol)
    assert np.all(limits[dropped_s] >= pi[dropped_s] - tol)
    _conserves(p0, res, atol=1e-10)


def test_price_cap_rationing_conserves_cash():
    p0 = np.array([0.5])
    x = _mkt_move(p0, [[0.9999]] * 5)
    limits = np.full((5, 1), INF)
    res = clear_limit_batch(p0, x, limits, B)
    assert np.allclose(res["p1"], 1.0 - 1e-4)
    # "fill" is the auction fill (1 here); the cap rations executions pro-rata
    assert np.allclose(res["fill"], 1.0)
    f = res["x_exec"][:, 0] / x[:, 0]
    assert np.allclose(f, f[0]) and f[0] < 1.0
    _conserves(p0, res)


# --------------------------------------------------------------------------- #
# engine-level: the batch_lmsr_limit mechanism
# --------------------------------------------------------------------------- #

from batch_amm.engine import Config, run_market  # noqa: E402
from batch_amm.envs import GalanisEnv, GaussianEnv  # noqa: E402
from batch_amm.metrics import summarize  # noqa: E402


def _pnl(env, res):
    return res["holdings"] * env.payout[:, None] + res["cash"]


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("sizing", ["competitive", "full"])
def test_engine_infinite_slack_reproduces_market_orders_bitexact(r, sizing):
    env = GaussianEnv(n=5, m=400, sigma_eps=1.0, seed=11)
    res_m = run_market(
        env, Config(mech="batch_lmsr", rounds=r, b=B, sizing=sizing,
                    disclosure="price"))
    res_l = run_market(
        env, Config(mech="batch_lmsr_limit", rounds=r, b=B, sizing=sizing,
                    disclosure="price", limit_slack=np.inf))
    for key in ("final_price", "round_prices", "cash", "holdings", "mm_cash"):
        assert np.array_equal(res_m[key], res_l[key]), key


def test_engine_infinite_slack_manip_bitexact_galanis():
    env = GalanisEnv()
    res_m = run_market(
        env, Config(mech="batch_lmsr", rounds=3, b=0.01, sizing="competitive",
                    disclosure="price", manip_seat=0, bounty=0.05))
    res_l = run_market(
        env, Config(mech="batch_lmsr_limit", rounds=3, b=0.01,
                    sizing="competitive", disclosure="price", manip_seat=0,
                    bounty=0.05, limit_slack=np.inf))
    for key in ("final_price", "round_prices", "cash", "holdings", "mm_cash"):
        assert np.array_equal(res_m[key], res_l[key]), key


def test_engine_single_trader_never_executes_past_posterior():
    env = GaussianEnv(n=1, m=800, sigma_eps=1.0, seed=13)
    posterior = env.honest_target(0, env.make_state())
    res = run_market(
        env,
        Config(mech="batch_lmsr_limit", rounds=1, b=B, limit_slack=0.0,
               disclosure="price"),
    )
    buys = res["holdings"][:, 0] > 0
    sells = res["holdings"][:, 0] < 0
    pi_paid = np.where(
        res["holdings"][:, 0] != 0.0,
        res["cash"][:, 0] / -np.where(res["holdings"][:, 0] != 0, res["holdings"][:, 0], 1.0),
        0.5,
    )
    assert np.all(pi_paid[buys] <= posterior[buys] + 1e-9)
    assert np.all(pi_paid[sells] >= posterior[sells] - 1e-9)
    # alone, phi(full move) never reaches the posterior: full fill at N=1
    assert np.allclose(res["final_price"], posterior)


def test_engine_manip_extreme_limit_fills_fully_under_tight_honest_limits():
    env = GaussianEnv(n=3, m=500, sigma_eps=1.0, seed=17)
    cfg = Config(mech="batch_lmsr_limit", rounds=1, b=B, limit_slack=0.0,
                 disclosure="price", manip_seat=0, bounty=0.5)
    res = run_market(env, cfg)
    # the manipulator's +/-inf limit is never the marginal one: full fill
    # (auction fill; the rare price-cap rationing is the only haircut)
    sub = res["volume_submitted"][:, 0]
    got = res["volume"][:, 0]
    capped = np.isclose(res["final_price"], 1.0 - 1e-4) | np.isclose(
        res["final_price"], 1e-4
    )
    assert np.all(got[~capped] >= sub[~capped] - 1e-12)


@pytest.mark.parametrize("slack", [0.0, 0.05, np.inf])
def test_engine_conservation_and_welfare_identity(slack):
    env = GaussianEnv(n=5, m=400, sigma_eps=1.0, seed=19)
    res = run_market(
        env,
        Config(mech="batch_lmsr_limit", rounds=3, b=B, limit_slack=slack,
               disclosure="price"),
    )
    s = summarize(env, res)
    assert s["conservation_maxabs"] < 1e-10
    # LMSR welfare identity: total trader PnL = b*(ln2 - LL(p_final))
    pnl_tot = _pnl(env, res).sum(axis=1)
    pf = np.clip(res["final_price"], 1e-12, 1 - 1e-12)
    ll = -np.where(env.payout > 0.5, np.log(pf), np.log1p(-pf))
    np.testing.assert_allclose(pnl_tot, B * (np.log(2.0) - ll), atol=1e-10)


def test_engine_seat_invariance_with_manip():
    env = GaussianEnv(n=4, m=300, sigma_eps=1.0, seed=23)
    outs = []
    for seat in (0, 3):
        perm = [(j - seat) % 4 for j in range(4)]
        cfg = Config(mech="batch_lmsr_limit", rounds=2, b=B, limit_slack=0.0,
                     disclosure="price", manip_seat=seat, bounty=0.15)
        outs.append(run_market(env.with_trader_order(perm), cfg))
    np.testing.assert_allclose(
        outs[0]["final_price"], outs[1]["final_price"], atol=1e-12
    )
    np.testing.assert_allclose(
        outs[0]["cash"][:, 0], outs[1]["cash"][:, 3], atol=1e-12
    )


def test_galanis_limit_updater_matches_market_run_when_limits_never_bind():
    """slack=1.0 puts every limit at the price bounds (never binds), but the
    engine still routes disclosure through the exact limit-order consistency
    updater — it must land on the market-order anonymous run's prices."""
    env = GalanisEnv()
    res_m = run_market(
        env, Config(mech="batch_lmsr", rounds=2, b=0.01,
                    sizing="competitive", disclosure="price"))
    res_l = run_market(
        env, Config(mech="batch_lmsr_limit", rounds=2, b=0.01,
                    sizing="competitive", disclosure="price",
                    limit_slack=1.0))
    np.testing.assert_allclose(res_l["final_price"], res_m["final_price"],
                               atol=1e-9)
    assert res_l["jams"] == 0


def test_galanis_tight_limits_honest_run_keeps_decision_accuracy():
    env = GalanisEnv()
    res = run_market(
        env, Config(mech="batch_lmsr_limit", rounds=2, b=0.01,
                    sizing="competitive", disclosure="price",
                    limit_slack=0.0))
    assert res["jams"] == 0  # honest play is always self-consistent
    s = summarize(env, res)
    assert s["decision_acc"] == 1.0


def test_galanis_jam_counter_market_order_anon_manip():
    """BATCH.md section 9: a manipulated anonymous aggregate is unexplainable
    under strict consistency -> the update jams (belief frozen)."""
    env = GalanisEnv()
    res_h = run_market(
        env, Config(mech="batch_lmsr", rounds=3, b=0.01,
                    sizing="competitive", disclosure="price"))
    assert res_h["jams"] == 0
    res_m = run_market(
        env, Config(mech="batch_lmsr", rounds=3, b=0.01,
                    sizing="competitive", disclosure="price",
                    manip_seat=0, bounty=0.2))
    assert res_m["jams"] > 0


def test_engine_tight_limits_drop_like_minded_fills():
    """With correlated one-directional flow and slack=0, some honest traders
    must drop out of the fill (the execution-quality question, Q2)."""
    env = GaussianEnv(n=10, m=2000, sigma_eps=1.0, seed=29)
    res = run_market(
        env,
        Config(mech="batch_lmsr_limit", rounds=1, b=B, limit_slack=0.0,
               disclosure="price"),
    )
    fill_rate = res["volume"].sum() / res["volume_submitted"].sum()
    assert fill_rate < 0.999  # strictly some unfilled volume
    res_inf = run_market(
        env,
        Config(mech="batch_lmsr_limit", rounds=1, b=B, limit_slack=np.inf,
               disclosure="price"),
    )
    rate_inf = res_inf["volume"].sum() / res_inf["volume_submitted"].sum()
    assert rate_inf > fill_rate
