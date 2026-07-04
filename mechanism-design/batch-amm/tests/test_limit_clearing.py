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
