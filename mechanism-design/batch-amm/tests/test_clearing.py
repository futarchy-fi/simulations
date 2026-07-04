"""Unit tests for the batch clearing rule and mechanism identities."""

import numpy as np
import pytest

from batch_amm import lmsr_np as lmsr
from batch_amm.engine import Config, manip_target_lmsr, run_market


class FakeEnv:
    """Environment stub with scripted targets (one round's worth, reused)."""

    exact = True

    def __init__(self, targets):
        # targets: (N,) or (N, M) fixed price targets
        t = np.asarray(targets, dtype=float)
        if t.ndim == 1:
            t = t[:, None]
        self.targets = t  # (N, M)
        self.n, self.m = self.targets.shape
        self.weights = np.full(self.m, 1.0 / self.m)
        self.payout = np.ones(self.m)

    def make_state(self):
        return {}

    def honest_target(self, i, state):
        return self.targets[i].copy()

    def reveal(self, i, implied, state, first_time):
        pass

    def reveal_batch(self, implied, state, first_time):
        pass

    def full_info_price(self):
        return np.full(self.m, 0.5)


B = 0.1


def test_offsetting_orders_zero_curve_movement():
    """Netting identity: logit-symmetric targets around p=0.5 cancel exactly."""
    env = FakeEnv([0.7, 0.3])
    res = run_market(env, Config(mech="batch_lmsr", rounds=1, b=B, sizing="full"))
    assert np.allclose(res["final_price"], 0.5, atol=1e-12)
    assert np.allclose(res["mm_cash"], 0.0, atol=1e-12)
    # both fill at mid (pi = 0.5): buyer's cash = -0.5*x, seller's = +0.5*x
    x = B * (lmsr.logit(0.7) - lmsr.logit(0.5))
    assert np.allclose(res["cash"][0, 0], -0.5 * x)
    assert np.allclose(res["cash"][0, 1], +0.5 * x)
    assert np.allclose(res["slip_own"], 0.0, atol=1e-12)  # mid execution


def test_single_trader_batch_reduces_to_sequential():
    for sizing in ("full", "competitive"):
        env = FakeEnv([0.8])
        seq = run_market(env, Config(mech="seq_lmsr", rounds=1, b=B))
        bat = run_market(
            env, Config(mech="batch_lmsr", rounds=1, b=B, sizing=sizing)
        )
        assert np.allclose(seq["final_price"], bat["final_price"])
        assert np.allclose(seq["holdings"], bat["holdings"])
        assert np.allclose(seq["cash"], bat["cash"], atol=1e-12)
        assert np.allclose(seq["mm_cash"], bat["mm_cash"])


def test_uniform_price_between_posted_and_post_trade():
    env = FakeEnv([0.7, 0.8, 0.9])
    res = run_market(env, Config(mech="batch_lmsr", rounds=1, b=B, sizing="full"))
    p1 = res["final_price"][0]
    x = res["holdings"][0]
    pi = -res["cash"][0, 0] / x[0]
    assert 0.5 < pi < p1
    # all traders fill at the same uniform price
    pis = -res["cash"][0] / x
    assert np.allclose(pis, pi)
    # AMM receives exactly the cost of the net move
    assert np.allclose(res["mm_cash"][0], pi * x.sum())


def test_competitive_sizing_identical_targets_clear_at_target():
    env = FakeEnv([0.8, 0.8, 0.8, 0.8, 0.8])
    res = run_market(
        env, Config(mech="batch_lmsr", rounds=1, b=B, sizing="competitive")
    )
    assert np.allclose(res["final_price"], 0.8, atol=1e-12)


def test_full_sizing_identical_targets_overshoot():
    env = FakeEnv([0.8, 0.8, 0.8])
    res = run_market(env, Config(mech="batch_lmsr", rounds=1, b=B, sizing="full"))
    lp = 3 * lmsr.logit(0.8)  # sum of individual logit moves from 0
    assert np.allclose(res["final_price"], lmsr.sigmoid(lp))


def test_cash_conservation_all_mechanisms():
    rng = np.random.default_rng(0)
    targets = rng.uniform(0.05, 0.95, size=(4, 64))
    env = FakeEnv(targets)
    for mech in ("seq_lmsr", "batch_lmsr", "batch_kyle"):
        for sizing in ("full", "competitive"):
            res = run_market(env, Config(mech=mech, rounds=3, b=B, sizing=sizing))
            trader_cash = res["cash"].sum(axis=1)
            assert np.allclose(trader_cash + res["mm_cash"], 0.0, atol=1e-10), (
                mech,
                sizing,
            )


def test_seq_mm_revenue_is_path_independent():
    """LMSR aggregate 'leak to the curve' depends only on (p0, p_final)."""
    rng = np.random.default_rng(1)
    targets = rng.uniform(0.1, 0.9, size=(5, 32))
    env = FakeEnv(targets)
    res = run_market(env, Config(mech="seq_lmsr", rounds=2, b=B))
    expected = lmsr.cost_to_move(np.full(env.m, 0.5), res["final_price"], B)
    assert np.allclose(res["mm_cash"], expected, atol=1e-12)


def test_batch_kyle_netting_and_single_trader():
    # offsetting linear orders cancel: equal and opposite targets around p
    env = FakeEnv([0.7, 0.3])
    res = run_market(env, Config(mech="batch_kyle", rounds=1, b=B, sizing="full"))
    assert np.allclose(res["final_price"], 0.5, atol=1e-12)
    assert np.allclose(res["mm_cash"], 0.0, atol=1e-12)
    # single trader fills at full post-impact price p + lam*x = (p+t)/2
    env1 = FakeEnv([0.8])
    res1 = run_market(env1, Config(mech="batch_kyle", rounds=1, b=B, sizing="full"))
    lam = 0.25 / B
    x = (0.8 - 0.5) / (2 * lam)
    assert np.allclose(res1["final_price"], 0.5 + lam * x)
    assert np.allclose(res1["cash"][0, 0], -(0.5 + lam * x) * x)


def test_manip_target_reduces_to_posterior_at_zero_bounty():
    q = np.array([0.1, 0.4, 0.9])
    assert np.allclose(manip_target_lmsr(q, B, 0.0), q)


@pytest.mark.parametrize("bounty", [0.01, 0.05, 0.3])
@pytest.mark.parametrize("q", [0.2, 0.5, 0.85])
def test_manip_target_is_argmax_of_myopic_utility(bounty, q):
    p0 = 0.5
    grid = np.linspace(0.001, 0.999, 4001)

    def util(t):
        shares = B * (lmsr.logit(t) - lmsr.logit(p0))
        cost = lmsr.cost_to_move(np.array([p0]), t, B)
        return q * shares - cost + bounty * (t - 0.5)

    t_star = float(manip_target_lmsr(np.array([q]), B, bounty)[0])
    u_grid = util(grid)
    assert util(np.array([t_star]))[0] >= u_grid.max() - 1e-9
    assert t_star > q  # bounty always pulls the target up


def test_pro_rata_rationing_at_price_cap():
    """Extreme same-direction flow hits the price cap; orders scale pro-rata."""
    env = FakeEnv([0.9999, 0.9999, 0.9999, 0.9999, 0.9999])
    res = run_market(env, Config(mech="batch_lmsr", rounds=1, b=B, sizing="full"))
    from batch_amm.envs import PRICE_EPS

    assert np.allclose(res["final_price"], 1.0 - PRICE_EPS)
    trader_cash = res["cash"].sum(axis=1)
    assert np.allclose(trader_cash + res["mm_cash"], 0.0, atol=1e-10)
    # equal orders ration equally
    assert np.allclose(res["holdings"][0], res["holdings"][0][0])
