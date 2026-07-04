"""Disclosure-regime plumbing tests (pseudo-anonymity).

Regimes: "full" (per-trader orders), "aggregate" (clearing price + net flow),
"price" (clearing price only). Against a deterministic AMM the last two are
informationally identical — tested — so anonymity (attribution) is the only
knob with bite.
"""

import numpy as np

from batch_amm import lmsr_np as lmsr
from batch_amm.engine import Config, run_market
from batch_amm.envs import GalanisEnv, GaussianEnv
from batch_amm.metrics import summarize

B = 0.1


def test_aggregate_and_price_regimes_identical():
    env = GaussianEnv(n=5, m=400, seed=11)
    for mech in ("batch_lmsr", "batch_kyle"):
        a = run_market(env, Config(mech=mech, rounds=3, b=B, disclosure="aggregate"))
        p = run_market(env, Config(mech=mech, rounds=3, b=B, disclosure="price"))
        for key in ("final_price", "cash", "holdings", "mm_cash"):
            assert np.array_equal(a[key], p[key]), (mech, key)


def test_price_only_observer_recovers_net_flow():
    """The LMSR clearing price pins down the executed net flow exactly."""
    env = GaussianEnv(n=5, m=400, seed=12)
    res = run_market(env, Config(mech="batch_lmsr", rounds=1, b=B, disclosure="price"))
    x_net = res["holdings"].sum(axis=1)
    x_from_price = B * (lmsr.logit(res["final_price"]) - lmsr.logit(0.5))
    assert np.allclose(x_net, x_from_price, atol=1e-10)


def test_single_trader_anon_batch_reduces_to_sequential():
    env = GaussianEnv(n=1, m=300, seed=13)
    seq = run_market(env, Config(mech="seq_lmsr", rounds=3, b=B))
    for sizing in ("full", "competitive"):
        bat = run_market(
            env,
            Config(
                mech="batch_lmsr", rounds=3, b=B, sizing=sizing, disclosure="aggregate"
            ),
        )
        assert np.allclose(seq["final_price"], bat["final_price"], atol=1e-12)
        assert np.allclose(seq["cash"], bat["cash"], atol=1e-12)


def test_gaussian_anon_meanfield_exact_for_equal_signals():
    """When all signals coincide, the mean-field inversion is exact and the
    anonymous R=2 batch recovers the full-information posterior."""
    env = GaussianEnv(n=4, m=64, seed=14)
    env.signals = np.repeat(env.signals[:, :1], env.n, axis=1)  # equal signals
    res = run_market(
        env,
        Config(
            mech="batch_lmsr", rounds=2, b=B, sizing="competitive",
            disclosure="aggregate",
        ),
    )
    interior = (env.full_info_price() > 2e-4) & (env.full_info_price() < 1 - 2e-4)
    assert interior.mean() > 0.8
    assert np.allclose(
        res["final_price"][interior], env.full_info_price()[interior], atol=1e-6
    )


def test_galanis_anon_round1_reveals_bit_count_classes():
    env = GalanisEnv()
    state = env.make_state()
    ts = np.stack([env.honest_target(i, state) for i in range(env.n)])
    t_total = lmsr.logit(ts).sum(axis=0)
    env.reveal_batch_anon(t_total, state, first_time=True)
    from galanis_market.structures import STATES

    bitcount = np.array([sum(STATES[w]) for w in range(8)])
    for rep in range(8):
        support = state["belief"][rep] > 0
        expected = bitcount == bitcount[rep]
        assert np.array_equal(support, expected), rep


def test_galanis_anon_r2_matches_full_disclosure_aggregation():
    """Bit-count is sufficient for the symmetric payoff X = 1{>=2 bits}, so
    anonymity costs nothing here: R=2 anonymous == R=2 full, acc 1.0."""
    env = GalanisEnv()
    full = run_market(
        env, Config(mech="batch_lmsr", rounds=2, b=0.01, disclosure="full")
    )
    anon = run_market(
        env, Config(mech="batch_lmsr", rounds=2, b=0.01, disclosure="aggregate")
    )
    assert np.allclose(full["final_price"], anon["final_price"], atol=1e-9)
    s = summarize(env, anon)
    assert s["decision_acc"] == 1.0
    assert abs(s["log_loss_final"] - np.log(1 / 0.9)) < 1e-3


def test_seat_invariance_under_anonymity():
    env = GaussianEnv(n=5, m=200, seed=15)
    base = run_market(
        env,
        Config(
            mech="batch_lmsr", rounds=3, b=B, manip_seat=0, bounty=0.15,
            disclosure="aggregate",
        ),
    )
    perm = [4, 1, 2, 3, 0]
    moved = run_market(
        env.with_trader_order(perm),
        Config(
            mech="batch_lmsr", rounds=3, b=B, manip_seat=4, bounty=0.15,
            disclosure="aggregate",
        ),
    )
    assert np.allclose(base["final_price"], moved["final_price"], atol=1e-10)


def test_full_disclosure_default_unchanged():
    env = GaussianEnv(n=4, m=200, seed=16)
    default = run_market(env, Config(mech="batch_lmsr", rounds=3, b=B))
    explicit = run_market(env, Config(mech="batch_lmsr", rounds=3, b=B, disclosure="full"))
    assert np.array_equal(default["final_price"], explicit["final_price"])
