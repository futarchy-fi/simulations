"""Environment tests: inversion exactness, aggregation benchmarks, parity
with galanis-market's myopic.py."""

import numpy as np

from galanis_market.myopic import myopic_final_prices
from galanis_market.structures import STRUCTURES

from batch_amm.engine import Config, run_market
from batch_amm.envs import GalanisEnv, GaussianEnv, clip_price
from batch_amm.metrics import summarize

B = 0.1


def _unsaturated(env, state_targets):
    """Reps where no posted price touched the [eps, 1-eps] clip (the price
    floor deliberately loses tail information; exactness holds off the clip)."""
    from batch_amm.envs import PRICE_EPS

    t = np.stack(state_targets)
    lo, hi = PRICE_EPS * 1.5, 1.0 - PRICE_EPS * 1.5
    return np.all((t > lo) & (t < hi), axis=0)


def test_gaussian_seq_inversion_recovers_true_signals():
    env = GaussianEnv(n=4, m=256, seed=3)
    state = env.make_state()
    targets = []
    for i in range(env.n):
        t = env.honest_target(i, state)
        targets.append(t)
        env.reveal(i, t, state, first_time=True)
    mask = _unsaturated(env, targets)
    assert mask.mean() > 0.9
    assert np.allclose(state["pseudo"][mask], env.signals[mask], atol=1e-6)


def test_gaussian_seq_final_price_is_full_info_posterior():
    env = GaussianEnv(n=5, m=512, seed=4)
    res = run_market(env, Config(mech="seq_lmsr", rounds=1, b=B))
    mask = _unsaturated(
        env, list(res["round_prices"]) + [res["final_price"], env.full_info_price()]
    )
    # sequential myopic-Bayes play with exact inversion is fully revealing:
    # the last trader's quote IS the full-information posterior
    state = env.make_state()
    targets = [env.honest_target(0, state)]
    for i in range(env.n):
        t = env.honest_target(i, state)
        targets.append(t)
        env.reveal(i, t, state, first_time=True)
    mask &= _unsaturated(env, targets)
    assert mask.mean() > 0.8
    assert np.allclose(
        res["final_price"][mask], env.full_info_price()[mask], atol=1e-6
    )


def test_gaussian_batch_competitive_r3_converges_to_full_info():
    env = GaussianEnv(n=5, m=512, seed=5)
    res = run_market(
        env, Config(mech="batch_lmsr", rounds=3, b=B, sizing="competitive")
    )
    # round-1 individual targets must be interior for exact inversion
    state = env.make_state()
    targets = [env.honest_target(i, state) for i in range(env.n)]
    mask = _unsaturated(env, targets + [env.full_info_price()])
    assert mask.mean() > 0.9
    assert np.allclose(
        res["final_price"][mask], env.full_info_price()[mask], atol=1e-6
    )


def test_gaussian_conservation():
    env = GaussianEnv(n=10, m=1000, seed=6)
    for mech in ("seq_lmsr", "batch_lmsr", "batch_kyle"):
        res = run_market(env, Config(mech=mech, rounds=3, b=B))
        s = summarize(env, res)
        assert s["conservation_maxabs"] < 1e-9, mech


def test_galanis_seq_matches_myopic_reference():
    """SEQ R=1 (3 turns) reproduces galanis-market's myopic trajectory."""
    env = GalanisEnv()
    res = run_market(env, Config(mech="seq_lmsr", rounds=1, b=B))
    ref = myopic_final_prices(STRUCTURES["t3s111y2"], num_rounds=3)
    # quotes are capped to the env's 0.1/0.9 grid-comparable bounds
    ref_arr = np.clip(np.array([ref[k] for k in "abcdefgh"]), *env.target_bounds)
    assert np.allclose(res["final_price"], ref_arr, atol=1e-9)


def test_galanis_seq_aggregates_perfectly():
    env = GalanisEnv()
    res = run_market(env, Config(mech="seq_lmsr", rounds=1, b=B))
    s = summarize(env, res)
    assert s["decision_acc"] == 1.0


def test_galanis_batch_seat_invariance():
    """Batch outcomes are invariant to trader ordering by construction."""
    env = GalanisEnv()
    for mech in ("batch_lmsr", "batch_kyle"):
        base = run_market(
            env, Config(mech=mech, rounds=3, b=B, manip_seat=0, bounty=0.05)
        )
        perm = [2, 0, 1]  # original trader 0 now sits at seat 1
        env_p = env.with_trader_order(perm)
        moved = run_market(
            env_p, Config(mech=mech, rounds=3, b=B, manip_seat=1, bounty=0.05)
        )
        assert np.allclose(base["final_price"], moved["final_price"], atol=1e-12)
        # per-trader records permute with the seats
        assert np.allclose(
            base["cash"][:, [2, 0, 1]], moved["cash"], atol=1e-12
        )


def test_gaussian_batch_seat_invariance():
    env = GaussianEnv(n=5, m=200, seed=7)
    base = run_market(
        env, Config(mech="batch_lmsr", rounds=3, b=B, manip_seat=0, bounty=0.1)
    )
    perm = [4, 1, 2, 3, 0]
    env_p = env.with_trader_order(perm)
    moved = run_market(
        env_p, Config(mech="batch_lmsr", rounds=3, b=B, manip_seat=4, bounty=0.1)
    )
    assert np.allclose(base["final_price"], moved["final_price"], atol=1e-10)


def test_manipulator_shifts_price_up():
    env = GalanisEnv()
    honest = run_market(env, Config(mech="seq_lmsr", rounds=1, b=B))
    manip = run_market(
        env, Config(mech="seq_lmsr", rounds=1, b=B, manip_seat=2, bounty=0.2)
    )
    w = env.weights
    assert ((manip["final_price"] - honest["final_price"]) * w).sum() > 0.01
