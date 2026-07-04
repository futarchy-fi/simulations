"""Limit-order sweeps (BATCH-LMSR-LIMIT vs the market-order baselines):
writes results/limits_core.json, results/limits_manip.json,
results/limits_jam.json.

Orders in the limit mechanism are (direction, quantity, limit): a buy fills
only if the uniform clearing price pi <= limit, a sell only if pi >= limit
(clearing.py). Honest limit = posterior +/- slack; slack = 0 is "never trade
past my belief", slack = inf reproduces the market-order mechanism
bit-exactly (unit-tested). Everything runs under the price-only disclosure
regime (the standing pseudo-anonymity rule).

Sweeps (competitive sizing throughout; CRN with run_sweeps.py via the same
SEED and per-N env seeds):
  core  — honest population, env x N x R in {1,2,3,5} x slack; arms
          seq_lmsr / batch_lmsr (market orders, price disclosure) /
          batch_lmsr_limit per slack; paired diffs vs both baselines +
          fill rates.                                    -> Q2 (execution),
                                                            Q3 (aggregation)
  manip — one bribed trader (bounty * (p_final - 0.5)); SEQ per-seat rows;
          batch market-order rows; batch limit rows per slack with the
          manipulator best-responding over an order-size multiplier grid
          GAMMAS (limit = +/-inf: they want fills); utility = market PnL +
          bounty*(p_final - 0.5); full grid recorded + argmax chosen; seat
          invariance spot-checked at the chosen gamma.    -> Q1 (manip cost)
  jam   — Galanis, R = 3, bounty x slack x gamma grid, jams counter =
          rounds x reps where the strict-consistency anonymous update
          rejected the aggregate as unexplainable.        -> Q4 (jamming)

Usage: python scripts/run_limits.py [--fast] [--only core|manip|jam]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from batch_amm.engine import Config, run_market
from batch_amm.envs import GalanisEnv, GaussianEnv
from batch_amm.metrics import paired_diff, summarize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SEED = 20260704  # same base as run_sweeps.py -> same Gaussian draws per N
N_SWEEP = [3, 5, 10, 25]
R_SWEEP = [1, 2, 3, 5]
SLACKS = [0.0, 0.02, 0.05, float("inf")]
MANIP_SLACKS = [0.0, 0.05, float("inf")]
GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
GAUSS_BOUNTIES = [0.05, 0.15, 0.5]
GALANIS_BOUNTIES = [0.02, 0.05, 0.2]
B_LIQ = 0.1           # Gaussian env
B_LIQ_GALANIS = 0.01  # matches galanis-market CFR+ setup (MANIPULATION.md)


def _b(env_name: str) -> float:
    return B_LIQ if env_name == "gaussian" else B_LIQ_GALANIS


def _slack_key(slack: float) -> str:
    return "inf" if np.isinf(slack) else f"{slack:g}"


def _dump(name: str, obj) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}")


def _gauss_env(n: int, m: int) -> GaussianEnv:
    return GaussianEnv(n=n, m=m, sigma_eps=1.0, seed=SEED + n)  # CRN per N


def _fill_rate(res) -> float:
    sub = res["volume_submitted"].sum()
    return float(res["volume"].sum() / sub) if sub > 0 else 1.0


def core(m_reps: int) -> None:
    records = []
    for env_name in ("gaussian", "galanis"):
        ns = N_SWEEP if env_name == "gaussian" else [3]
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            b = _b(env_name)
            for r in R_SWEEP:
                seq_res = run_market(env, Config(mech="seq_lmsr", rounds=r, b=b))
                records.append(
                    {"env": env_name, "N": n, "R": r, "mech": "seq_lmsr",
                     "slack": None, "summary": summarize(env, seq_res)}
                )
                mkt_res = run_market(
                    env, Config(mech="batch_lmsr", rounds=r, b=b,
                                sizing="competitive", disclosure="price")
                )
                records.append(
                    {"env": env_name, "N": n, "R": r, "mech": "batch_lmsr",
                     "slack": None, "summary": summarize(env, mkt_res),
                     "fill_rate": _fill_rate(mkt_res),
                     "paired_vs_seq": paired_diff(env, mkt_res, seq_res)}
                )
                for slack in SLACKS:
                    res = run_market(
                        env, Config(mech="batch_lmsr_limit", rounds=r, b=b,
                                    sizing="competitive", disclosure="price",
                                    limit_slack=slack)
                    )
                    records.append(
                        {"env": env_name, "N": n, "R": r,
                         "mech": "batch_lmsr_limit", "slack": _slack_key(slack),
                         "summary": summarize(env, res),
                         "fill_rate": _fill_rate(res),
                         "jams": int(res["jams"]),
                         "paired_vs_seq": paired_diff(env, res, seq_res),
                         "paired_vs_batch_market": paired_diff(env, res, mkt_res)}
                    )
                print(f"core {env_name} N={n} R={r} done", flush=True)
    _dump("limits_core.json", records)


def _manip_row(env, env_name, n, r, mech, seat, bounty, slack, gamma, res_h_cache):
    """One manipulated run vs its honest counterfactual (same mechanism)."""
    b = _b(env_name)
    perm = [(j - seat) % n for j in range(n)]
    env_k = env.with_trader_order(perm)
    sizing = "full" if mech == "seq_lmsr" else "competitive"
    disclosure = "full" if mech == "seq_lmsr" else "price"
    slack_val = slack if slack is not None else 0.0  # ignored off-limit mechs
    hkey = (mech, seat, r, _slack_key(slack) if slack is not None else None)
    if hkey not in res_h_cache:
        res_h_cache[hkey] = run_market(
            env_k, Config(mech=mech, rounds=r, b=b, sizing=sizing,
                          disclosure=disclosure, limit_slack=slack_val)
        )
    res_h = res_h_cache[hkey]
    cfg_m = Config(mech=mech, rounds=r, b=b, sizing=sizing,
                   disclosure=disclosure, limit_slack=slack_val,
                   manip_seat=seat, bounty=bounty,
                   manip_scale=gamma if mech == "batch_lmsr_limit" else 1.0)
    res_m = run_market(env_k, cfg_m)
    w = env.weights
    payout = env.payout

    def wm(x):
        return float((x * w).sum() / w.sum())

    pnl_h = res_h["holdings"] * payout[:, None] + res_h["cash"]
    pnl_m = res_m["holdings"] * payout[:, None] + res_m["cash"]
    honest_idx = [j for j in range(n) if j != seat]
    d_pf = wm(res_m["final_price"] - res_h["final_price"])
    manip_pnl = wm(pnl_m[:, seat])
    manip_pnl_h = wm(pnl_h[:, seat])
    cost = manip_pnl_h - manip_pnl  # market-PnL drop vs honest play, same draws
    sm = summarize(env_k, res_m)
    sub = res_m["volume_submitted"][:, seat].sum()
    return {
        "env": env_name, "N": n, "R": r, "mech": mech,
        "slack": _slack_key(slack) if slack is not None else None,
        "gamma": gamma, "seat": seat, "bounty": bounty,
        "delta_final_price": d_pf,
        "manip_market_pnl": manip_pnl,
        "manip_market_pnl_honest_run": manip_pnl_h,
        "manip_cost": cost,
        "cost_per_unit_dp": cost / d_pf if abs(d_pf) > 1e-12 else None,
        "manip_utility": wm(
            pnl_m[:, seat] + bounty * (res_m["final_price"] - 0.5)
        ),
        "manip_fill_rate": (
            float(res_m["volume"][:, seat].sum() / sub) if sub > 0 else 1.0
        ),
        "honest_pnl_total": wm(pnl_m[:, honest_idx].sum(axis=1)),
        "honest_pnl_total_honest_run": wm(pnl_h[:, honest_idx].sum(axis=1)),
        "decision_acc": sm["decision_acc"],
        "decision_acc_honest_run": summarize(env_k, res_h)["decision_acc"],
        "log_loss_final": sm["log_loss_final"],
        "jams": int(res_m["jams"]),
    }


def manip(m_reps: int) -> None:
    records = []
    for env_name, ns, bounties in [
        ("gaussian", [3, 10], GAUSS_BOUNTIES),
        ("galanis", [3], GALANIS_BOUNTIES),
    ]:
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            for r in [1, 3]:
                res_h_cache = {}
                for bounty in bounties:
                    # SEQ: manipulator rotated through every seat
                    for seat in range(n):
                        records.append(
                            _manip_row(env, env_name, n, r, "seq_lmsr",
                                       seat, bounty, None, 1.0, res_h_cache)
                        )
                    # batch market-order baseline (price disclosure)
                    for seat in (0, n - 1):
                        records.append(
                            _manip_row(env, env_name, n, r, "batch_lmsr",
                                       seat, bounty, None, 1.0, res_h_cache)
                        )
                    # batch limit arms: manipulator best-responds over gamma
                    for slack in MANIP_SLACKS:
                        grid = [
                            _manip_row(env, env_name, n, r, "batch_lmsr_limit",
                                       0, bounty, slack, g, res_h_cache)
                            for g in GAMMAS
                        ]
                        best = max(grid, key=lambda rec: rec["manip_utility"])
                        spot = _manip_row(env, env_name, n, r, "batch_lmsr_limit",
                                          n - 1, bounty, slack, best["gamma"],
                                          res_h_cache)
                        rec = dict(best)
                        rec["gamma_grid"] = [
                            {"gamma": g["gamma"],
                             "manip_utility": g["manip_utility"],
                             "delta_final_price": g["delta_final_price"],
                             "manip_cost": g["manip_cost"],
                             "manip_fill_rate": g["manip_fill_rate"]}
                            for g in grid
                        ]
                        rec["seat_invariance_dpf_gap"] = abs(
                            best["delta_final_price"] - spot["delta_final_price"]
                        )
                        records.append(rec)
                print(f"manip {env_name} N={n} R={r} done", flush=True)
    _dump("limits_manip.json", records)


def jam(m_reps: int) -> None:
    del m_reps  # Galanis is an exact 8-state enumeration
    records = []
    env = GalanisEnv()
    n, r = 3, 3
    res_h_cache = {}
    for bounty in GALANIS_BOUNTIES:
        for seat in (0,):
            records.append(
                _manip_row(env, "galanis", n, r, "batch_lmsr",
                           seat, bounty, None, 1.0, res_h_cache)
            )
        for slack in SLACKS:
            for gamma in GAMMAS:
                records.append(
                    _manip_row(env, "galanis", n, r, "batch_lmsr_limit",
                               0, bounty, slack, gamma, res_h_cache)
                )
        print(f"jam galanis B={bounty} done", flush=True)
    _dump("limits_jam.json", records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small M for smoke runs")
    ap.add_argument("--only", choices=["core", "manip", "jam"], default=None)
    args = ap.parse_args()
    m = 2000 if args.fast else 20000
    t0 = time.time()
    if args.only in (None, "core"):
        core(m)
    if args.only in (None, "manip"):
        manip(m)
    if args.only in (None, "jam"):
        jam(m)
    print(f"all limit sweeps done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
