"""Disclosure-regime sweeps (pseudo-anonymity): writes results/disclosure.json
and results/disclosure_manip.json.

Regimes between batch rounds:
  full      — per-trader orders (the original assumption)
  aggregate — clearing price + net flow only (Kelvin's pseudo-anonymity)
  price     — clearing price only (== aggregate against a deterministic AMM;
              swept anyway as a plumbing check)

Sweeps (common random numbers across regimes and vs the SEQ benchmark):
  aggregation — honest, N x R in {2,3,5} x {batch_lmsr, batch_kyle} x regime,
                competitive sizing; SEQ benchmark rows; full-sizing R=5 rows
                for the oscillation/damping check.
  manip       — one bribed trader, bounty sweep, batch arms under full vs
                aggregate, first/last seat (invariance spot check).

Usage: python scripts/run_disclosure.py [--fast]
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
R_SWEEP = [2, 3, 5]
REGIMES = ["full", "aggregate", "price"]
GAUSS_BOUNTIES = [0.05, 0.15, 0.5]
GALANIS_BOUNTIES = [0.02, 0.05, 0.2]


def _b(env_name: str) -> float:
    return 0.1 if env_name == "gaussian" else 0.01


def _gauss_env(n: int, m: int) -> GaussianEnv:
    return GaussianEnv(n=n, m=m, sigma_eps=1.0, seed=SEED + n)


def _dump(name: str, obj) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}")


def aggregation(m_reps: int) -> None:
    records = []
    for env_name in ("gaussian", "galanis"):
        ns = N_SWEEP if env_name == "gaussian" else [3]
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            b = _b(env_name)
            seq = run_market(env, Config(mech="seq_lmsr", rounds=1, b=b))
            s = summarize(env, seq)
            records.append(
                {"env": env_name, "N": n, "R": 1, "mech": "seq_lmsr",
                 "sizing": "full", "disclosure": "n/a", "summary": s}
            )
            for r in R_SWEEP:
                for mech in ("batch_lmsr", "batch_kyle"):
                    res_by_regime = {}
                    for regime in REGIMES:
                        cfg = Config(
                            mech=mech, rounds=r, b=b, sizing="competitive",
                            disclosure=regime,
                        )
                        res = run_market(env, cfg)
                        res_by_regime[regime] = res
                        rec = {
                            "env": env_name, "N": n, "R": r, "mech": mech,
                            "sizing": "competitive", "disclosure": regime,
                            "summary": summarize(env, res),
                            "paired_vs_seq": paired_diff(env, res, seq),
                        }
                        if regime != "full":
                            rec["paired_vs_full_disclosure"] = paired_diff(
                                env, res, res_by_regime["full"]
                            )
                        records.append(rec)
                # oscillation / damping check under anonymity (full sizing)
                if r == 5:
                    for regime in ("full", "aggregate"):
                        res = run_market(
                            env,
                            Config(mech="batch_lmsr", rounds=r, b=b,
                                   sizing="full", disclosure=regime),
                        )
                        s = summarize(env, res)
                        s["round_price_mean_abs_logit"] = [
                            float(np.mean(np.abs(np.log(pp / (1 - pp)))))
                            for pp in np.clip(res["round_prices"], 1e-4, 1 - 1e-4)
                        ]
                        records.append(
                            {"env": env_name, "N": n, "R": r,
                             "mech": "batch_lmsr", "sizing": "full",
                             "disclosure": regime, "summary": s}
                        )
                print(f"agg {env_name} N={n} R={r} done", flush=True)
    _dump("disclosure.json", records)


def manip(m_reps: int) -> None:
    records = []
    for env_name, ns, bounties, r_list in [
        ("gaussian", [3, 10], GAUSS_BOUNTIES, [3]),
        ("galanis", [3], GALANIS_BOUNTIES, [1, 3]),
    ]:
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            b = _b(env_name)
            w = env.weights
            payout = env.payout

            def wm(x):
                return float((x * w).sum() / w.sum())

            def wse(x):
                if env.exact:
                    return 0.0
                mu = wm(x)
                return float(np.sqrt(wm((x - mu) ** 2) / len(x)))

            for r in r_list:
                for mech in ("batch_lmsr", "batch_kyle"):
                    for regime in ("full", "aggregate"):
                        for bounty in bounties:
                            # manipulator = original trader 0 rotated through
                            # seats with the same draws, so the two seat rows
                            # must coincide exactly (batch seat-invariance)
                            for seat in (0, n - 1):
                                perm = [(j - seat) % n for j in range(n)]
                                env_k = env.with_trader_order(perm)
                                cfg_h = Config(mech=mech, rounds=r, b=b,
                                               sizing="competitive",
                                               disclosure=regime)
                                res_h = run_market(env_k, cfg_h)
                                pnl_h = (
                                    res_h["holdings"] * payout[:, None]
                                    + res_h["cash"]
                                )
                                cfg_m = Config(
                                    mech=mech, rounds=r, b=b, sizing="competitive",
                                    disclosure=regime, manip_seat=seat, bounty=bounty,
                                )
                                res_m = run_market(env_k, cfg_m)
                                pnl_m = (
                                    res_m["holdings"] * payout[:, None] + res_m["cash"]
                                )
                                honest_idx = [j for j in range(n) if j != seat]
                                d_pf = res_m["final_price"] - res_h["final_price"]
                                sm = summarize(env, res_m)
                                records.append(
                                    {
                                        "env": env_name, "N": n, "R": r,
                                        "mech": mech, "disclosure": regime,
                                        "seat": seat, "bounty": bounty,
                                        "delta_final_price": wm(d_pf),
                                        "delta_final_price_se": wse(d_pf),
                                        "manip_market_pnl": wm(pnl_m[:, seat]),
                                        "manip_market_pnl_honest_run": wm(pnl_h[:, seat]),
                                        "honest_pnl_total": wm(
                                            pnl_m[:, honest_idx].sum(axis=1)
                                        ),
                                        "decision_acc": sm["decision_acc"],
                                        "decision_acc_honest_run": summarize(env, res_h)[
                                            "decision_acc"
                                        ],
                                        "log_loss_final": sm["log_loss_final"],
                                    }
                                )
                print(f"manip {env_name} N={n} R={r} done", flush=True)
    _dump("disclosure_manip.json", records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    m = 2000 if args.fast else 20000
    t0 = time.time()
    aggregation(m)
    manip(m)
    print(f"disclosure sweeps done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
