"""Run all batch-amm sweeps and write results/*.json.

Sweeps:
  core   — honest population, N x R x mechanism x sizing, both envs (CRN).
  manip  — one bribed trader (bounty * (p_final - 0.5)), bounty sweep,
           per-seat in SEQ (manipulator = original trader 0 rotated through
           seats with the same draws), seat-0 in BATCH (+ invariance check).
  seats  — honest ordering value in SEQ: rotation-averaged per-seat PnL.

Usage: python scripts/run_sweeps.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from batch_amm.engine import Config, run_market
from batch_amm.envs import GalanisEnv, GaussianEnv
from batch_amm.metrics import paired_diff, summarize

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SEED = 20260704
N_SWEEP = [3, 5, 10, 25]
R_SWEEP = [1, 3]
ARMS = [
    ("seq_lmsr", "full"),  # sizing ignored for seq
    ("batch_lmsr", "full"),
    ("batch_lmsr", "competitive"),
    ("batch_kyle", "full"),
    ("batch_kyle", "competitive"),
]
GAUSS_BOUNTIES = [0.05, 0.15, 0.5]
GALANIS_BOUNTIES = [0.02, 0.05, 0.2]
B_LIQ = 0.1          # Gaussian env
B_LIQ_GALANIS = 0.01  # matches galanis-market CFR+ setup (MANIPULATION.md)


def _b(env_name: str) -> float:
    return B_LIQ if env_name == "gaussian" else B_LIQ_GALANIS


def _dump(name: str, obj) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}")


def _gauss_env(n: int, m: int) -> GaussianEnv:
    return GaussianEnv(n=n, m=m, sigma_eps=1.0, seed=SEED + n)  # CRN per N


def core(m_reps: int) -> None:
    records = []
    for env_name in ("gaussian", "galanis"):
        ns = N_SWEEP if env_name == "gaussian" else [3]
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            for r in R_SWEEP:
                seq_res = None
                for mech, sizing in ARMS:
                    cfg = Config(mech=mech, rounds=r, b=_b(env_name), sizing=sizing)
                    res = run_market(env, cfg)
                    if mech == "seq_lmsr":
                        seq_res = res
                    s = summarize(env, res)
                    s["round_price_mean_abs_logit"] = [
                        float(np.mean(np.abs(np.log(p / (1 - p)))))
                        for p in np.clip(res["round_prices"], 1e-4, 1 - 1e-4)
                    ]
                    rec = {
                        "env": env_name,
                        "N": n,
                        "R": r,
                        "mech": mech,
                        "sizing": sizing,
                        "summary": s,
                    }
                    if mech != "seq_lmsr":
                        rec["paired_vs_seq"] = paired_diff(env, res, seq_res)
                    records.append(rec)
                print(f"core {env_name} N={n} R={r} done", flush=True)
    _dump("core.json", records)


def manip(m_reps: int) -> None:
    records = []

    def run_manip(env_name, env, n, r, mech, sizing, seat, bounty):
        """Manipulator = original trader 0, seated at `seat` (SEQ rotation)."""
        perm = [(j - seat) % n for j in range(n)]
        env_k = env.with_trader_order(perm)
        cfg_h = Config(mech=mech, rounds=r, b=_b(env_name), sizing=sizing)
        cfg_m = Config(
            mech=mech, rounds=r, b=_b(env_name), sizing=sizing,
            manip_seat=seat, bounty=bounty,
        )
        res_h = run_market(env_k, cfg_h)
        res_m = run_market(env_k, cfg_m)
        w = env.weights
        payout = env.payout
        pnl_h = res_h["holdings"] * payout[:, None] + res_h["cash"]
        pnl_m = res_m["holdings"] * payout[:, None] + res_m["cash"]
        honest_idx = [j for j in range(n) if j != seat]

        def wm(x):
            return float((x * w).sum() / w.sum())

        def wse(x):
            if env.exact:
                return 0.0
            mu = wm(x)
            return float(np.sqrt(wm((x - mu) ** 2) / len(x)))

        d_pf = res_m["final_price"] - res_h["final_price"]
        sm = summarize(env_k, res_m)
        return {
            "env": env_name, "N": n, "R": r, "mech": mech, "sizing": sizing,
            "seat": seat, "bounty": bounty,
            "delta_final_price": wm(d_pf),
            "delta_final_price_se": wse(d_pf),
            "manip_market_pnl": wm(pnl_m[:, seat]),
            "manip_market_pnl_honest_run": wm(pnl_h[:, seat]),
            "manip_bounty_received": wm(bounty * (res_m["final_price"] - 0.5)),
            "honest_pnl_total": wm(pnl_m[:, honest_idx].sum(axis=1)),
            "honest_pnl_total_honest_run": wm(pnl_h[:, honest_idx].sum(axis=1)),
            "mm_loss": sm["mm_loss"],
            "decision_acc": sm["decision_acc"],
            "decision_acc_honest_run": summarize(env_k, res_h)["decision_acc"],
            "log_loss_final": sm["log_loss_final"],
            "log_loss_final_honest_run": summarize(env_k, res_h)["log_loss_final"],
        }

    for env_name, ns, bounties in [
        ("gaussian", [3, 10], GAUSS_BOUNTIES),
        ("galanis", [3], GALANIS_BOUNTIES),
    ]:
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            for r in R_SWEEP:
                for bounty in bounties:
                    # SEQ: manipulator rotated through every seat
                    for seat in range(n):
                        records.append(
                            run_manip(env_name, env, n, r, "seq_lmsr", "full", seat, bounty)
                        )
                    # BATCH arms: seat 0 (invariance verified in unit tests;
                    # also spot-check last seat here)
                    for mech in ("batch_lmsr", "batch_kyle"):
                        for seat in (0, n - 1):
                            records.append(
                                run_manip(env_name, env, n, r, mech, "competitive", seat, bounty)
                            )
                print(f"manip {env_name} N={n} R={r} done", flush=True)
    _dump("manip.json", records)


def seats(m_reps: int) -> None:
    """Honest ordering value: rotation-averaged per-seat PnL in SEQ."""
    records = []
    for env_name, ns in [("gaussian", N_SWEEP), ("galanis", [3])]:
        for n in ns:
            env = _gauss_env(n, m_reps) if env_name == "gaussian" else GalanisEnv()
            w = env.weights
            payout = env.payout
            for r in R_SWEEP:
                for mech, sizing in [("seq_lmsr", "full"), ("batch_lmsr", "competitive")]:
                    # rotate the same trader population through all seats
                    seat_pnl = np.zeros(n)
                    seat_pnl_sq = np.zeros(n)
                    seat_leak = np.zeros(n)
                    total_samples = 0
                    for k in range(n):
                        perm = [(j + k) % n for j in range(n)]
                        env_k = env.with_trader_order(perm)
                        res = run_market(
                            env_k, Config(mech=mech, rounds=r, b=_b(env_name), sizing=sizing)
                        )
                        pnl = res["holdings"] * payout[:, None] + res["cash"]
                        for seat in range(n):
                            seat_pnl[seat] += (pnl[:, seat] * w).sum()
                            seat_pnl_sq[seat] += (pnl[:, seat] ** 2 * w).sum()
                            seat_leak[seat] += (res["slip_own"][:, seat] * w).sum()
                        total_samples += 1
                    seat_pnl /= total_samples
                    seat_leak /= total_samples
                    var = seat_pnl_sq / total_samples - seat_pnl**2
                    se = (
                        np.zeros(n)
                        if env.exact
                        else np.sqrt(var / (env.m * total_samples))
                    )
                    records.append(
                        {
                            "env": env_name, "N": n, "R": r,
                            "mech": mech, "sizing": sizing,
                            "seat_pnl": seat_pnl.tolist(),
                            "seat_pnl_se": se.tolist(),
                            "seat_leak": seat_leak.tolist(),
                            "seat_spread": float(seat_pnl.max() - seat_pnl.min()),
                        }
                    )
                print(f"seats {env_name} N={n} R={r} done", flush=True)
    _dump("seats.json", records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small M for smoke runs")
    ap.add_argument("--only", choices=["core", "manip", "seats"], default=None)
    args = ap.parse_args()
    m = 2000 if args.fast else 20000
    t0 = time.time()
    if args.only in (None, "core"):
        core(m)
    if args.only in (None, "manip"):
        manip(m)
    if args.only in (None, "seats"):
        seats(m)
    print(f"all sweeps done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
