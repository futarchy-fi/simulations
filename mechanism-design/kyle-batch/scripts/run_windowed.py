#!/usr/bin/env python
"""Windowed-TWAP sweep (Q4 extension): settlement statistic = mean of the
LAST K batch prices, K in {1, 2, 4, T}, T in {4, 8, 16}, covert uninformed
open-loop manipulator (same threat model as Q4).

Per (T, B): optimal K* and the per-K decomposition
    baseline_cost(K)      = DQ(K, B=0) - DQ(K=1, B=0)   (early-price memory)
    damage(K, B)          = DQ(K, 0) - DQ(K, B)          (manipulation damage)
    damage_reduction(K,B) = damage(1, B) - damage(K, B)
plus the concealed-window variant: manipulator best-responds to a UNIFORM-
RANDOM K in {1,2,4} drawn after trading, vs knowing K.

Writes results/twap_windowed.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyle_batch import twap as tw

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASE = dict(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3)
BS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def run() -> None:
    t0 = time.time()
    out = {"config": {**BASE, "B_grid": BS, "manip": "covert uninformed open-loop",
                      "statistic": "mean of last K batch prices"},
           "known_K": [], "concealed": [], "mc_checks": []}

    for T in (4, 8, 16):
        p = tw.TwapParams(**BASE, T=T)
        dyn = tw.solve_honest_dynamics(p)
        pr = tw.push_response(dyn)
        Ks = sorted({1, 2, 4, T})
        stats = {K: f"win:{K}" for K in Ks}

        # ---- known-K sweep ------------------------------------------------
        dq0 = {}
        for K in Ks:
            ev0 = tw.evaluate(dyn, np.zeros(T), stats[K])
            dq0[K] = ev0["decision_quality"]
        for K in Ks:
            for B in BS:
                if B == 0.0:
                    al = np.zeros(T)
                    ev = tw.evaluate(dyn, al, stats[K])
                else:
                    al = tw.solve_manipulator_fast(pr, B=B, statistics=[stats[K]])
                    ev = tw.evaluate(dyn, al, stats[K])
                out["known_K"].append({
                    "T": T, "K": K, "B": B,
                    "dq": ev["decision_quality"],
                    "damage": dq0[K] - ev["decision_quality"],
                    "baseline_cost_vs_K1": dq0[K] - dq0[1],
                    "stat_bias": ev["stat_bias"], "corr_Pv": ev["corr_Pv"],
                    "approval": ev["approval_prob"],
                    "alphas": list(al),
                    "manip_trading_pnl": ev["manip_trading_pnl"],
                })
            print(f"  T={T} K={K} done ({time.time()-t0:.0f}s)")

        # ---- concealed window: uniform K in {1,2,4} drawn after trading ---
        cstats = [stats[K] for K in (1, 2, 4)]
        for B in BS:
            if B == 0.0:
                al_c = np.zeros(T)
            else:
                al_c = tw.solve_manipulator_fast(pr, B=B, statistics=cstats)
            mix_c = tw.evaluate_mixture(dyn, al_c, cstats, B=B)
            # known-K comparison: same 1/3 mixture but manipulator tailored per K
            dq_known, u_known, per_known = 0.0, 0.0, {}
            for s in cstats:
                al_k = (np.zeros(T) if B == 0.0
                        else tw.solve_manipulator_fast(pr, B=B, statistics=[s]))
                ev_k = tw.evaluate(dyn, al_k, s)
                per_known[s] = ev_k["decision_quality"]
                dq_known += ev_k["decision_quality"] / 3.0
                u_known += (ev_k["manip_trading_pnl"]
                            + B * ev_k["approval_prob"]) / 3.0
            out["concealed"].append({
                "T": T, "B": B,
                "dq_concealed_mix": mix_c["decision_quality_mix"],
                "dq_known_mix": dq_known,
                "concealment_gain_dq": mix_c["decision_quality_mix"] - dq_known,
                "dq_concealed_per_K": {s: mix_c["per_statistic"][s]["decision_quality"]
                                       for s in cstats},
                "dq_known_per_K": per_known,
                "manip_value_concealed": mix_c["manip_total"],
                "manip_value_known": u_known,
                "alphas_concealed": list(al_c),
            })
        print(f"  T={T} concealed done ({time.time()-t0:.0f}s)")

        # ---- MC verification at B=2, K=2 ----------------------------------
        al = tw.solve_manipulator_fast(pr, B=2.0, statistics=[stats[2]])
        ana = tw.evaluate(dyn, al, stats[2])
        mc = tw.mc_check(dyn, al, stats[2], n=400_000, seed=17)
        out["mc_checks"].append({
            "T": T, "K": 2, "B": 2.0,
            "analytic_dq": ana["decision_quality"], "mc_dq": mc["decision_quality"],
            "mc_dq_se": mc["decision_quality_se"],
            "analytic_bias": ana["stat_bias"], "mc_bias": mc["stat_bias"],
            "dq_agree_4se": abs(ana["decision_quality"] - mc["decision_quality"])
                            < 4 * mc["decision_quality_se"],
        })

    # ---- K*(T, B) table ----------------------------------------------------
    kstar = []
    for T in (4, 8, 16):
        for B in BS:
            rows = [r for r in out["known_K"] if r["T"] == T and r["B"] == B]
            best = max(rows, key=lambda r: r["dq"])
            kstar.append({"T": T, "B": B, "K_star": best["K"], "dq": best["dq"],
                          "dq_by_K": {r["K"]: r["dq"] for r in rows}})
    out["K_star"] = kstar

    path = RESULTS / "twap_windowed.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
