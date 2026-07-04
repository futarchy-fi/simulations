#!/usr/bin/env python
"""Q7 sweep: windowed TWAP under in-window information arrival.

Variants:
  (b) public signal stream z_t = v + eta_t (t = 2..T), MM conditions on it;
      arrival fraction phi in {0, .25, .5, .75} of a FIXED total fundamental
      precision Pi = 3 (phi=0 == the static Q4b model exactly).
  (a) staggered private signals: N=4 traders, n_late = phi*N signals arrive
      spread over batches 2..T (same fixed total precision).

Per (variant, T, phi): push half-life; per (K, B): covert uninformed
open-loop manipulator's EXACT best response (BFGS + exact gradient) against
the win:K statistic -> dq, damage, statistic bias, manipulator cost; named
restricted schedules (re-time-to-window-start / sustained / last-batch) for
the narrative; concealed uniform-K in {1,2,4} mixture (Q4b defence); K*(T,B)
frontier per phi; MC verification at (B=2, K=2).

Writes results/arrival.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyle_batch import arrival as ar
from kyle_batch.twap import solve_manipulator_fast

RESULTS = Path(__file__).resolve().parents[1] / "results"
BS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
PHIS = [0.0, 0.25, 0.5, 0.75]
TS = [4, 8]
SCHED_BS = [2.0, 5.0, 20.0]


def sweep_variant(variant: str, out: dict, t0: float) -> None:
    for T in TS:
        for phi in PHIS:
            if variant == "public":
                p = ar.ArrivalParams(T=T, phi=phi)
                dyn = ar.solve_honest_dynamics_arrival(p)
                extra = {"sigma_eps": p.sigma_eps,
                         "sigma_z": p.sigma_z if phi > 0 else None}
            else:
                times = ar.staggered_times(4, T, phi)
                p = ar.StaggeredParams(N=4, T=T, arrival=times)
                dyn = ar.solve_honest_dynamics_staggered(p)
                extra = {"arrival_times": list(times), "sigma_eps": p.sigma_eps}
            pr = ar.push_response(dyn)
            key = {"variant": variant, "T": T, "phi": phi}

            dec = ar.push_decay(pr, push_batch=0)
            out["half_life"].append({**key, **extra, **dec})

            Ks = list(range(1, T + 1))
            dq0 = {K: ar.evaluate(dyn, np.zeros(T), f"win:{K}")["decision_quality"]
                   for K in Ks}
            for K in Ks:
                stat = f"win:{K}"
                for B in BS:
                    al = (np.zeros(T) if B == 0.0
                          else solve_manipulator_fast(pr, B=B, statistics=[stat]))
                    ev = ar.evaluate(dyn, al, stat)
                    out["known_K"].append({
                        **key, "K": K, "B": B,
                        "dq": ev["decision_quality"],
                        "damage": dq0[K] - ev["decision_quality"],
                        "baseline_cost_vs_K1": dq0[K] - dq0[1],
                        "stat_bias": ev["stat_bias"], "corr_Pv": ev["corr_Pv"],
                        "approval": ev["approval_prob"],
                        "alphas": list(al),
                        "manip_cost": -ev["manip_trading_pnl"],
                        "manip_total": ev["manip_trading_pnl"]
                                       + B * ev["approval_prob"],
                    })

            # named restricted schedules vs the exact BR (narrative table)
            for B in SCHED_BS:
                for K in (2, 4):
                    stat = f"win:{K}"
                    al_star = solve_manipulator_fast(pr, B=B, statistics=[stat])
                    ev_star = ar.evaluate(dyn, al_star, stat)
                    row = {**key, "K": K, "B": B,
                           "exact": {"u": ev_star["manip_trading_pnl"]
                                          + B * ev_star["approval_prob"],
                                     "dq": ev_star["decision_quality"],
                                     "alphas": list(al_star)}}
                    for name, shape in ar.schedule_shapes(T, K).items():
                        al_s, u_s = ar.solve_scalar_schedule(pr, B=B,
                                                             statistic=stat,
                                                             shape=shape)
                        ev_s = ar.evaluate(dyn, al_s, stat)
                        row[name] = {"u": u_s, "dq": ev_s["decision_quality"],
                                     "push_size": float(al_s.sum())}
                    out["schedules"].append(row)

            # concealed uniform-K in {1,2,4} (Q4b randomized read)
            cstats = ["win:1", "win:2", "win:4"]
            for B in BS:
                al_c = (np.zeros(T) if B == 0.0
                        else solve_manipulator_fast(pr, B=B, statistics=cstats))
                mix = ar.evaluate_mixture(dyn, al_c, cstats, B=B)
                out["concealed"].append({
                    **key, "B": B,
                    "dq_concealed_mix": mix["decision_quality_mix"],
                    "alphas": list(al_c),
                })

            # MC verification
            al = solve_manipulator_fast(pr, B=2.0, statistics=["win:2"])
            ana = ar.evaluate(dyn, al, "win:2")
            mc = (ar.mc_check_arrival if variant == "public"
                  else ar.mc_check_staggered)(dyn, al, "win:2",
                                              n=400_000, seed=17)
            out["mc_checks"].append({
                **key, "K": 2, "B": 2.0,
                "analytic_dq": ana["decision_quality"],
                "mc_dq": mc["decision_quality"],
                "mc_dq_se": mc["decision_quality_se"],
                "analytic_bias": ana["stat_bias"], "mc_bias": mc["stat_bias"],
                "dq_agree_4se": abs(ana["decision_quality"]
                                    - mc["decision_quality"])
                                < 4 * mc["decision_quality_se"],
            })
            print(f"  {variant} T={T} phi={phi} done ({time.time()-t0:.0f}s)",
                  flush=True)

    # K*(variant, T, phi, B)
    for T in TS:
        for phi in PHIS:
            for B in BS:
                rows = [r for r in out["known_K"]
                        if r["variant"] == variant and r["T"] == T
                        and r["phi"] == phi and r["B"] == B]
                best = max(rows, key=lambda r: r["dq"])
                out["K_star"].append({
                    "variant": variant, "T": T, "phi": phi, "B": B,
                    "K_star": best["K"], "dq": best["dq"],
                    "dq_by_K": {r["K"]: r["dq"] for r in rows}})


def sweep_buffer(out: dict, t0: float) -> None:
    """Staggered arrival AT vs AWAY FROM the read batch (T=8): the K*>1
    effect is driven by fresh private info arriving at/near the read; a
    settlement buffer (arrivals end a few batches before the read) is the
    cheaper alternative to averaging."""
    cases = [("at-read", (1, 1, 2, 8)), ("buffer-2", (1, 1, 3, 6)),
             ("at-read", (1, 2, 5, 8)), ("buffer-2", (1, 2, 4, 6))]
    for label, times in cases:
        p = ar.StaggeredParams(N=4, T=8, arrival=times)
        dyn = ar.solve_honest_dynamics_staggered(p)
        pr = ar.push_response(dyn)
        dqs = {}
        for K in range(1, 9):
            stat = f"win:{K}"
            dq0 = ar.evaluate(dyn, np.zeros(8), stat)["decision_quality"]
            dqs[K] = {"dq0": dq0}
            for B in (5.0, 10.0, 20.0):
                al = solve_manipulator_fast(pr, B=B, statistics=[stat])
                dqs[K][B] = ar.evaluate(dyn, al, stat)["decision_quality"]
        for B in (5.0, 10.0, 20.0):
            best = max(dqs, key=lambda K: dqs[K][B])
            out["buffer"].append({
                "label": label, "arrival_times": list(times), "B": B,
                "K_star": best, "dq_Kstar": dqs[best][B],
                "dq_K1": dqs[1][B], "damage_K1": dqs[1]["dq0"] - dqs[1][B],
                "lams": [float(l) for l in dyn["lams"]],
            })
        print(f"  buffer {times} done ({time.time()-t0:.0f}s)", flush=True)


def run() -> None:
    t0 = time.time()
    out = {"config": {"B_grid": BS, "phi_grid": PHIS, "T_grid": TS, "Pi": 3.0,
                      "tau": 0.3, "sigma_u": 1.0,
                      "manip": "covert uninformed open-loop (exact BR)",
                      "public": "N=3; z_t = v + eta_t at start of t=2..T",
                      "staggered": "N=4; n_late=phi*N signals spread over 2..T"},
           "half_life": [], "known_K": [], "schedules": [], "concealed": [],
           "mc_checks": [], "K_star": [], "buffer": []}
    sweep_variant("public", out, t0)
    sweep_variant("staggered", out, t0)
    sweep_buffer(out, t0)
    path = RESULTS / "arrival.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
