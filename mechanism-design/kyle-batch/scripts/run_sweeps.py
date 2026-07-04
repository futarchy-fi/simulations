#!/usr/bin/env python
"""Run all kyle-batch sweeps and write results/*.json.

Usage: run_sweeps.py [baseline|corruption|frontier|entry|twap|amm|all]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyle_batch.closed_forms import baseline
from kyle_batch.decision import (E_q, E_qprime, decision_quality, ORACLE_DQ,
                                 oracle_q_dq)
from kyle_batch.onebatch import (Params, solve_equilibrium, metrics,
                                 metrics_mixture, bayes_mm_metrics,
                                 bayes_manip_fixed_point)
from kyle_batch.mc import simulate, deviation_report, sup_deviation_manip, sup_deviation_honest
from kyle_batch import twap as tw

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

BASE = dict(N=3, sigma_eps=1.0, sigma_u=1.0, tau=0.3)


def dump(name: str, obj) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {path}")


# ---------------------------------------------------------------- baseline
def run_baseline():
    rows = []
    for N in (1, 2, 3, 5, 10, 20):
        for se in (0.5, 1.0, 2.0):
            for su in (0.5, 1.0, 2.0):
                b = baseline(N, se, su)
                for tau in (0.1, 0.3, 1.0):
                    dq = decision_quality(b.var_p, 0.0, b.var_p, tau)
                    rows.append({
                        "N": N, "sigma_eps": se, "sigma_u": su, "tau": tau,
                        "lam": b.lam, "beta": b.beta, "corr_pv": b.corr_pv,
                        "dq_logistic": dq,
                        "dq_hard_threshold": b.corr_pv * ORACLE_DQ,
                        "dq_oracle_hard": ORACLE_DQ,
                        "dq_oracle_logistic": oracle_q_dq(tau),
                        "informed_profit_each": b.informed_profit_each,
                        "informed_profit_total": b.informed_profit_total,
                    })
    # MC spot checks
    checks = []
    for (N, se, su) in [(3, 1.0, 1.0), (5, 0.5, 2.0), (1, 0.0, 1.0)]:
        p = Params(N=N, sigma_eps=se, sigma_u=su, tau=0.3, manip="none")
        prof = solve_equilibrium(p)
        ana = metrics(prof, p, False)
        mc = simulate(prof, p, False, n=2_000_000, seed=42)
        checks.append({"N": N, "sigma_eps": se, "sigma_u": su,
                       "analytic": {k: ana[k] for k in
                                    ("corr_pv", "decision_quality", "profit_honest_each")},
                       "mc": {k: mc[k] for k in
                              ("corr_pv", "decision_quality", "decision_quality_se",
                               "profit_honest_each", "profit_honest_each_se")}})
    dump("baseline.json", {"rows": rows, "mc_checks": checks})


# -------------------------------------------------------------- corruption
def corruption_row(p: Params, ref0: dict | None, deviations: bool = True) -> dict:
    prof = solve_equilibrium(p)
    m = metrics_mixture(prof, p)
    m1, m0 = m["present"], m["absent"]
    row = {
        "B": p.B, "rho": p.rho, "tau": p.tau, "sigma_u": p.sigma_u,
        "manip": p.manip,
        "a_m": prof.a_m, "b_m": prof.b_m, "b_h": prof.b_h, "lam": prof.lam,
        "mu": prof.mu,
        "bias_present": m1["price_bias"], "bias_absent": m0["price_bias"],
        "corr_pv_present": m1["corr_pv"],
        "dq_present": m1["decision_quality"], "dq_absent": m0["decision_quality"],
        "dq_mix": m["decision_quality_mix"],
        "approval_present": m1["approval_prob"],
        "manip_trading_pnl": m1.get("manip_trading_pnl"),
        "manip_bounty": m1.get("manip_bounty"),
        "profit_honest_each_present": m1["profit_honest_each"],
        # linear-quadratic (probit / small-B) closed-form prediction
        "a_m_pred_lq": p.B * E_qprime(m1["price_bias"], np.sqrt(m1["var_p"]), p.tau) / (2 - p.rho),
        "bias_pred_lq": prof.lam * (1 - p.rho) * prof.a_m,
    }
    if ref0 is not None:
        row["d_dq_present"] = row["dq_present"] - ref0["dq_present"]
        row["d_dq_mix"] = row["dq_mix"] - ref0["dq_mix"]
        row["d_manip_trading"] = (row["manip_trading_pnl"] or 0) - (ref0["manip_trading_pnl"] or 0)
        row["d_profit_honest"] = (row["profit_honest_each_present"]
                                  - ref0["profit_honest_each_present"])
    if deviations:
        row["dev_honest_sup"] = sup_deviation_honest(prof, p)
        row["dev_manip_sup"] = sup_deviation_manip(prof, p)["gain"]
    return row


def run_corruption():
    out = {"B_sweep": [], "tau_sweep": [], "grid_deviation_spots": []}
    Bs = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    for rho in (1.0, 0.5, 0.0):
        ref0 = None
        for B in Bs:
            p = Params(**BASE, B=B, manip="informed", rho=max(rho, 1e-12))
            row = corruption_row(p, ref0)
            if B == 0.0:
                ref0 = row
            out["B_sweep"].append(row)
            print(f"  corruption rho={rho} B={B}: bias={row['bias_present']:+.4f} "
                  f"dDQ={row.get('d_dq_present', 0):+.5f}")
    for tau in (0.1, 0.3, 1.0, 3.0):
        for rho in (1.0, 0.5, 0.0):
            base_kw = dict(BASE)
            base_kw["tau"] = tau
            ref0 = corruption_row(Params(**base_kw, B=0.0, manip="informed",
                                         rho=max(rho, 1e-12)), None, deviations=False)
            row = corruption_row(Params(**base_kw, B=1.0, manip="informed",
                                        rho=max(rho, 1e-12)), ref0, deviations=False)
            out["tau_sweep"].append(row)
    # heavier grid-MC deviation certificates at a few spots
    for rho, B in [(1.0, 1.0), (0.5, 1.0), (0.0, 2.0)]:
        p = Params(**BASE, B=B, manip="informed", rho=max(rho, 1e-12))
        prof = solve_equilibrium(p)
        rep = deviation_report(prof, p, n=400_000)
        rep_small = {
            "rho": rho, "B": B,
            "honest_sup": rep["honest_sup_gain"],
            "honest_grid_max_gain": rep["honest_grid"]["max_gain"],
            "honest_grid_se": rep["honest_grid"]["se"],
            "manip_sup_gain": rep["manip_sup"]["gain"],
            "manip_eq_value": rep["manip_sup"]["eq_value"],
            "manip_grid_max_gain": rep["manip_grid"]["max_gain"],
            "manip_grid_se": rep["manip_grid"]["se"],
        }
        out["grid_deviation_spots"].append(rep_small)
    dump("corruption.json", out)


# ---------------------------------------------------------------- frontier
def run_frontier():
    """sigma_u x B; covert manipulator. Linear MM (rho=0.5 and rho=0) and the
    exact Bayesian mixture MM (rho=0.5) for the camouflage leg."""
    out = []
    sus = [0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0]
    for su in sus:
        for B in (0.5, 1.0, 2.0, 5.0):
            for rho in (0.5, 0.0):
                kw = dict(BASE)
                kw["sigma_u"] = su
                p0 = Params(**kw, B=0.0, manip="informed", rho=max(rho, 1e-12))
                pB = Params(**kw, B=B, manip="informed", rho=max(rho, 1e-12))
                prof0, profB = solve_equilibrium(p0), solve_equilibrium(pB)
                m0, mB = metrics(prof0, p0, True), metrics(profB, pB, True)
                row = {
                    "sigma_u": su, "B": B, "rho": rho, "mm": "linear",
                    "lam": profB.lam,
                    "subsidy": profB.lam * su**2,   # noise traders' expected loss
                    "bias": mB["price_bias"],
                    "dq0": m0["decision_quality"], "dq": mB["decision_quality"],
                    "d_dq": mB["decision_quality"] - m0["decision_quality"],
                    "d_approval": mB["approval_prob"] - m0["approval_prob"],
                    "manip_cost": (mB["manip_trading_pnl"] - m0["manip_trading_pnl"]),
                    "a_m": profB.a_m,
                }
                out.append(row)
                if rho == 0.5:
                    # exact Bayesian mixture MM: manipulator intercept fixed point
                    profBB, infoB = bayes_manip_fixed_point(pB, profB)
                    bm1 = bayes_mm_metrics(profBB, pB, True)
                    prof0B, info0 = bayes_manip_fixed_point(p0, prof0)
                    bm0 = bayes_mm_metrics(prof0B, p0, True)
                    out.append({
                        "sigma_u": su, "B": B, "rho": rho, "mm": "bayes",
                        "lam": profB.lam, "subsidy": profB.lam * su**2,
                        "bias": bm1["price_bias"],
                        "dq0": bm0["decision_quality"], "dq": bm1["decision_quality"],
                        "d_dq": bm1["decision_quality"] - bm0["decision_quality"],
                        "d_approval": bm1["approval_prob"] - bm0["approval_prob"],
                        "a_m": profBB.a_m,
                        "fp_converged": infoB["converged"],
                        "fp_residual": infoB["residual"],
                    })
        print(f"  frontier sigma_u={su} done")
    dump("frontier.json", out)


# ------------------------------------------------------------------- entry
def run_entry():
    """Q3: (a) N honest; (b) N + informed manipulator entrant; (c) N +
    uninformed manipulator entrant; each vs (a). Known (rho=1) and covert
    (rho in {0.5, 0.25}); plus the T2u analog (type uncertainty)."""
    N = 2  # BASE-2 analog: 2 honest incumbents
    out = {"config": {"N": N, **{k: v for k, v in BASE.items() if k != "N"}},
           "reference": {}, "rows": []}
    kw = {k: v for k, v in BASE.items() if k != "N"}
    # references
    b2 = solve_equilibrium(Params(N=N, **kw, manip="none"))
    p2 = Params(N=N, **kw, manip="none")
    b3 = solve_equilibrium(Params(N=N + 1, **kw, manip="none"))
    p3 = Params(N=N + 1, **kw, manip="none")
    out["reference"]["BASE-2"] = metrics(b2, p2, False)
    out["reference"]["BASE-3"] = metrics(b3, p3, False)

    Bs = [0.0, 0.5, 1.0, 2.0, 5.0]
    for manip in ("informed", "uninformed"):
        for rho in (1.0, 0.5, 0.25):
            for B in Bs:
                p = Params(N=N, **kw, B=B, manip=manip, rho=rho)
                row = corruption_row(p, None, deviations=True)
                row["treatment"] = f"{manip}-entrant"
                row["dq_vs_BASE2"] = (row["dq_present"]
                                      - out["reference"]["BASE-2"]["decision_quality"])
                row["dq_vs_BASE3"] = (row["dq_present"]
                                      - out["reference"]["BASE-3"]["decision_quality"])
                out["rows"].append(row)
        print(f"  entry {manip} done")
    # T2u analog: entrant always present, bribed-type w.p. rho
    for rho in (0.5, 0.25):
        for B in Bs:
            p = Params(N=N, **kw, B=B, manip="informed", rho=rho, absent="honest")
            prof = solve_equilibrium(p)
            mB_, mH = metrics(prof, p, True), metrics(prof, p, False)
            out["rows"].append({
                "treatment": "T2u-type-uncertainty", "B": B, "rho": rho,
                "a_m": prof.a_m, "a_e": prof.a_e, "b_m": prof.b_m,
                "b_e": prof.b_e, "b_h": prof.b_h, "lam": prof.lam,
                "bias_bribed": mB_["price_bias"], "bias_honest": mH["price_bias"],
                "dq_bribed": mB_["decision_quality"],
                "dq_honest": mH["decision_quality"],
                "dq_mix": rho * mB_["decision_quality"] + (1 - rho) * mH["decision_quality"],
                "dq_vs_BASE3": (rho * mB_["decision_quality"]
                                + (1 - rho) * mH["decision_quality"]
                                - out["reference"]["BASE-3"]["decision_quality"]),
                "entrant_honest_pnl": mH.get("entrant_honest_pnl"),
                "manip_trading_pnl": mB_.get("manip_trading_pnl"),
                "dev_honest_sup": sup_deviation_honest(prof, p),
                "dev_manip_sup": sup_deviation_manip(prof, p)["gain"],
            })
    print("  entry T2u done")
    dump("entry.json", out)


# -------------------------------------------------------------------- twap
def run_twap():
    out = []
    for T in (1, 2, 4, 8):
        p = tw.TwapParams(**BASE, T=T)
        dyn = tw.solve_honest_dynamics(p)
        for stat in ("twap", "last"):
            ev0 = tw.evaluate(dyn, np.zeros(T), stat)
            for B in (0.0, 0.5, 1.0, 2.0, 5.0):
                if B == 0.0:
                    al = np.zeros(T)
                    ev = ev0
                else:
                    al = tw.solve_manipulator(dyn, B, stat)
                    ev = tw.evaluate(dyn, al, stat)
                out.append({
                    "T": T, "statistic": stat, "B": B,
                    "alphas": list(al),
                    "betas": dyn["betas"], "lams": dyn["lams"],
                    "stat_bias": ev["stat_bias"], "corr_Pv": ev["corr_Pv"],
                    "dq": ev["decision_quality"],
                    "d_dq": ev["decision_quality"] - ev0["decision_quality"],
                    "approval": ev["approval_prob"],
                    "manip_trading_pnl": ev["manip_trading_pnl"],
                    "price_biases": ev["price_biases"],
                })
        # MC verification at B=2
        al = tw.solve_manipulator(dyn, 2.0, "twap")
        mc = tw.mc_check(dyn, al, "twap", n=400_000)
        ana = tw.evaluate(dyn, al, "twap")
        out.append({"T": T, "statistic": "twap-mc-check", "B": 2.0,
                    "analytic_dq": ana["decision_quality"],
                    "mc_dq": mc["decision_quality"],
                    "mc_dq_se": mc["decision_quality_se"],
                    "analytic_bias": ana["stat_bias"], "mc_bias": mc["stat_bias"]})
        print(f"  twap T={T} done")
    dump("twap.json", out)


# --------------------------------------------------------------------- amm
def run_amm():
    """Q5: fixed-impact curve (linearised subsidised LMSR) at kappa =
    multiplier * Kyle-lambda; recompute the Q2 sweep."""
    ref = baseline(BASE["N"] + 1, BASE["sigma_eps"], BASE["sigma_u"])
    out = []
    for mult in (0.5, 1.0, 2.0):
        kappa = mult * ref.lam
        ref0 = None
        for B in (0.0, 0.5, 1.0, 2.0, 5.0):
            p = Params(**BASE, B=B, manip="informed", rho=1.0,
                       mm="fixed", kappa=kappa)
            prof = solve_equilibrium(p)
            m1 = metrics(prof, p, True)
            row = {
                "kappa_mult": mult, "kappa": kappa, "B": B,
                "a_m": prof.a_m, "a_h": prof.a_h, "b_h": prof.b_h,
                "b_m": prof.b_m,
                "bias": m1["price_bias"], "corr_pv": m1["corr_pv"],
                "dq": m1["decision_quality"],
                "approval": m1["approval_prob"],
                "manip_trading_pnl": m1["manip_trading_pnl"],
                "profit_honest_each": m1["profit_honest_each"],
                "profit_mm": m1["profit_mm"],
                "dev_honest_sup": sup_deviation_honest(prof, p),
                "dev_manip_sup": sup_deviation_manip(prof, p)["gain"],
            }
            if B == 0.0:
                ref0 = row
            row["d_dq"] = row["dq"] - ref0["dq"]
            row["d_profit_honest"] = (row["profit_honest_each"]
                                      - ref0["profit_honest_each"])
            out.append(row)
        print(f"  amm kappa_mult={mult} done")
    dump("amm.json", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    todo = ("baseline", "corruption", "frontier", "entry", "twap", "amm") \
        if which == "all" else (which,)
    for name in todo:
        print(f"== {name} ==")
        globals()[f"run_{name}"]()
    print(f"done in {time.time() - t0:.0f}s")
