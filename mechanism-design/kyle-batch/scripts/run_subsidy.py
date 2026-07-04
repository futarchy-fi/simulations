#!/usr/bin/env python
"""Subsidy comparison (Q6): two ways to spend a per-market subsidy budget S.

(a) NOISE FLOW: size sigma_u so the expected informed transfer (the noise
    flow's expected loss) lambda * sigma_u^2 = S.  Since the whole affine
    equilibrium scales with sigma_u (flows in sigma_u units, lambda ~ 1/sigma_u),
    sigma_u(S) = S / lambda0(sigma_u=1); verified numerically per row.

(b) AMM DEPTH: the Q5 fixed-impact curve p = kappa*y IS an LMSR with liquidity
    b seen through the logit link: LMSR price pi(Q) = logistic(Q/b) and the
    market's v-price is p = tau*logit(pi) = (tau/b)*Q exactly, so
    kappa = tau/b.  LMSR worst-case maker loss = b*ln2 (binary settlement),
    so b = S/ln2, kappa = tau*ln2/S.  (Bridge caveat: our curve settles at
    unbounded v, the loss bound holds for the bounded binary LMSR; stated in
    KYLE.md.)

Threat model (headline, apples-to-apples): fully covert informed manipulator
(rho -> 0: neither the price rule nor honest traders know it exists), reference
bounty B = 2.  Secondary AMM variant: honest traders aware (rho = 1, the Q5
frame) -- they counter-trade and absorb N/(N+1) of the push.

Also: the frozen-beta behavioral bound for the noise leg (traders do NOT scale
beta up with sigma_u; MM re-fits lambda to the actual flow) -- the pessimistic
aggregation-dampening case flagged by results/llm-decision-market/RESULTS.md
(v1 finding 4: no within-market strategic adaptation).

Writes results/subsidy.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyle_batch.closed_forms import baseline
from kyle_batch.onebatch import Params, Profile, solve_equilibrium, metrics, mm_update
from kyle_batch.mc import simulate, sup_deviation_honest, sup_deviation_manip

RESULTS = Path(__file__).resolve().parents[1] / "results"
N, SE, TAU = 3, 1.0, 0.3
B_REF = 2.0
RHO0 = 1e-12
S_GRID = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
LN2 = float(np.log(2.0))


def noise_row(S: float) -> dict:
    lam0_unit = baseline(N, SE, 1.0).lam        # MM belief: no manipulator
    su = S / lam0_unit
    p0 = Params(N=N, sigma_eps=SE, sigma_u=su, tau=TAU, B=0.0,
                manip="informed", rho=RHO0)
    pB = Params(N=N, sigma_eps=SE, sigma_u=su, tau=TAU, B=B_REF,
                manip="informed", rho=RHO0)
    prof0, profB = solve_equilibrium(p0), solve_equilibrium(pB)
    m0, mB = metrics(prof0, p0, True), metrics(profB, pB, True)
    subsidy_realized = prof0.lam * su**2        # noise flow's expected loss
    # frozen-beta behavioral bound: traders keep their sigma_u=1 strategies,
    # MM re-fits lambda to the actual (thinner-signal) flow
    p1 = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=0.0,
                manip="informed", rho=RHO0)
    prof1 = solve_equilibrium(p1)
    lam_f, mu_f = mm_update(prof1, p0)          # refit on frozen strategies, new su
    prof_f = Profile(prof1.a_h, prof1.b_h, prof1.a_m, prof1.b_m, lam_f, mu_f)
    mf = metrics(prof_f, p0, True)
    return {
        "instrument": "noise", "S": S, "sigma_u": su, "lam": profB.lam,
        "subsidy_realized": subsidy_realized,
        "dq0": m0["decision_quality"], "bias": mB["price_bias"],
        "dq": mB["decision_quality"],
        "damage": m0["decision_quality"] - mB["decision_quality"],
        "corr_pv0": m0["corr_pv"], "approval": mB["approval_prob"],
        "a_m": profB.a_m,
        "dq0_frozen_beta": mf["decision_quality"],
        "corr_pv0_frozen_beta": mf["corr_pv"],
        "dev_honest_sup": sup_deviation_honest(profB, pB),
        "dev_manip_sup": sup_deviation_manip(profB, pB)["gain"],
    }


def amm_row(S: float, aware: bool) -> dict:
    b = S / LN2
    kappa = TAU / b
    rho = 1.0 if aware else RHO0
    p0 = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=0.0,
                manip="informed", rho=rho, mm="fixed", kappa=kappa)
    pB = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=B_REF,
                manip="informed", rho=rho, mm="fixed", kappa=kappa)
    prof0, profB = solve_equilibrium(p0), solve_equilibrium(pB)
    m0, mB = metrics(prof0, p0, True), metrics(profB, pB, True)
    return {
        "instrument": "amm_aware" if aware else "amm_covert",
        "S": S, "b_lmsr": b, "kappa": kappa,
        "worst_case_maker_loss": b * LN2,       # = S by construction
        "maker_expected_loss_B0": -m0["profit_mm"],
        "maker_expected_loss_Bref": -mB["profit_mm"],
        "dq0": m0["decision_quality"], "bias": mB["price_bias"],
        "dq": mB["decision_quality"],
        "damage": m0["decision_quality"] - mB["decision_quality"],
        "corr_pv0": m0["corr_pv"], "approval": mB["approval_prob"],
        "a_m": profB.a_m, "a_h": profB.a_h,
        "dev_honest_sup": sup_deviation_honest(profB, pB),
        "dev_manip_sup": sup_deviation_manip(profB, pB)["gain"],
    }


def run() -> None:
    t0 = time.time()
    lam_star = baseline(N + 1, SE, 1.0).lam
    out = {"config": {"N": N, "sigma_eps": SE, "tau": TAU, "B_ref": B_REF,
                      "S_grid": S_GRID, "rho_headline": "covert (1e-12)",
                      "lmsr_bridge": "kappa = tau*ln2/S, b = S/ln2, p=(tau/b)Q exact",
                      "kappa_equals_kyle_lambda_at_S": TAU * LN2 / lam_star},
           "rows": [], "mc_checks": []}
    for S in S_GRID:
        out["rows"].append(noise_row(S))
        out["rows"].append(amm_row(S, aware=False))
        out["rows"].append(amm_row(S, aware=True))
        print(f"  S={S} done ({time.time()-t0:.0f}s)")

    # MC verification at S = 0.4 for both headline instruments
    for row in out["rows"]:
        if row["S"] != 0.4 or row["instrument"] == "amm_aware":
            continue
        if row["instrument"] == "noise":
            p = Params(N=N, sigma_eps=SE, sigma_u=row["sigma_u"], tau=TAU,
                       B=B_REF, manip="informed", rho=RHO0)
        else:
            p = Params(N=N, sigma_eps=SE, sigma_u=1.0, tau=TAU, B=B_REF,
                       manip="informed", rho=RHO0, mm="fixed",
                       kappa=row["kappa"])
        prof = solve_equilibrium(p)
        ana = metrics(prof, p, True)
        mc = simulate(prof, p, True, n=2_000_000, seed=99)
        out["mc_checks"].append({
            "instrument": row["instrument"], "S": row["S"],
            "analytic_dq": ana["decision_quality"], "mc_dq": mc["decision_quality"],
            "mc_dq_se": mc["decision_quality_se"],
            "analytic_bias": ana["price_bias"], "mc_bias": mc["price_bias"],
            "mc_bias_se": mc["price_bias_se"],
            "dq_agree_4se": abs(ana["decision_quality"] - mc["decision_quality"])
                            < 4 * mc["decision_quality_se"],
            "bias_agree_4se": abs(ana["price_bias"] - mc["price_bias"])
                              < 4 * mc["price_bias_se"],
        })

    path = RESULTS / "subsidy.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
