"""Baseline (B = 0) closed forms for the one-batch N-trader Kyle market.

Model: v ~ N(0,1); trader i in 1..N sees s_i = v + eps_i, eps_i ~ N(0, se2) iid;
one batch of market orders x_i = beta * s_i; noise u ~ N(0, su2); competitive MM
sets p = E[v|y] = lambda * y with y = sum_i x_i + u.  (Kyle 1985, one-shot;
N-trader simultaneous version in the style of Holden & Subrahmanyam [verify].)

Derivation (verified symbolically in `sympy_derivation()` and by Monte Carlo in tests):

Trader FOC:  max_x E[x (v - lambda (x + sum_{j!=i} beta s_j + u)) | s_i]
    E[v|s_i] = rho s_i, rho = 1/(1+se2);   E[s_j|s_i] = rho s_i
    => x = rho s_i (1 - lambda beta (N-1)) / (2 lambda)
    symmetric fixed point:  lambda beta = c := rho / (2 + (N-1) rho)

MM zero-profit:  lambda = Cov(v,y)/Var(y) = beta N / (beta^2 N (N + se2) + su2)
    combined:  lambda = sqrt(c N (1 - c (N + se2))) / su,   beta = c / lambda.

Implications:
    Var(p) = Cov(v,p) = c N  (projection identity)  =>  corr(p,v) = sqrt(cN),
    independent of su: the price DISTRIBUTION is invariant to noise depth.
    Total informed profit = lambda su2 (noise traders' expected loss; MM breaks even),
    proportional to su.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Baseline:
    N: int
    sigma_eps: float
    sigma_u: float
    rho: float      # E[v|s_i] slope
    c: float        # lambda * beta
    lam: float
    beta: float
    var_p: float    # = Cov(v,p) = c N
    corr_pv: float  # = sqrt(cN)
    informed_profit_total: float  # = lam * su^2
    informed_profit_each: float


def baseline(N: int, sigma_eps: float, sigma_u: float) -> Baseline:
    se2 = sigma_eps**2
    rho = 1.0 / (1.0 + se2)
    c = rho / (2.0 + (N - 1) * rho)
    inner = c * N * (1.0 - c * (N + se2))
    assert inner > 0.0, "second-order/positivity condition violated"
    lam = np.sqrt(inner) / sigma_u
    beta = c / lam
    var_p = c * N
    total = lam * sigma_u**2
    return Baseline(
        N=N, sigma_eps=sigma_eps, sigma_u=sigma_u, rho=rho, c=c, lam=lam,
        beta=beta, var_p=var_p, corr_pv=float(np.sqrt(var_p)),
        informed_profit_total=total, informed_profit_each=total / N,
    )


def sympy_derivation(N_val: int = 3, se2_val: float = 0.5) -> dict:
    """Re-derive the equilibrium with SymPy from the FOC + MM condition and
    confirm it matches `baseline()`. Returns the symbolic solution dict."""
    import sympy as sp

    beta, lam, x, s = sp.symbols("beta lam x s", positive=True)
    Nn, se2, su2 = sp.symbols("N sigma_e2 sigma_u2", positive=True)
    rho = 1 / (1 + se2)
    # FOC of trader i given others play beta:
    # U(x) = x*(rho*s - lam*(x + (N-1)*beta*rho*s))   [u and eps mean out]
    U = x * (rho * s - lam * (x + (Nn - 1) * beta * rho * s))
    xstar = sp.solve(sp.diff(U, x), x)[0]
    beta_eq = sp.solve(sp.Eq(xstar, beta * s), beta)[0]  # symmetric fixed point
    # MM: lam = beta*N / (beta^2 * N * (N + se2) + su2)
    lam_eq = sp.solve(
        sp.Eq(lam, beta_eq * Nn / (beta_eq**2 * Nn * (Nn + se2) + su2)), lam
    )
    lam_pos = [sol for sol in lam_eq if sp.simplify(sol.subs({Nn: N_val, se2: se2_val, su2: 1})) > 0][0]

    subs = {Nn: N_val, se2: se2_val, su2: 1.0}
    lam_num = float(lam_pos.subs(subs))
    beta_num = float(beta_eq.subs({lam: lam_num, **subs}))
    ref = baseline(N_val, float(np.sqrt(se2_val)), 1.0)
    assert abs(lam_num - ref.lam) < 1e-10, (lam_num, ref.lam)
    assert abs(beta_num - ref.beta) < 1e-10, (beta_num, ref.beta)
    return {"beta(lam)": beta_eq, "lam": lam_pos, "lam_num": lam_num, "beta_num": beta_num}
