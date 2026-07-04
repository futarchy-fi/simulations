"""Monte Carlo verification and unilateral-deviation tests.

Two deviation tests back every reported equilibrium:

1. `grid_deviation_mc` -- Monte Carlo, common random numbers: a single trader
   deviates to each affine strategy (a, b) on a grid around the equilibrium
   while everyone else (and the MM rule) stays fixed; reports the maximum
   estimated gain and its standard error.

2. `sup_deviation` -- semi-analytic *supremum* over ALL strategies (any
   measurable x(s)): the deviator's conditional problem is solved pointwise in
   the signal (closed form for honest traders, whose problem is conditionally
   quadratic; 1-d numeric optimization + Gauss-Hermite for the manipulator,
   whose bounty term is nonlinear). This dominates any grid and quantifies the
   error of the affine-strategy restriction.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .decision import E_q, logistic_q
from .onebatch import Params, Profile, _manip_utility


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def simulate(prof: Profile, p: Params, present: bool, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    eps = rng.standard_normal((p.N, n)) * p.sigma_eps
    u = rng.standard_normal(n) * p.sigma_u
    s = v[None, :] + eps
    x_h = prof.a_h + prof.b_h * s
    y = x_h.sum(axis=0) + u
    x_m = np.zeros(n)
    if p.has_manip:
        s_m = v + rng.standard_normal(n) * p.sigma_eps
        if present:
            x_m = prof.a_m + prof.b_m * s_m
            y = y + x_m
        elif p.absent == "honest":
            y = y + prof.a_e + prof.b_e * s_m
    price = prof.lam * (y - prof.mu)
    q = logistic_q(price, p.tau)
    approve = rng.random(n) < q
    out = {
        "n": n,
        "price_bias": float(price.mean()),
        "price_bias_se": float(price.std() / np.sqrt(n)),
        "corr_pv": float(np.corrcoef(price, v)[0, 1]),
        "var_p": float(price.var()),
        "approval_prob": float(q.mean()),
        "decision_quality": float((v * q).mean()),           # exact-q version
        "decision_quality_se": float((v * q).std() / np.sqrt(n)),
        "decision_quality_bernoulli": float((v * approve).mean()),
        "profit_honest_each": float((x_h[0] * (v - price)).mean()),
        "profit_honest_each_se": float((x_h[0] * (v - price)).std() / np.sqrt(n)),
        "profit_noise": float((u * (v - price)).mean()),
    }
    if present and p.has_manip:
        pnl = x_m * (v - price)
        out["manip_trading_pnl"] = float(pnl.mean())
        out["manip_trading_pnl_se"] = float(pnl.std() / np.sqrt(n))
        out["manip_bounty"] = float(p.B * q.mean())
    return out


# --------------------------------------------------------------------------
# deviation tests
# --------------------------------------------------------------------------

def grid_deviation_mc(prof: Profile, p: Params, role: str, n: int = 400_000,
                      seed: int = 1, half_width: float = 0.5, steps: int = 21) -> dict:
    """Max MC-estimated gain for `role` in {'honest','manip'} deviating to an
    affine strategy on a (a, b) grid centered at equilibrium (common random
    numbers; bounty evaluated as B*q(p), same expectation as Bernoulli)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    eps = rng.standard_normal((p.N, n)) * p.sigma_eps
    u = rng.standard_normal(n) * p.sigma_u
    s = v[None, :] + eps
    s_m = v + rng.standard_normal(n) * p.sigma_eps

    x_h_all = prof.a_h + prof.b_h * s
    x_m = (prof.a_m + prof.b_m * s_m) if p.has_manip else np.zeros(n)

    if role == "honest":
        # honest trader's belief: entrant bribed w.p. rho -> simulate the type
        pres = rng.random(n) < (p.rho if p.has_manip else 0.0)
        x_alt = (prof.a_e + prof.b_e * s_m) if (p.has_manip and p.absent == "honest") else np.zeros(n)
        others = x_h_all[1:].sum(axis=0) + u + np.where(pres, x_m, x_alt)
        sig = s[0]
        a0, b0 = prof.a_h, prof.b_h
        bounty = 0.0
    elif role == "manip":
        assert p.has_manip
        others = x_h_all.sum(axis=0) + u
        sig = s_m if p.manip == "informed" else np.zeros(n)
        a0, b0 = prof.a_m, prof.b_m
        bounty = p.B
    else:
        raise ValueError(role)

    def payoff(a, b):
        x = a + b * sig
        price = prof.lam * (x + others - prof.mu)
        val = x * (v - price)
        if bounty:
            val = val + bounty * logistic_q(price, p.tau)
        return val

    base = payoff(a0, b0)
    base_mean = base.mean()
    best_gain, best_ab, best_se = -np.inf, (a0, b0), 0.0
    for da in np.linspace(-half_width, half_width, steps):
        for db in (np.linspace(-half_width, half_width, steps)
                   if (role == "manip" and p.manip == "informed") or role == "honest"
                   else [0.0]):
            diff = payoff(a0 + da, b0 + db) - base
            g = diff.mean()
            if g > best_gain:
                best_gain, best_ab = g, (a0 + da, b0 + db)
                best_se = diff.std() / np.sqrt(n)
    return {"role": role, "max_gain": float(best_gain), "se": float(best_se),
            "argmax": best_ab, "baseline_payoff": float(base_mean),
            "grid_half_width": half_width, "grid_steps": steps, "n": n}


def sup_deviation_honest(prof: Profile, p: Params) -> float:
    """Exact sup-over-all-strategies gain for an honest trader: their problem
    is conditionally quadratic, so the pointwise optimum is affine and the gain
    is lam * E[(x* - x_eq)^2]."""
    from .onebatch import honest_br
    a_star, b_star = honest_br(prof, p)
    da, db = a_star - prof.a_h, b_star - prof.b_h
    return float(prof.lam * (da**2 + db**2 * (1.0 + p.se2)))


def sup_deviation_manip(prof: Profile, p: Params, n_gh: int = 64) -> dict:
    """Semi-analytic supremum over ALL manipulator strategies x(s_m):
    E_s[max_x h(x, s)] - U(affine eq).  h's bounty term uses the exact
    conditional Gaussian law of the others' flow."""
    assert p.has_manip
    lam, mu = prof.lam, prof.mu
    N, se2, su2 = p.N, p.se2, p.su2
    rho_v = 1.0 / (1.0 + se2)

    # conditional law of Z = N a_h + b_h * S + u given s_m (informed case)
    var_v_s = se2 / (1.0 + se2)          # Var(v | s_m)
    var_S_s = N * N * var_v_s + N * se2  # Var(S | s_m)
    sd_z_s = np.sqrt(prof.b_h**2 * var_S_s + su2)

    def h_max(s):
        Ev = rho_v * s
        Ez = N * prof.a_h + prof.b_h * N * rho_v * s

        def neg(x):
            m_p = lam * (x + Ez - mu)
            trading = x * (Ev - lam * x - lam * Ez + lam * mu)
            return -(trading + p.B * E_q(m_p, lam * sd_z_s, p.tau))

        res = optimize.minimize_scalar(neg, bounds=(-1e3, 1e3), method="bounded",
                                       options={"xatol": 1e-11})
        return -res.fun

    if p.manip == "informed":
        z, w = np.polynomial.hermite_e.hermegauss(n_gh)
        w = w / np.sqrt(2 * np.pi)
        sd_s = np.sqrt(1.0 + se2)
        sup_val = float(sum(wi * h_max(sd_s * zi) for zi, wi in zip(z, w)))
    else:
        # uninformed: single unconditional problem; Z's law is unconditional
        sd_z = np.sqrt(prof.b_h**2 * (N * N + N * se2) + su2)

        def neg(x):
            m_p = lam * (x + N * prof.a_h - mu)
            trading = x * (0.0 - lam * x - lam * N * prof.a_h + lam * mu)
            return -(trading + p.B * E_q(m_p, lam * sd_z, p.tau))

        res = optimize.minimize_scalar(neg, bounds=(-1e3, 1e3), method="bounded",
                                       options={"xatol": 1e-11})
        sup_val = -res.fun

    u_eq = _manip_utility(prof.a_m, prof.b_m, prof, p)
    return {"sup_value": sup_val, "eq_value": float(u_eq),
            "gain": float(sup_val - u_eq)}


def deviation_report(prof: Profile, p: Params, n: int = 400_000, seed: int = 1) -> dict:
    """Full deviation certificate for one equilibrium."""
    rep = {
        "honest_sup_gain": sup_deviation_honest(prof, p),
        "honest_grid": grid_deviation_mc(prof, p, "honest", n=n, seed=seed),
    }
    if p.has_manip:
        rep["manip_sup"] = sup_deviation_manip(prof, p)
        rep["manip_grid"] = grid_deviation_mc(prof, p, "manip", n=n, seed=seed + 1)
    gains = [rep["honest_sup_gain"], rep["honest_grid"]["max_gain"]]
    if p.has_manip:
        gains += [rep["manip_sup"]["gain"], rep["manip_grid"]["max_gain"]]
    rep["max_gain_any"] = float(max(gains))
    return rep
