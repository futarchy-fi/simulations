"""Q4: T batches; decision statistic = TWAP of batch prices vs last batch price.

Model
    Same v, signals drawn once; each batch t has fresh noise u_t ~ N(0, su2).
    Honest traders play MYOPIC round-by-round linear strategies
        x_{i,t} = beta_t * (s_i - E[s_i | public info_{t-1}])
    and the MM is a myopic-lambda Bayesian projection MM: after each batch
        p_t = m_t = m_{t-1} + lam_t (y_t - E[y_t | F_{t-1}]),
    both computed under the belief that there is NO manipulator (fully covert
    manipulator, the rho=0 case of `onebatch`; with a known manipulator the
    deterministic push is subtracted every round and both decision statistics
    are unbiased -- the covert case is where the TWAP question lives).

    The manipulator is UNINFORMED and open-loop: it commits to deterministic
    pushes (alpha_1..alpha_T) maximizing
        sum_t E[alpha_t (v - p_t)] + B E[q(P_stat)],
    where P_stat = mean(p_1..p_T) ("twap") or p_T ("last"). This isolates the
    decision-statistic timing channel (the CFR entry sweeps showed the
    last-mover attack is information-free: T1-last = T2-last).

    This is a myopic equilibrium, NOT a dynamic Nash: honest traders do not
    internalise future rounds (cf. dynamic multi-informed Kyle models, e.g.
    Holden-Subrahmanyam [verify]); stated as such in KYLE.md.

Everything is computed EXACTLY by affine propagation over the Gaussian basis
(v, eps_1..eps_N, u_1..u_T) -- no Monte Carlo needed (MC used only to verify).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .decision import E_q, E_zq, logistic_q


@dataclass(frozen=True)
class TwapParams:
    N: int = 3
    sigma_eps: float = 1.0
    sigma_u: float = 1.0
    tau: float = 0.3
    T: int = 4
    B: float = 0.0
    statistic: str = "twap"   # "twap" | "last"


class _Aff:
    """Affine objects over the Gaussian basis; var = diagonal variances."""

    def __init__(self, var: np.ndarray):
        self.var = var
        self.d = len(var)

    def zero(self):
        return np.zeros(self.d + 1)  # last entry = constant

    def unit(self, i):
        z = self.zero()
        z[i] = 1.0
        return z

    def cov(self, a, b):
        return float(np.sum(a[:-1] * b[:-1] * self.var))

    def mean(self, a):
        return float(a[-1])


def solve_honest_dynamics(p: TwapParams) -> dict:
    """Belief pass: honest N-trader myopic dynamic equilibrium (no manipulator).

    Returns per-round beta_t, lam_t and the MM's raw-flow functionals:
        m_t   = sum_r M[t, r] * y_r          (posterior mean of v)
        ehat_t = sum_r G[t, r] * y_r         (posterior mean of any eps_i;
                                              identical across i by symmetry)
    plus belief-pass affine reps of prices for diagnostics.
    """
    N, T = p.N, p.T
    se2, su2 = p.sigma_eps**2, p.sigma_u**2
    var = np.concatenate([[1.0], np.full(N, se2), np.full(T, su2)])
    A = _Aff(var)
    v = A.unit(0)
    eps = [A.unit(1 + i) for i in range(N)]
    s = [v + e for e in eps]
    u = [A.unit(1 + N + t) for t in range(T)]

    betas, lams = [], []
    M = np.zeros((T + 1, T))      # m_t functional on raw y_1..y_T
    G = np.zeros((T + 1, T))      # E[eps_i | F_t] functional (any i)
    ytilde_raw = []               # innovations as raw-y functionals
    y_aff, m_aff, price_aff = [], [A.zero()], []
    ehat_aff = A.zero()

    m_prev = A.zero()
    ehat_prev = A.zero()
    for t in range(T):
        # conditional (on F_{t-1}) private innovation of trader i:
        #   shat_i = s_i - m_{t-1} - ehat_{t-1}
        shat = [s[i] - m_prev - ehat_prev for i in range(N)]
        v_res = v - m_prev
        var_shat = A.cov(shat[0], shat[0])
        kappa = A.cov(v_res, shat[0]) / var_shat
        gamma = (A.cov(shat[0], shat[1]) / var_shat) if N > 1 else 0.0
        c_t = kappa / (2.0 + (N - 1) * gamma)          # = lam_t * beta_t (myopic FOC)
        Ssum = sum(shat)
        C = A.cov(v_res, Ssum)
        V = A.cov(Ssum, Ssum)
        inner = c_t * (C - c_t * V)
        assert inner > 0, "myopic second-order condition violated"
        lam_t = np.sqrt(inner) / p.sigma_u
        beta_t = c_t / lam_t
        betas.append(beta_t)
        lams.append(lam_t)

        y_t = beta_t * Ssum + u[t]
        y_aff.append(y_t)
        # strategies subtract E[s_i|F_{t-1}] already, so E_b[y_t|F_{t-1}] = 0:
        # the raw flow IS the innovation (orthogonal to F_{t-1} by construction)
        r = np.zeros(T)
        r[t] = 1.0
        ytilde_raw.append(r)
        ytilde = y_t
        var_yt = A.cov(ytilde, ytilde)
        assert all(abs(A.cov(ytilde, y_aff[rr])) < 1e-10 for rr in range(t)), \
            "innovation not orthogonal to past flows"
        # posterior updates
        cv = A.cov(v_res, ytilde) / var_yt
        ce = A.cov(eps[0] - ehat_prev, ytilde) / var_yt
        M[t + 1] = M[t] + cv * r
        G[t + 1] = G[t] + ce * r
        m_prev = m_prev + cv * ytilde
        ehat_prev = ehat_prev + ce * ytilde
        m_aff.append(m_prev)
        price_aff.append(m_prev)
        # consistency: lam_t must equal cv (projection onto raw innovation)
        assert abs(lam_t - cv) < 1e-9, (lam_t, cv)

    return {"p": p, "A": A, "betas": betas, "lams": lams, "M": M, "G": G,
            "price_aff": price_aff, "y_aff": y_aff, "v": v}


def actual_pass(dyn: dict, alphas: np.ndarray) -> dict:
    """Actual pass: rebuild flows/prices when a covert manipulator adds the
    deterministic push alpha_t each round.  Honest traders and the MM apply
    their belief-pass functionals to the ACTUAL observed flows."""
    p: TwapParams = dyn["p"]
    N, T = p.N, p.T
    A: _Aff = dyn["A"]
    betas, M, G = dyn["betas"], dyn["M"], dyn["G"]
    v = dyn["v"]
    eps = [A.unit(1 + i) for i in range(N)]
    s = [v + e for e in eps]
    u = [A.unit(1 + N + t) for t in range(T)]

    y_act, m_act = [], []
    prices = []
    for t in range(T):
        m_prev = sum((M[t, r] * y_act[r] for r in range(t)), A.zero())
        ehat_prev = sum((G[t, r] * y_act[r] for r in range(t)), A.zero())
        push = A.zero()
        push[-1] = alphas[t]
        y_t = sum((betas[t] * (s[i] - m_prev - ehat_prev) for i in range(N)),
                  A.zero()) + u[t] + push
        y_act.append(y_t)
        m_t = sum((M[t + 1, r] * y_act[r] for r in range(t + 1)), A.zero())
        m_act.append(m_t)
        prices.append(m_t)
    return {"y": y_act, "prices": prices}


def stat_weights(T: int, statistic: str) -> np.ndarray:
    """Weight vector w over batch prices: statistic = sum_t w_t p_t.

    "last"    -> e_T                     (K = 1)
    "twap"    -> uniform 1/T             (K = T)
    "win:K"   -> mean of the LAST K batch prices (windowed TWAP; K <= T)
    """
    w = np.zeros(T)
    if statistic == "twap":
        w[:] = 1.0 / T
    elif statistic == "last":
        w[-1] = 1.0
    elif statistic.startswith("win:"):
        K = int(statistic.split(":")[1])
        assert 1 <= K <= T, (K, T)
        w[T - K:] = 1.0 / K
    else:
        raise ValueError(statistic)
    return w


def statistic_aff(dyn: dict, prices: list, statistic: str):
    w = stat_weights(dyn["p"].T, statistic)
    return sum((w[t] * prices[t] for t in range(dyn["p"].T)), dyn["A"].zero())


def evaluate(dyn: dict, alphas: np.ndarray, statistic: str) -> dict:
    """Exact metrics for a given push vector."""
    p: TwapParams = dyn["p"]
    A: _Aff = dyn["A"]
    act = actual_pass(dyn, alphas)
    P = statistic_aff(dyn, act["prices"], statistic)
    v = dyn["v"]
    m_P, var_P = A.mean(P), A.cov(P, P)
    sd_P = np.sqrt(var_P)
    cov_vP = A.cov(v, P)
    dq = cov_vP / var_P * E_zq(m_P, sd_P, p.tau) if var_P > 0 else 0.0
    biases = [A.mean(pr) for pr in act["prices"]]
    manip_trading = -float(np.sum(np.asarray(alphas) * np.asarray(biases)))
    approval = E_q(m_P, sd_P, p.tau)
    return {
        "stat_bias": m_P, "stat_sd": sd_P, "corr_Pv": cov_vP / sd_P,
        "decision_quality": dq, "approval_prob": approval,
        "price_biases": biases,
        "manip_trading_pnl": manip_trading,
        "manip_total": manip_trading + p.B * approval,
    }


def solve_manipulator(dyn: dict, B: float, statistic: str) -> np.ndarray:
    """Open-loop optimal pushes alpha_1..alpha_T (exact objective, scipy)."""
    p: TwapParams = dyn["p"]
    T = p.T

    def neg(alphas):
        ev = evaluate(dyn, alphas, statistic)
        return -(ev["manip_trading_pnl"] + B * ev["approval_prob"])

    res = optimize.minimize(neg, np.zeros(T), method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12,
                                     "maxiter": 20000, "maxfev": 40000})
    assert res.success or res.fun is not None
    return res.x


# --------------------------------------------------------------------------
# fast exact solver via the push-response matrix (windowed / concealed K)
# --------------------------------------------------------------------------

def push_response(dyn: dict) -> dict:
    """Everything the manipulator's open-loop problem needs, precomputed.

    Price means are LINEAR in the push vector (alpha enters the affine
    propagation through constants only): bias_t = (D @ alpha)_t with
    D[t, r] = E[p_t | alpha = e_r]; all covariances are alpha-independent.
    Hence for any statistic P = w'p:
        E[P] = w'D alpha,  sd(P), Cov(v, P) fixed,
        manip trading pnl = -alpha' D alpha,
        U(alpha) = -alpha'D alpha + B * E_q(w'D alpha, sd_P).
    Verified against `evaluate` (full affine propagation) in tests.
    """
    p: TwapParams = dyn["p"]
    T = p.T
    A: _Aff = dyn["A"]
    D = np.zeros((T, T))
    for r in range(T):
        e = np.zeros(T)
        e[r] = 1.0
        act = actual_pass(dyn, e)
        D[:, r] = [A.mean(pr) for pr in act["prices"]]
    prices0 = actual_pass(dyn, np.zeros(T))["prices"]
    cov_pp = np.array([[A.cov(prices0[i], prices0[j]) for j in range(T)]
                       for i in range(T)])
    cov_vp = np.array([A.cov(dyn["v"], prices0[t]) for t in range(T)])
    return {"p": p, "D": D, "cov_pp": cov_pp, "cov_vp": cov_vp}


def _stat_moments(pr: dict, w: np.ndarray) -> tuple[float, float]:
    """(sd_P, cov_vP) for statistic weights w (alpha-independent)."""
    sd = float(np.sqrt(w @ pr["cov_pp"] @ w))
    return sd, float(pr["cov_vp"] @ w)


def solve_manipulator_fast(pr: dict, B: float, statistics: list[str],
                           probs: list[float] | None = None) -> np.ndarray:
    """Open-loop optimal pushes when the settlement statistic is drawn from
    `statistics` with probabilities `probs` AFTER trading (concealed window);
    a single statistic = the known-K case.  Exact objective and gradient:

        U(alpha) = -alpha'D alpha + B sum_k pi_k E_q(w_k'D alpha, sd_k)
        grad U   = -(D + D')alpha + B sum_k pi_k E[q'](m_k, sd_k) D'w_k
    """
    from .decision import E_qprime
    p: TwapParams = pr["p"]
    T, D = p.T, pr["D"]
    ws = [stat_weights(T, s) for s in statistics]
    pis = np.full(len(ws), 1.0 / len(ws)) if probs is None else np.asarray(probs)
    sds = [_stat_moments(pr, w)[0] for w in ws]

    def negU_and_grad(alpha):
        Da = D @ alpha
        U = -alpha @ Da
        g = -(D + D.T) @ alpha
        for w, pi, sd in zip(ws, pis, sds):
            m = float(w @ Da)
            U += B * pi * E_q(m, sd, p.tau)
            g = g + B * pi * E_qprime(m, sd, p.tau) * (D.T @ w)
        return -U, -g

    best = None
    for x0 in (np.zeros(T), np.full(T, 0.1)):
        res = optimize.minimize(negU_and_grad, x0, jac=True, method="BFGS",
                                options={"gtol": 1e-12, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    assert np.linalg.norm(best.jac) < 1e-7, ("gradient not stationary", best.jac)
    return best.x


def evaluate_mixture(dyn: dict, alphas: np.ndarray, statistics: list[str],
                     B: float, probs: list[float] | None = None) -> dict:
    """Realized metrics when K is drawn after trading: per-statistic metrics
    (via the exact affine `evaluate`) plus the probability mixture."""
    pis = (np.full(len(statistics), 1.0 / len(statistics)) if probs is None
           else np.asarray(probs))
    per = {s: evaluate(dyn, alphas, s) for s in statistics}
    dq = float(sum(pi * per[s]["decision_quality"] for s, pi in zip(statistics, pis)))
    ap = float(sum(pi * per[s]["approval_prob"] for s, pi in zip(statistics, pis)))
    trading = per[statistics[0]]["manip_trading_pnl"]  # statistic-independent
    return {"per_statistic": per, "decision_quality_mix": dq,
            "approval_prob_mix": ap, "manip_trading_pnl": trading,
            "manip_total": trading + B * ap}


def mc_check(dyn: dict, alphas: np.ndarray, statistic: str,
             n: int = 400_000, seed: int = 3) -> dict:
    """Monte Carlo verification of the affine propagation."""
    p: TwapParams = dyn["p"]
    N, T = p.N, p.T
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    eps = rng.standard_normal((N, n)) * p.sigma_eps
    s = v[None] + eps
    M, G, betas = dyn["M"], dyn["G"], dyn["betas"]
    ys = np.zeros((T, n))
    prices = np.zeros((T, n))
    for t in range(T):
        m_prev = (M[t, :t, None] * ys[:t]).sum(axis=0) if t else np.zeros(n)
        e_prev = (G[t, :t, None] * ys[:t]).sum(axis=0) if t else np.zeros(n)
        x = betas[t] * (s - m_prev[None] - e_prev[None])
        ys[t] = x.sum(axis=0) + rng.standard_normal(n) * p.sigma_u + alphas[t]
        prices[t] = (M[t + 1, :t + 1, None] * ys[:t + 1]).sum(axis=0)
    P = stat_weights(T, statistic) @ prices
    q = logistic_q(P, p.tau)
    return {
        "stat_bias": float(P.mean()), "stat_sd": float(P.std()),
        "corr_Pv": float(np.corrcoef(P, v)[0, 1]),
        "decision_quality": float((v * q).mean()),
        "decision_quality_se": float((v * q).std() / np.sqrt(n)),
        "approval_prob": float(q.mean()),
        "price_biases": [float(prices[t].mean()) for t in range(T)],
    }
