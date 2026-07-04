"""One-batch linear-Gaussian Kyle decision market with an optional manipulator.

Players
    N symmetric honest informed traders, signals s_i = v + eps_i.
    Optional manipulator ("informed": own signal s_m = v + eps_m, unique;
    "uninformed": no signal), who receives bounty B * 1[approved].
    Approval ~ Bernoulli(q(p)), q(p) = logistic(p/tau).  Outcome settlement:
    every trader's market payoff is x * (v - p).

Market maker
    Linear-projection MM ("linear MM"): believes the manipulator is present
    with probability rho_belief (rho=1: presence common knowledge; rho<1:
    covert). Sets p = lam * (y - mu) with lam = Cov(v,y)/Var(y) and
    mu = E[y] under the mixture belief -- the best *linear* unbiased price.
    An exact Bayesian mixture MM (nonlinear p(y)) is provided separately for
    the covert/camouflage analysis (`bayes_mm_*` functions).

Equilibrium concept
    All traders restricted to affine strategies x = a + b*s.  Honest best
    responses are exactly affine (their conditional problem is quadratic), so
    the restriction binds only on the manipulator, whose bounty term makes the
    true best response nonlinear; the gap is measured by the unilateral
    deviation test in `mc.py` (pointwise sup over ALL strategies).
    Solved by damped fixed-point iteration; manipulator's affine best response
    by numerical optimization of the exact expected utility (closed-form
    trading term + Gauss-Hermite bounty term).
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field, asdict

import numpy as np
from scipy import optimize
from scipy.stats import norm

from .decision import E_q, E_qprime, E_zq, decision_quality, logistic_q
from .closed_forms import baseline


# --------------------------------------------------------------------------
# parameters and profile
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    N: int = 3                 # honest informed traders
    sigma_eps: float = 1.0     # signal noise (honest and informed manipulator)
    sigma_u: float = 1.0       # noise-flow depth
    tau: float = 0.3           # logistic implementation temperature
    B: float = 0.0             # bounty paid to manipulator iff approved
    manip: str = "none"        # "none" | "informed" | "uninformed"
    rho: float = 1.0           # MM's (and honest traders') prior that manipulator is present

    @property
    def se2(self) -> float:
        return self.sigma_eps**2

    @property
    def su2(self) -> float:
        return self.sigma_u**2

    @property
    def has_manip(self) -> bool:
        return self.manip != "none"


@dataclass
class Profile:
    a_h: float
    b_h: float
    a_m: float
    b_m: float
    lam: float
    mu: float     # flow mean subtracted by the MM (belief-weighted)

    def vec(self) -> np.ndarray:
        return np.array([self.a_h, self.b_h, self.a_m, self.b_m, self.lam, self.mu])


# --------------------------------------------------------------------------
# affine algebra over the Gaussian basis (v, eps_1..eps_N, eps_m, u)
# --------------------------------------------------------------------------

class Basis:
    """Random vector basis with diagonal covariance; affine objects are
    (coef, const) pairs. Kills covariance-formula bugs."""

    def __init__(self, p: Params):
        self.p = p
        self.d = p.N + 3
        self.var = np.concatenate([[1.0], np.full(p.N, p.se2), [p.se2], [p.su2]])

    def unit(self, i: int, const: float = 0.0):
        c = np.zeros(self.d)
        c[i] = 1.0
        return (c, const)

    def const(self, k: float):
        return (np.zeros(self.d), k)

    @property
    def v(self):
        return self.unit(0)

    def s(self, i: int):  # honest signal i in 0..N-1
        c = np.zeros(self.d)
        c[0] = 1.0
        c[1 + i] = 1.0
        return (c, 0.0)

    @property
    def s_m(self):
        c = np.zeros(self.d)
        c[0] = 1.0
        c[self.p.N + 1] = 1.0
        return (c, 0.0)

    @property
    def u(self):
        return self.unit(self.p.N + 2)

    @staticmethod
    def lin(alpha: float, x, beta: float = 1.0, y=None):
        """alpha + beta*x (+ y)."""
        c = beta * x[0].copy()
        k = alpha + beta * x[1]
        if y is not None:
            c += y[0]
            k += y[1]
        return (c, k)

    @staticmethod
    def add(*xs):
        c = sum(x[0] for x in xs)
        k = sum(x[1] for x in xs)
        return (c, k)

    def mean(self, x) -> float:
        return float(x[1])

    def cov(self, x, y) -> float:
        return float(np.sum(x[0] * y[0] * self.var))

    def varof(self, x) -> float:
        return self.cov(x, x)


def _flow(bs: Basis, prof: Profile, present: bool):
    """Net order flow y as an affine object, given actual presence."""
    p = bs.p
    xs = [Basis.lin(prof.a_h, bs.s(i), prof.b_h) for i in range(p.N)]
    xs.append(bs.u)
    if present and p.has_manip:
        xs.append(Basis.lin(prof.a_m, bs.s_m, prof.b_m))
    return Basis.add(*xs)


# --------------------------------------------------------------------------
# best responses and MM update
# --------------------------------------------------------------------------

def mm_update(prof: Profile, p: Params) -> tuple[float, float]:
    """Best linear price rule under the MM's mixture belief:
    lam = Cov(v,y)/Var(y), mu = E[y] (mixture moments)."""
    bs = Basis(p)
    rho = p.rho if p.has_manip else 0.0
    y1 = _flow(bs, prof, True)
    y0 = _flow(bs, prof, False)
    mu1, mu0 = bs.mean(y1), bs.mean(y0)
    cov = rho * bs.cov(bs.v, y1) + (1 - rho) * bs.cov(bs.v, y0)
    var = (rho * bs.varof(y1) + (1 - rho) * bs.varof(y0)
           + rho * (1 - rho) * (mu1 - mu0) ** 2)
    mu = rho * mu1 + (1 - rho) * mu0
    return cov / var, mu


def honest_br(prof: Profile, p: Params) -> tuple[float, float]:
    """Exact affine best response of an honest trader (belief rho about the
    manipulator's presence):
        x(s) = [ E[v|s] - lam*E[y_{-i}|s] + lam*mu ] / (2 lam).
    """
    rho_v = 1.0 / (1.0 + p.se2)
    rho = p.rho if p.has_manip else 0.0
    lam, mu = prof.lam, prof.mu
    # E[y_{-i}|s] = (N-1)(a_h + b_h rho_v s) + rho (a_m + b_m rho_v s)
    a = (mu - (p.N - 1) * prof.a_h - rho * prof.a_m) / 2.0
    b = (rho_v - lam * rho_v * ((p.N - 1) * prof.b_h + rho * prof.b_m)) / (2.0 * lam)
    return a, b


def _manip_utility(a: float, b: float, prof: Profile, p: Params) -> float:
    """Exact expected utility of a present manipulator playing x = a + b s_m
    against honest (a_h, b_h) and MM (lam, mu).  Closed-form trading term +
    Gauss-Hermite bounty term."""
    lam, mu = prof.lam, prof.mu
    N, se2, su2 = p.N, p.se2, p.su2
    a_h, b_h = prof.a_h, prof.b_h
    trading = (b
               - lam * (a * a + b * b * (1.0 + se2))
               - lam * (a * N * a_h + b * b_h * N)
               + lam * mu * a)
    if p.B == 0.0:
        return trading
    m_p = lam * (a + N * a_h - mu)
    var_p = lam**2 * (b * b * (1 + se2) + b_h**2 * (N * N + N * se2)
                      + 2 * b * b_h * N + su2)
    return trading + p.B * E_q(m_p, np.sqrt(max(var_p, 0.0)), p.tau)


def manip_br(prof: Profile, p: Params) -> tuple[float, float]:
    """Affine best response of the manipulator (numerical)."""
    informed = p.manip == "informed"

    def neg(theta):
        a = theta[0]
        b = theta[1] if informed else 0.0
        return -_manip_utility(a, b, prof, p)

    x0 = [prof.a_m] + ([prof.b_m] if informed else [])
    res = optimize.minimize(neg, np.array(x0), method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 4000})
    a = float(res.x[0])
    b = float(res.x[1]) if informed else 0.0
    return a, b


def solve_equilibrium(p: Params, damping: float = 0.5, tol: float = 1e-11,
                      max_iter: int = 4000) -> Profile:
    """Damped fixed point over (a_h, b_h, a_m, b_m, lam, mu)."""
    ref = baseline(p.N + (1 if p.manip == "informed" else 0), p.sigma_eps, p.sigma_u) \
        if p.manip == "informed" else baseline(max(p.N, 1), p.sigma_eps, p.sigma_u)
    prof = Profile(a_h=0.0, b_h=ref.beta, a_m=0.0,
                   b_m=ref.beta if p.manip == "informed" else 0.0,
                   lam=ref.lam, mu=0.0)
    for it in range(max_iter):
        a_h, b_h = honest_br(prof, p)
        if p.has_manip:
            a_m, b_m = manip_br(prof, p)
        else:
            a_m, b_m = 0.0, 0.0
        tgt = Profile(a_h, b_h, a_m, b_m, prof.lam, prof.mu)
        lam, mu = mm_update(tgt, p)
        new = Profile(a_h, b_h, a_m, b_m, lam, mu)
        vec_old, vec_new = prof.vec(), new.vec()
        vec = damping * vec_new + (1 - damping) * vec_old
        prof = Profile(*vec)
        if np.max(np.abs(vec_new - vec_old)) < tol:
            return prof
    raise RuntimeError(f"no convergence after {max_iter} iters: {prof}")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(prof: Profile, p: Params, present: bool) -> dict:
    """All closed-form/quadrature metrics for a given actual presence state,
    with the equilibrium (belief-based) MM rule."""
    bs = Basis(p)
    y = _flow(bs, prof, present)
    price = Basis.lin(-prof.lam * prof.mu, y, prof.lam)  # p = lam*(y - mu)
    m_p, var_p = bs.mean(price), bs.varof(price)
    sd_p = float(np.sqrt(var_p))
    cov_vp = bs.cov(bs.v, price)

    out = {
        "present": present,
        "lam": prof.lam, "mu": prof.mu,
        "a_h": prof.a_h, "b_h": prof.b_h, "a_m": prof.a_m, "b_m": prof.b_m,
        "price_bias": m_p,                     # E[p] (baseline: 0)
        "price_slope_on_v": cov_vp,            # E[p|v] = bias + slope*v
        "corr_pv": cov_vp / sd_p if sd_p > 0 else 0.0,
        "var_p": var_p,
        "approval_prob": E_q(m_p, sd_p, p.tau),
        "decision_quality": decision_quality(cov_vp, m_p, var_p, p.tau),
    }
    # trader profits
    x_h = Basis.lin(prof.a_h, bs.s(0), prof.b_h)
    out["profit_honest_each"] = (bs.cov(x_h, bs.v) - bs.cov(x_h, price)
                                 - bs.mean(x_h) * m_p)
    noise_profit = -bs.cov(bs.u, price)
    out["profit_noise"] = noise_profit
    if present and p.has_manip:
        x_m = Basis.lin(prof.a_m, bs.s_m, prof.b_m)
        tp = (bs.cov(x_m, bs.v) - bs.cov(x_m, price) - bs.mean(x_m) * m_p)
        out["manip_trading_pnl"] = tp
        out["manip_bounty"] = p.B * out["approval_prob"]
        out["manip_total"] = tp + out["manip_bounty"]
        out["profit_mm"] = -(p.N * out["profit_honest_each"] + tp + noise_profit)
    else:
        out["profit_mm"] = -(p.N * out["profit_honest_each"] + noise_profit)
    return out


def metrics_mixture(prof: Profile, p: Params) -> dict:
    """rho-weighted mixture of present/absent metrics (decision quality and
    approval mix linearly; correlations do not, so both branches kept)."""
    m1 = metrics(prof, p, True)
    m0 = metrics(prof, p, False)
    rho = p.rho if p.has_manip else 0.0
    return {
        "present": m1, "absent": m0,
        "decision_quality_mix": rho * m1["decision_quality"] + (1 - rho) * m0["decision_quality"],
        "approval_prob_mix": rho * m1["approval_prob"] + (1 - rho) * m0["approval_prob"],
    }


# --------------------------------------------------------------------------
# exact Bayesian mixture MM (nonlinear p(y)) -- covert / camouflage analysis
# --------------------------------------------------------------------------

class BayesMM:
    """Exact p(y) = E[v|y] when the MM's belief is the two-component mixture
    {present w.p. rho, absent w.p. 1-rho} over affine trader strategies."""

    def __init__(self, prof: Profile, p: Params):
        self.p = p
        self.rho = p.rho if p.has_manip else 0.0
        bs = Basis(p)
        self.comp = []
        for present in (True, False):
            y = _flow(bs, prof, present)
            self.comp.append({
                "mu": bs.mean(y), "var": bs.varof(y),
                "cov_vy": bs.cov(bs.v, y),
            })

    def price(self, y: np.ndarray) -> np.ndarray:
        c1, c0 = self.comp
        y = np.asarray(y, dtype=float)
        l1 = norm.pdf(y, c1["mu"], np.sqrt(c1["var"])) * self.rho
        l0 = norm.pdf(y, c0["mu"], np.sqrt(c0["var"])) * (1 - self.rho)
        w = np.where(l1 + l0 > 0, l1 / np.maximum(l1 + l0, 1e-300), self.rho)
        m1 = c1["cov_vy"] / c1["var"] * (y - c1["mu"])
        m0 = c0["cov_vy"] / c0["var"] * (y - c0["mu"])
        return w * m1 + (1 - w) * m0


def bayes_mm_metrics(prof: Profile, p: Params, present: bool, n_gh: int = 400) -> dict:
    """Decision quality / approval / bias under the exact Bayesian mixture MM,
    for a fixed affine trader profile (honest side frozen; see module docstring).
    1-d Gauss-Legendre over the actual y-distribution."""
    bs = Basis(p)
    mmm = BayesMM(prof, p)
    y_aff = _flow(bs, prof, present)
    mu_y, var_y = bs.mean(y_aff), bs.varof(y_aff)
    cov_vy = bs.cov(bs.v, y_aff)
    sd = np.sqrt(var_y)
    z, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / np.sqrt(2 * np.pi)
    y = mu_y + sd * z
    pr = mmm.price(y)
    q = logistic_q(pr, p.tau)
    Ev_given_y = cov_vy / var_y * (y - mu_y)
    return {
        "present": present,
        "price_bias": float(np.sum(w * pr)),
        "approval_prob": float(np.sum(w * q)),
        "decision_quality": float(np.sum(w * Ev_given_y * q)),
    }
