"""Q7: T batches WITH exogenous information arrival during the trading window.

Q4/Q4b are static-information models: every private signal exists at t=0, so a
manipulator's push enters the posterior and persists (decaying only through
honest price-anchored correction), and the optimal read window is K*=1
everywhere.  This module tests whether that verdict is an artifact of zero
in-window information arrival.  Two arrival variants, same myopic-lambda
Bayesian projection MM and covert uninformed open-loop manipulator as Q4:

(b) PUBLIC SIGNAL STREAM (`ArrivalParams`, headline variant).
    At the start of each batch t = 2..T a public noisy observation
        z_t = v + eta_t,   eta_t ~ N(0, sigma_z^2) i.i.d.
    is released (a live unconditional stock price) and the MM conditions the
    price on it: p_t = E[v | y_1..y_t, z_2..z_t].  Honest traders condition on
    the same public info.  Parameterization holds the TOTAL end-of-window
    fundamental precision fixed at Pi (reference Pi = 3 = the Q4 N=3,
    sigma_eps=1 private precision) and moves a fraction phi of it into the
    public stream, spread evenly over batches 2..T:
        private:  N signals with sigma_eps^2 = N / ((1-phi) Pi)
        public:   each z_t has precision phi Pi / (T-1),
                  i.e. sigma_z^2 = (T-1) / (phi Pi).
    phi = 0 is EXACTLY the Q4/Q4b model (asserted in tests); phi = "fraction
    of total information arriving after batch 1".

(a) STAGGERED PRIVATE SIGNALS (`StaggeredParams`).
    Trader i's signal s_i = v + eps_i arrives at batch t_i (it does not exist
    before); traders trade 0 until arrival, then play myopic linear strategies
    on their residual private info.  Arrival times heterogeneous => per-trader
    (beta_{i,t}) solved as a per-round linear system given lambda_t, with a
    damped fixed point on lambda_t (reduces to the symmetric closed form of
    `twap.solve_honest_dynamics` when all t_i = 1; asserted in tests).

The manipulator's push enters the affine propagation through CONSTANTS only,
so price means remain LINEAR in the push vector and all covariances are
push-independent: the push-response machinery of `twap` (D matrix, fast BFGS
solver, windowed/concealed statistics) is reused verbatim.  The public z_t
carries NO push -- the market's z-innovation z_t - m_{t-1} is therefore biased
*against* an outstanding push (the pushed posterior overpredicts z), which is
the wash-out channel the hypothesis is about.

Everything is exact affine Gaussian propagation (MC used only to verify).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision import E_q, E_zq, logistic_q
from .twap import _Aff, stat_weights


# --------------------------------------------------------------------------
# (b) public signal stream
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArrivalParams:
    N: int = 3
    sigma_u: float = 1.0
    tau: float = 0.3
    T: int = 4
    B: float = 0.0
    phi: float = 0.0        # fraction of total info arriving publicly after batch 1
    Pi: float = 3.0         # total fundamental precision beyond the prior

    @property
    def sigma_eps(self) -> float:
        assert 0.0 <= self.phi < 1.0
        return float(np.sqrt(self.N / ((1.0 - self.phi) * self.Pi)))

    @property
    def sigma_z(self) -> float:
        assert self.phi > 0.0 and self.T > 1
        return float(np.sqrt((self.T - 1) / (self.phi * self.Pi)))

    @property
    def n_z(self) -> int:
        return self.T - 1 if self.phi > 0.0 else 0


def solve_honest_dynamics_arrival(p: ArrivalParams) -> dict:
    """Belief pass with a public stream: honest myopic dynamics, no manipulator.

    Observations arrive in the order y_1, z_2, y_2, z_3, ..., z_T, y_T (z_t at
    the START of batch t, so p_t already reflects z_t).  Returns per-round
    beta_t, lam_t and the MM's raw-observation functionals
        Mrows[k] : E[v | first k observations]      (row over raw obs)
        Grows[k] : E[eps_i | first k observations]  (any i, symmetric)
    plus the observation schedule and belief-pass affine reps.
    """
    N, T = p.N, p.T
    se2, su2 = p.sigma_eps**2, p.sigma_u**2
    n_z = p.n_z
    sz2 = p.sigma_z**2 if n_z else 0.0
    var = np.concatenate([[1.0], np.full(N, se2), np.full(T, su2),
                          np.full(n_z, sz2)])
    A = _Aff(var)
    v = A.unit(0)
    eps = [A.unit(1 + i) for i in range(N)]
    s = [v + e for e in eps]
    u = [A.unit(1 + N + t) for t in range(T)]
    eta = [A.unit(1 + N + T + j) for j in range(n_z)]

    n_obs = T + n_z
    schedule: list[tuple[str, int]] = []      # ("z"|"y", batch index t)
    Mrows = [np.zeros(n_obs)]
    Grows = [np.zeros(n_obs)]
    betas, lams = [], []
    obs_aff = []                              # belief-pass affine reps
    price_step = []                           # k such that p_t = Mrows[k] @ obs

    m_prev = A.zero()
    ehat_prev = A.zero()
    for t in range(T):
        if n_z and t >= 1:
            z = v + eta[t - 1]
            ztilde = z - m_prev               # E[z_t | F] = m_{t-1}
            var_z = A.cov(ztilde, ztilde)
            cv = A.cov(v - m_prev, ztilde) / var_z
            ce = A.cov(eps[0] - ehat_prev, ztilde) / var_z
            k = len(obs_aff)
            e_k = np.zeros(n_obs)
            e_k[k] = 1.0
            innov_raw = e_k - Mrows[-1]       # z - m_prev as raw functional
            Mrows.append(Mrows[-1] + cv * innov_raw)
            Grows.append(Grows[-1] + ce * innov_raw)
            m_prev = m_prev + cv * ztilde
            ehat_prev = ehat_prev + ce * ztilde
            obs_aff.append(z)
            schedule.append(("z", t))
        # trading (same myopic FOC as twap.solve_honest_dynamics, with the
        # current conditional covariances)
        shat = [s[i] - m_prev - ehat_prev for i in range(N)]
        v_res = v - m_prev
        var_shat = A.cov(shat[0], shat[0])
        kappa = A.cov(v_res, shat[0]) / var_shat
        gamma = (A.cov(shat[0], shat[1]) / var_shat) if N > 1 else 0.0
        c_t = kappa / (2.0 + (N - 1) * gamma)
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
        # strategies subtract E[s_i|F] => the raw flow IS the innovation
        assert all(abs(A.cov(y_t, o)) < 1e-10 for o in obs_aff), \
            "flow innovation not orthogonal to past observations"
        var_yt = A.cov(y_t, y_t)
        cv = A.cov(v_res, y_t) / var_yt
        ce = A.cov(eps[0] - ehat_prev, y_t) / var_yt
        k = len(obs_aff)
        e_k = np.zeros(n_obs)
        e_k[k] = 1.0
        Mrows.append(Mrows[-1] + cv * e_k)
        Grows.append(Grows[-1] + ce * e_k)
        m_prev = m_prev + cv * y_t
        ehat_prev = ehat_prev + ce * y_t
        obs_aff.append(y_t)
        schedule.append(("y", t))
        price_step.append(len(obs_aff))
        assert abs(lam_t - cv) < 1e-9, (lam_t, cv)

    return {"p": p, "A": A, "betas": betas, "lams": lams,
            "Mrows": Mrows, "Grows": Grows, "schedule": schedule,
            "price_step": price_step, "obs_aff": obs_aff, "v": v}


def actual_pass_arrival(dyn: dict, alphas: np.ndarray) -> dict:
    """Actual pass with a covert deterministic push alpha_t per batch.

    Honest traders and the MM apply their belief-pass functionals to the
    ACTUAL observations; the public z_t carries no push."""
    p: ArrivalParams = dyn["p"]
    N, T = p.N, p.T
    A: _Aff = dyn["A"]
    betas, Mrows, Grows = dyn["betas"], dyn["Mrows"], dyn["Grows"]
    v = dyn["v"]
    eps = [A.unit(1 + i) for i in range(N)]
    s = [v + e for e in eps]
    u = [A.unit(1 + N + t) for t in range(T)]
    eta = [A.unit(1 + N + T + j) for j in range(p.n_z)]

    obs_act: list[np.ndarray] = []
    prices = []

    def apply(row):
        return sum((row[k] * obs_act[k] for k in range(len(obs_act))), A.zero())

    for t in range(T):
        if p.n_z and t >= 1:
            obs_act.append(v + eta[t - 1])
        k_now = len(obs_act)
        m_prev = apply(Mrows[k_now])
        ehat_prev = apply(Grows[k_now])
        push = A.zero()
        push[-1] = alphas[t]
        y_t = sum((betas[t] * (s[i] - m_prev - ehat_prev) for i in range(N)),
                  A.zero()) + u[t] + push
        obs_act.append(y_t)
        prices.append(apply(Mrows[len(obs_act)]))
    return {"obs": obs_act, "prices": prices}


# --------------------------------------------------------------------------
# (a) staggered private signals
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StaggeredParams:
    N: int = 4
    sigma_u: float = 1.0
    tau: float = 0.3
    T: int = 4
    B: float = 0.0
    arrival: tuple[int, ...] = (1, 1, 1, 1)   # batch (1-based) signal i arrives
    Pi: float = 3.0                            # total private precision

    @property
    def sigma_eps(self) -> float:
        return float(np.sqrt(self.N / self.Pi))


def staggered_times(N: int, T: int, phi: float) -> tuple[int, ...]:
    """Arrival schedule with a fraction phi of the N signals arriving after
    batch 1, spread evenly across batches 2..T (late arrivals only)."""
    n_late = int(round(phi * N))
    assert n_late < N, "at least one signal must exist at batch 1"
    if n_late == 0:
        late = []
    elif n_late == 1:
        late = [int(round((2 + T) / 2))]
    else:
        late = [int(round(x)) for x in np.linspace(2, T, n_late)]
    return tuple([1] * (N - n_late) + late)


def solve_honest_dynamics_staggered(p: StaggeredParams) -> dict:
    """Belief pass with heterogeneous signal-arrival times.

    Trader i trades 0 before batch t_i.  Active traders play x_i = beta_{i,t} *
    (s_i - E[s_i | F_{t-1}]); per-round betas solve the myopic-Nash linear
    system given lambda_t,
        2 lam b_i + lam sum_{j != i} b_j gamma_ji = kappa_i,
        gamma_ji = Cov(shat_j, shat_i)/Var(shat_i), kappa_i = Cov(v_res, shat_i)/Var(shat_i),
    with lambda_t = Cov(v_res, y_t)/Var(y_t) closed by damped fixed point.
    Per-trader eps functionals Grows_i (E[eps_i|F] is no longer symmetric)."""
    N, T = p.N, p.T
    se2, su2 = p.sigma_eps**2, p.sigma_u**2
    var = np.concatenate([[1.0], np.full(N, se2), np.full(T, su2)])
    A = _Aff(var)
    v = A.unit(0)
    eps = [A.unit(1 + i) for i in range(N)]
    s = [v + e for e in eps]
    u = [A.unit(1 + N + t) for t in range(T)]

    Mrows = [np.zeros(T)]
    Grows = [np.zeros((N, T))]
    betas = np.zeros((T, N))     # beta_{t,i}; 0 when inactive
    lams = []
    obs_aff = []

    m_prev = A.zero()
    ehat_prev = [A.zero() for _ in range(N)]
    for t in range(T):
        active = [i for i in range(N) if p.arrival[i] <= t + 1]
        assert active, "no active informed trader in batch 1"
        shat = {i: s[i] - m_prev - ehat_prev[i] for i in active}
        v_res = v - m_prev
        var_i = {i: A.cov(shat[i], shat[i]) for i in active}
        kap = np.array([A.cov(v_res, shat[i]) / var_i[i] for i in active])
        n_a = len(active)
        Gam = np.zeros((n_a, n_a))   # Gam[a, b] = gamma_{jb -> ia}
        for a, i in enumerate(active):
            for b, j in enumerate(active):
                if i != j:
                    Gam[a, b] = A.cov(shat[j], shat[i]) / var_i[i]

        # damped fixed point on lambda; beta linear-solved given lambda
        lam = lams[-1] if lams else 0.4
        b_vec = None
        for _ in range(500):
            LHS = lam * (2.0 * np.eye(n_a) + Gam)
            b_vec = np.linalg.solve(LHS, kap)
            y_aff = sum((b_vec[a] * shat[i] for a, i in enumerate(active)),
                        A.zero()) + u[t]
            lam_new = A.cov(v_res, y_aff) / A.cov(y_aff, y_aff)
            if abs(lam_new - lam) < 1e-13:
                lam = lam_new
                break
            lam = 0.5 * lam + 0.5 * lam_new
        else:
            raise RuntimeError("lambda fixed point did not converge")
        assert lam > 0
        # verify the myopic FOCs hold at the fixed point
        for a, i in enumerate(active):
            foc = kap[a] - 2.0 * lam * b_vec[a] - lam * float(Gam[a] @ b_vec)
            assert abs(foc) < 1e-9, ("staggered FOC violated", t, i, foc)
        lams.append(lam)
        for a, i in enumerate(active):
            betas[t, i] = b_vec[a]

        y_t = sum((betas[t, i] * shat[i] for i in active), A.zero()) + u[t]
        assert all(abs(A.cov(y_t, o)) < 1e-10 for o in obs_aff), \
            "flow innovation not orthogonal to past flows"
        var_yt = A.cov(y_t, y_t)
        cv = A.cov(v_res, y_t) / var_yt
        assert abs(lam - cv) < 1e-9, (lam, cv)
        e_k = np.zeros(T)
        e_k[t] = 1.0
        Mrows.append(Mrows[-1] + cv * e_k)
        Gnew = Grows[-1].copy()
        for i in range(N):
            ce = A.cov(eps[i] - ehat_prev[i], y_t) / var_yt
            Gnew[i] += ce * e_k
            ehat_prev[i] = ehat_prev[i] + ce * y_t
        Grows.append(Gnew)
        m_prev = m_prev + cv * y_t
        obs_aff.append(y_t)

    return {"p": p, "A": A, "betas": betas, "lams": lams,
            "Mrows": Mrows, "Grows": Grows, "obs_aff": obs_aff, "v": v,
            "schedule": [("y", t) for t in range(T)],
            "price_step": list(range(1, T + 1))}


def actual_pass_staggered(dyn: dict, alphas: np.ndarray) -> dict:
    p: StaggeredParams = dyn["p"]
    N, T = p.N, p.T
    A: _Aff = dyn["A"]
    betas, Mrows, Grows = dyn["betas"], dyn["Mrows"], dyn["Grows"]
    v = dyn["v"]
    eps = [A.unit(1 + i) for i in range(N)]
    s = [v + e for e in eps]
    u = [A.unit(1 + N + t) for t in range(T)]

    obs_act: list[np.ndarray] = []
    prices = []
    for t in range(T):
        m_prev = sum((Mrows[t][k] * obs_act[k] for k in range(t)), A.zero())
        push = A.zero()
        push[-1] = alphas[t]
        y_t = u[t] + push
        for i in range(N):
            if betas[t, i] != 0.0:
                ehat_i = sum((Grows[t][i, k] * obs_act[k] for k in range(t)),
                             A.zero())
                y_t = y_t + betas[t, i] * (s[i] - m_prev - ehat_i)
        obs_act.append(y_t)
        prices.append(sum((Mrows[t + 1][k] * obs_act[k] for k in range(t + 1)),
                          A.zero()))
    return {"obs": obs_act, "prices": prices}


# --------------------------------------------------------------------------
# shared evaluation / solver plumbing (works for both variants)
# --------------------------------------------------------------------------

def _actual_pass(dyn: dict, alphas: np.ndarray) -> dict:
    if isinstance(dyn["p"], StaggeredParams):
        return actual_pass_staggered(dyn, alphas)
    return actual_pass_arrival(dyn, alphas)


def evaluate(dyn: dict, alphas: np.ndarray, statistic: str) -> dict:
    """Exact metrics for a given push vector (mirrors twap.evaluate)."""
    p = dyn["p"]
    A: _Aff = dyn["A"]
    act = _actual_pass(dyn, alphas)
    w = stat_weights(p.T, statistic)
    P = sum((w[t] * act["prices"][t] for t in range(p.T)), A.zero())
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


def push_response(dyn: dict) -> dict:
    """Push-response precompute (same contract as twap.push_response, so
    twap.solve_manipulator_fast works unchanged on the result)."""
    p = dyn["p"]
    T = p.T
    A: _Aff = dyn["A"]
    D = np.zeros((T, T))
    for r in range(T):
        e = np.zeros(T)
        e[r] = 1.0
        act = _actual_pass(dyn, e)
        D[:, r] = [A.mean(pr) for pr in act["prices"]]
    prices0 = _actual_pass(dyn, np.zeros(T))["prices"]
    cov_pp = np.array([[A.cov(prices0[i], prices0[j]) for j in range(T)]
                       for i in range(T)])
    cov_vp = np.array([A.cov(dyn["v"], prices0[t]) for t in range(T)])
    return {"p": p, "D": D, "cov_pp": cov_pp, "cov_vp": cov_vp}


def evaluate_mixture(dyn: dict, alphas: np.ndarray, statistics: list[str],
                     B: float, probs: list[float] | None = None) -> dict:
    """Concealed-window mixture metrics (mirrors twap.evaluate_mixture)."""
    pis = (np.full(len(statistics), 1.0 / len(statistics)) if probs is None
           else np.asarray(probs))
    per = {s: evaluate(dyn, alphas, s) for s in statistics}
    dq = float(sum(pi * per[s]["decision_quality"] for s, pi in zip(statistics, pis)))
    ap = float(sum(pi * per[s]["approval_prob"] for s, pi in zip(statistics, pis)))
    trading = per[statistics[0]]["manip_trading_pnl"]
    return {"per_statistic": per, "decision_quality_mix": dq,
            "approval_prob_mix": ap, "manip_trading_pnl": trading,
            "manip_total": trading + B * ap}


def push_decay(pr: dict, push_batch: int = 0) -> dict:
    """Decay path of a unit push injected at `push_batch` (0-based): the bias
    it leaves in every later batch price, per-batch survival ratios, and the
    implied half-life (batches for the bias to halve, from the mean geometric
    decay rate over the remaining window)."""
    path = pr["D"][:, push_batch].copy()
    b0 = path[push_batch]
    ratios = [path[t + 1] / path[t]
              for t in range(push_batch, len(path) - 1) if path[t] > 1e-14]
    if not ratios or min(ratios) <= 0:
        half_life = np.inf
    else:
        rate = -float(np.mean(np.log(ratios)))
        half_life = np.log(2.0) / rate if rate > 0 else np.inf
    return {"bias_path": list(path), "impact": float(b0),
            "survival_ratios": [float(r) for r in ratios],
            "half_life_batches": float(half_life)}


def solve_scalar_schedule(pr: dict, B: float, statistic: str,
                          shape: np.ndarray) -> tuple[np.ndarray, float]:
    """Manipulator restricted to alpha = a * shape (a >= 0): named-schedule
    comparison (re-time-to-window-start vs sustained push).  Returns the
    optimal alpha vector and its exact objective value."""
    from scipy import optimize
    p = pr["p"]
    D = pr["D"]
    w = stat_weights(p.T, statistic)
    sd = float(np.sqrt(w @ pr["cov_pp"] @ w))

    def negU(a):
        al = a * shape
        Da = D @ al
        return -(-float(al @ Da) + B * E_q(float(w @ Da), sd, p.tau))

    res = optimize.minimize_scalar(negU, bounds=(0.0, 50.0), method="bounded",
                                   options={"xatol": 1e-12})
    return res.x * shape, -res.fun


def schedule_shapes(T: int, K: int) -> dict[str, np.ndarray]:
    """Named push schedules for a win:K read.  "retime": all mass at the
    window's opening batch (the Q4b attack); "sustained": uniform over the
    window; "last": all mass in the final batch."""
    retime = np.zeros(T)
    retime[T - K] = 1.0
    sustained = np.zeros(T)
    sustained[T - K:] = 1.0 / K
    last = np.zeros(T)
    last[-1] = 1.0
    return {"retime": retime, "sustained": sustained, "last": last}


# --------------------------------------------------------------------------
# Monte Carlo verification
# --------------------------------------------------------------------------

def mc_check_arrival(dyn: dict, alphas: np.ndarray, statistic: str,
                     n: int = 400_000, seed: int = 3) -> dict:
    """MC verification of the affine propagation, public-stream variant."""
    p: ArrivalParams = dyn["p"]
    N, T = p.N, p.T
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    eps = rng.standard_normal((N, n)) * p.sigma_eps
    s = v[None] + eps
    Mrows, Grows, betas = dyn["Mrows"], dyn["Grows"], dyn["betas"]
    n_obs = T + p.n_z
    obs = np.zeros((n_obs, n))
    prices = np.zeros((T, n))
    k = 0
    for t in range(T):
        if p.n_z and t >= 1:
            obs[k] = v + rng.standard_normal(n) * p.sigma_z
            k += 1
        m_prev = Mrows[k][:k] @ obs[:k] if k else np.zeros(n)
        e_prev = Grows[k][:k] @ obs[:k] if k else np.zeros(n)
        x = betas[t] * (s - m_prev[None] - e_prev[None])
        obs[k] = x.sum(axis=0) + rng.standard_normal(n) * p.sigma_u + alphas[t]
        k += 1
        prices[t] = Mrows[k][:k] @ obs[:k]
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


def mc_check_staggered(dyn: dict, alphas: np.ndarray, statistic: str,
                       n: int = 400_000, seed: int = 3) -> dict:
    """MC verification of the affine propagation, staggered variant."""
    p: StaggeredParams = dyn["p"]
    N, T = p.N, p.T
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    eps = rng.standard_normal((N, n)) * p.sigma_eps
    s = v[None] + eps
    Mrows, Grows, betas = dyn["Mrows"], dyn["Grows"], dyn["betas"]
    ys = np.zeros((T, n))
    prices = np.zeros((T, n))
    for t in range(T):
        m_prev = Mrows[t][:t] @ ys[:t] if t else np.zeros(n)
        flow = rng.standard_normal(n) * p.sigma_u + alphas[t]
        for i in range(N):
            if betas[t, i] != 0.0:
                e_prev = Grows[t][i, :t] @ ys[:t] if t else np.zeros(n)
                flow = flow + betas[t, i] * (s[i] - m_prev - e_prev)
        ys[t] = flow
        prices[t] = Mrows[t + 1][:t + 1] @ ys[:t + 1]
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
