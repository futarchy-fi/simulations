"""Probabilistic implementation rule and Gaussian expectation helpers.

The proposal is approved with probability q(p) = logistic(p / tau)
= 1 / (1 + exp(-p/tau)). All expectations of q and its derivatives
against Gaussian laws are computed with Gauss-Hermite quadrature
(exact for polynomials up to degree 2n-1; q is smooth and bounded, so
n = 96 nodes gives ~machine-precision results for the scales used here).

A probit stand-in q(p) = Phi(kappa * p) with kappa = sqrt(pi/8)/tau
matches the logistic's value and slope at p = 0 and admits closed forms
(E[Phi(kappa N(m, w^2))] = Phi(kappa m / sqrt(1 + kappa^2 w^2))); it is
used for closed-form cross-checks only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_GH_N = 96
_gh_x, _gh_w = np.polynomial.hermite_e.hermegauss(_GH_N)  # weights for standard normal * sqrt(2pi)
_gh_w = _gh_w / np.sqrt(2.0 * np.pi)  # now sum(w) = 1, nodes ~ N(0,1)


def logistic_q(p: np.ndarray | float, tau: float) -> np.ndarray | float:
    """Approval probability q(p) = 1/(1+exp(-p/tau))."""
    from scipy.special import expit
    z = np.asarray(p, dtype=float) / tau
    out = expit(z)
    return out if out.shape else float(out)


def logistic_qprime(p: np.ndarray | float, tau: float) -> np.ndarray | float:
    q = logistic_q(p, tau)
    out = q * (1.0 - q) / tau
    return out


def E_q(m: float, w: float, tau: float) -> float:
    """E[q(P)] for P ~ N(m, w^2)."""
    return float(np.sum(_gh_w * logistic_q(m + w * _gh_x, tau)))


def E_qprime(m: float, w: float, tau: float) -> float:
    """E[q'(P)] for P ~ N(m, w^2)."""
    return float(np.sum(_gh_w * logistic_qprime(m + w * _gh_x, tau)))


def E_zq(m: float, w: float, tau: float) -> float:
    """E[(P-m) q(P)] for P ~ N(m, w^2)   (= w * E[Z q(m+wZ)])."""
    return float(w * np.sum(_gh_w * _gh_x * logistic_q(m + w * _gh_x, tau)))


def E_q_probit(m: float, w: float, tau: float) -> float:
    """Closed-form cross-check with the moment-matched probit rule."""
    kappa = np.sqrt(np.pi / 8.0) / tau
    return float(norm.cdf(kappa * m / np.sqrt(1.0 + (kappa * w) ** 2)))


def decision_quality(cov_vp: float, m_p: float, var_p: float, tau: float) -> float:
    """E[v * q(p)] when (v, p) are jointly Gaussian, E[v]=0, Var(v)=1.

    E[v q(p)] = E[E[v|p] q(p)] = (cov_vp / var_p) E[(p - m_p) q(p)].
    """
    if var_p <= 0:
        return 0.0
    w = np.sqrt(var_p)
    return cov_vp / var_p * E_zq(m_p, w, tau)


ORACLE_DQ = 1.0 / np.sqrt(2.0 * np.pi)  # E[v 1{v>0}], v ~ N(0,1)


def oracle_q_dq(tau: float, corr: float = 1.0) -> float:
    """E[v q(v_hat)] when the rule reads a price v_hat with corr(v_hat,v)=corr
    and the projection scaling of the Kyle MM (v_hat = E[v|...], so
    Var(v_hat) = Cov(v, v_hat) = corr^2). corr=1 gives the best any
    logistic-tau implementation can do."""
    s2 = corr * corr
    return decision_quality(cov_vp=s2, m_p=0.0, var_p=s2, tau=tau)
