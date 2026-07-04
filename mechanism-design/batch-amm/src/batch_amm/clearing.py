"""Limit-order uniform-price clearing against the LMSR curve.

Orders are (signed quantity x_i, limit l_i): a buy (x_i > 0) fills only if
the uniform clearing price pi <= l_i, a sell (x_i < 0) only if pi >= l_i;
l_i = +/-inf recovers a market order. The AMM absorbs only the executed net
flow X along the curve (logit p' = logit p + X/b) and every fill pays the
single price pi = [C(p') - C(p)] / X, the average execution price of the net
move — so pi = phi(X) with phi strictly increasing, phi(0) = p.

Clearing is the unique crossing of g(pi) = phi(D(pi)) - pi, where D(pi) is
the eligible net demand (a nonincreasing step function of pi: rising pi
drops buys and admits sells), i.e. g is strictly decreasing with g > 0 at
the lower price bound and g < 0 at the upper one. Two cases (the standard
uniform-price auction treatment):

* the crossing lands inside a constancy interval of D — a continuity fixed
  point pi* = phi(D(pi*)): all eligible orders fill in full;
* g jumps across zero at an order's limit — the clearing price IS that
  marginal limit, pi* = l*, the executed net flow solves phi(X*) = pi*, and
  the marginal orders (|l_i - pi*| <= 1e-9) fill pro-rata. If both sides are
  marginal at the same price, the light side fills in full first
  (volume-maximizing convention); mixed-sign ties are measure-zero here.

Price bounds: if the executed net flow would push logit p' outside
[-L, L] with L = logit(1 - price_eps), the move is capped and ALL fills are
scaled pro-rata by alpha = X_cap / X (the base engine's rationing; in that
rare case pi = phi(X_cap) can sit slightly inside a marginal limit).

The final quantities are computed with arithmetic identical to the plain
batch engine applied to x * fill, so with all limits at +/-inf the result
reproduces engine.py's batch_lmsr branch bit-for-bit (unit-tested).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from batch_amm import lmsr_np as lmsr

PRICE_EPS = 1e-4  # same global clip as envs.py (kept local: envs imports us)
_MARGINAL_TOL = 1e-9


def avg_price(p: np.ndarray, x_net: np.ndarray, b: float) -> np.ndarray:
    """phi(X): average execution price of net flow X from posted price p."""
    p = np.asarray(p, dtype=float)
    x_net = np.asarray(x_net, dtype=float)
    p1 = lmsr.sigmoid(lmsr.logit(p) + x_net / b)
    dc = lmsr.cost_to_move(p, p1, b)
    small = np.abs(x_net) < 1e-14
    return np.where(small, p, dc / np.where(small, 1.0, x_net))


def clear_limit_batch(
    p: np.ndarray,
    x: np.ndarray,
    limits: np.ndarray,
    b: float,
    price_eps: float = PRICE_EPS,
) -> Dict[str, np.ndarray]:
    """Clear one batch of limit orders against the LMSR posted at p.

    p (M,) posted prices; x (N, M) signed order quantities; limits (N, M)
    limit prices (+/-inf allowed). Returns fill (N, M) auction fill
    fractions in [0, 1], x_exec (N, M) executed quantities (fill plus the
    rare price-cap rationing), p1 (M,) post-clearing prices, pi (M,) uniform
    clearing prices.
    """
    p = np.asarray(p, dtype=float)
    x = np.asarray(x, dtype=float)
    limits = np.asarray(limits, dtype=float)
    n, m = x.shape
    lp = lmsr.logit(p)
    lmax = float(lmsr.logit(1.0 - price_eps))
    buys = x > 0.0
    sells = x < 0.0

    def phi(x_net):
        x_c = np.clip(x_net, b * (-lmax - lp), b * (lmax - lp))
        p1 = lmsr.sigmoid(lp + x_c / b)
        dc = lmsr.cost_to_move(p, p1, b)
        small = np.abs(x_c) < 1e-14
        return np.where(small, p, dc / np.where(small, 1.0, x_c))

    def demand(pi):
        elig = (buys & (limits >= pi[None, :])) | (sells & (limits <= pi[None, :]))
        return np.where(elig, x, 0.0).sum(axis=0)

    # ---- crossing price: bisect g(pi) = phi(D(pi)) - pi (strictly decreasing;
    # g > 0 at the lower bound and g < 0 at the upper one since phi stays
    # strictly inside the price bounds)
    lo = np.full(m, price_eps)
    hi = np.full(m, 1.0 - price_eps)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pos = phi(demand(mid)) - mid > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    pi_hat = 0.5 * (lo + hi)

    # ---- classify orders at the crossing
    marg = (buys | sells) & (np.abs(limits - pi_hat[None, :]) <= _MARGINAL_TOL)
    has_marg = marg.any(axis=0)
    in_b = buys & ~marg & (limits > pi_hat[None, :])
    in_s = sells & ~marg & (limits < pi_hat[None, :])
    d_strict = np.where(in_b | in_s, x, 0.0).sum(axis=0)

    fill = np.where(in_b | in_s, 1.0, 0.0)

    if has_marg.any():
        # jump case: pi* is the marginal limit itself (ties within tol share it)
        l_star = np.where(marg, limits, -np.inf).max(axis=0)
        pi_star = np.where(has_marg, l_star, pi_hat)
        b_m = np.where(marg & buys, x, 0.0).sum(axis=0)
        s_m = np.where(marg & sells, x, 0.0).sum(axis=0)
        # executed net flow at the pinned price: phi(X*) = pi*, X* in [D+, D-]
        x_lo = d_strict + s_m
        x_hi = d_strict + b_m
        for _ in range(80):
            xm = 0.5 * (x_lo + x_hi)
            low = phi(xm) - pi_star < 0.0
            x_lo = np.where(low, xm, x_lo)
            x_hi = np.where(low, x_hi, xm)
        delta = 0.5 * (x_lo + x_hi) - d_strict
        # pro-rata fractions; when both sides are marginal fill the light
        # side fully first (volume-maximizing)
        with np.errstate(divide="ignore", invalid="ignore"):
            bb_only = np.where(b_m != 0.0, delta / b_m, 0.0)
            bs_only = np.where(s_m != 0.0, delta / s_m, 0.0)
            bb_try = np.where(b_m != 0.0, (delta - s_m) / b_m, 0.0)
            bs_swap = np.where(s_m != 0.0, (delta - b_m) / s_m, 0.0)
        both = (b_m != 0.0) & (s_m != 0.0)
        beta_b = np.where(both, bb_try, bb_only)
        beta_s = np.where(both, np.where(bb_try > 1.0, bs_swap, 1.0), bs_only)
        beta_b = np.clip(np.where(both & (bb_try > 1.0), 1.0, beta_b), 0.0, 1.0)
        beta_s = np.clip(beta_s, 0.0, 1.0)
        fill = np.where(marg & buys, beta_b[None, :], fill)
        fill = np.where(marg & sells, beta_s[None, :], fill)

    # ---- final quantities: identical arithmetic to the plain batch engine
    # applied to x * fill (bit-exact market-order reproduction when no limit
    # binds), including the price-cap pro-rata rationing
    x_eff = x * fill
    x_net = x_eff.sum(axis=0)
    lp1 = np.clip(lp + x_net / b, -lmax, lmax)
    x_exec_net = b * (lp1 - lp)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(x_net != 0.0, x_exec_net / x_net, 1.0)
    x_exec = x_eff * alpha[None, :]
    p1 = lmsr.sigmoid(lp1)
    dc = lmsr.cost_to_move(p, p1, b)
    small = np.abs(x_exec_net) < 1e-14
    pi = np.where(small, p, dc / np.where(small, 1.0, x_exec_net))
    return {"fill": fill, "x_exec": x_exec, "p1": p1, "pi": pi}


__all__ = ["avg_price", "clear_limit_batch", "PRICE_EPS"]
