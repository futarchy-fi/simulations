"""Summary metrics and paired (common-random-numbers) comparisons."""

from __future__ import annotations

from typing import Dict

import numpy as np

from batch_amm.envs import clip_price
from batch_amm.lmsr_np import logit


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / w.sum())


def _wse(x: np.ndarray, w: np.ndarray, exact: bool) -> float:
    """Standard error of the weighted mean (0 for exact enumeration)."""
    if exact:
        return 0.0
    m = len(x)
    mu = _wmean(x, w)
    var = _wmean((x - mu) ** 2, w)
    return float(np.sqrt(var / m))


def summarize(env, res: Dict[str, np.ndarray]) -> Dict:
    w = env.weights
    payout = env.payout
    pf = clip_price(res["final_price"])
    twap = clip_price(res["twap"])
    p_star = env.full_info_price()

    pnl = res["holdings"] * payout[:, None] + res["cash"]  # (M, N)
    mm_pnl = res["mm_cash"] - res["holdings"].sum(axis=1) * payout
    conservation = float(np.abs(pnl.sum(axis=1) + mm_pnl).max())

    log_loss_final = -np.where(payout > 0.5, np.log(pf), np.log1p(-pf))
    log_loss_twap = -np.where(payout > 0.5, np.log(twap), np.log1p(-twap))
    logit_err_final = np.abs(logit(pf) - logit(p_star))
    decision_acc = ((pf >= 0.5).astype(float) == payout).astype(float)

    leak_own = res["slip_own"].sum(axis=1)  # value paid above arrival price
    leak_round = res["slip_round"].sum(axis=1)
    vol = res["volume"].sum(axis=1)

    out = {
        "trader_pnl_mean": [
            _wmean(pnl[:, i], w) for i in range(env.n)
        ],
        "trader_pnl_se": [
            _wse(pnl[:, i], w, env.exact) for i in range(env.n)
        ],
        "trader_pnl_total": _wmean(pnl.sum(axis=1), w),
        "trader_pnl_total_se": _wse(pnl.sum(axis=1), w, env.exact),
        "mm_loss": -_wmean(mm_pnl, w),
        "mm_loss_se": _wse(-mm_pnl, w, env.exact),
        "leak_own": _wmean(leak_own, w),
        "leak_own_se": _wse(leak_own, w, env.exact),
        "leak_round": _wmean(leak_round, w),
        "volume": _wmean(vol, w),
        "leak_per_volume": _wmean(leak_own, w) / max(_wmean(vol, w), 1e-300),
        "log_loss_final": _wmean(log_loss_final, w),
        "log_loss_final_se": _wse(log_loss_final, w, env.exact),
        "log_loss_twap": _wmean(log_loss_twap, w),
        "logit_err_final": _wmean(logit_err_final, w),
        "logit_err_final_se": _wse(logit_err_final, w, env.exact),
        "decision_acc": _wmean(decision_acc, w),
        "mean_final_price": _wmean(pf, w),
        "conservation_maxabs": conservation,
        "exact": env.exact,
    }
    return out


def paired_diff(env, res_a: Dict, res_b: Dict) -> Dict:
    """Paired (CRN) differences A - B on the headline per-rep statistics."""
    w = env.weights
    payout = env.payout
    out = {}
    for name, fa, fb in [
        (
            "leak_own",
            res_a["slip_own"].sum(axis=1),
            res_b["slip_own"].sum(axis=1),
        ),
        (
            "trader_pnl_total",
            (res_a["holdings"] * payout[:, None] + res_a["cash"]).sum(axis=1),
            (res_b["holdings"] * payout[:, None] + res_b["cash"]).sum(axis=1),
        ),
        (
            "log_loss_final",
            -np.where(
                payout > 0.5,
                np.log(clip_price(res_a["final_price"])),
                np.log1p(-clip_price(res_a["final_price"])),
            ),
            -np.where(
                payout > 0.5,
                np.log(clip_price(res_b["final_price"])),
                np.log1p(-clip_price(res_b["final_price"])),
            ),
        ),
        ("final_price", res_a["final_price"], res_b["final_price"]),
    ]:
        d = fa - fb
        out[name] = {
            "diff": _wmean(d, w),
            "se": _wse(d, w, env.exact),
        }
    return out


__all__ = ["summarize", "paired_diff"]
