"""Market mechanisms: SEQ-LMSR, BATCH-LMSR, BATCH-KYLE.

All three run the same trader population (myopic-Bayes, see envs.py) on the
same environment draws (common random numbers), for R rounds. Prices start at
0.5 (the uniform prior in both environments).

Mechanism definitions (precise; unit-tested in tests/test_clearing.py):

SEQ-LMSR  — the galanis-market protocol. Each round, traders act one at a
  time in fixed seat order 0..N-1. On their turn a trader moves the LMSR
  price from p to their target t, paying C(t)-C(p) for b*(logit t - logit p)
  Yes-shares. Honest target = posterior price; manipulator target solves the
  myopic FOC below.

BATCH-LMSR — each round, every trader simultaneously observes the posted
  price p (the LMSR marginal price) and submits a market order
      x_i = scale * b * (logit t_i - logit p),
  i.e. (scale times) the exact trade they would have made alone against the
  curve. Orders are NETTED: X = sum_i x_i. Clearing rule:
    * The AMM absorbs only the NET flow: new state p' with
      logit p' = logit p + X/b.
    * Uniform clearing price  pi = [C(p') - C(p)] / X  (the average price of
      the net execution along the curve);  pi = p (mid) when X = 0.
      pi is continuous at X=0 and always lies between p and p'.
    * Every order fills in full at pi: trader i pays pi*x_i for x_i shares.
      Cash conservation: sum_i pi*x_i = pi*X = C(p')-C(p), exactly the AMM's
      cost of the net move; offsetting flow crosses trader-to-trader at pi.
  If p' would leave [PRICE_EPS, 1-PRICE_EPS], the price is capped and ALL
  orders are scaled pro-rata by alpha = X_exec/X_submitted (rare; documented).

  `sizing`:
    * "full"        — scale = 1. Each trader submits the full move to their
      posterior (the literal sequential trade). Satisfies the single-trader
      equivalence exactly, but N like-minded traders overshoot jointly
      (net logit move = sum of individual moves), and across rounds the
      overshoot oscillates divergently — reported as a finding.
    * "competitive" — scale = 1/N. Each trader submits 1/N of the move,
      i.e. expects (correctly, in the symmetric myopic model) to be one of N
      traders pushing the same way. At N=1 this equals "full", so the
      single-trader equivalence still holds; N identical targets clear
      exactly AT the target. This is the headline batch arm.

BATCH-KYLE — same batch protocol against a linear conditional-expectation
  market maker instead of a curve (bridge case). Posted price p; depth
  matched to the LMSR's local depth: lambda = p(1-p)/b (the LMSR's dp/dq at
  p). Orders x_i = scale_k * (t_i - p) / (2 lambda), the myopic optimum
  against own linear impact with uniform pricing (scale_k = 1 for "full",
  2/(N+1) for "competitive"; equal at N=1). Clearing: single uniform price
  pi = p' = clip(p + lambda * X); ALL fills (and the MM's -X position) at pi.
  Note the Kyle MM fills at the FULL post-impact price, not the curve's
  average price — the cost of linearity is borne by the traders; the MM has
  no convexity revenue.

Manipulator (optional, one seat): utility = market PnL + bounty*(p_final-0.5),
myopically treating the price they induce as final. Against the LMSR the FOC
  b*(q - t)/(t(1-t)) + B = 0   =>   B t^2 - (B-b) t - b q = 0,
with q their honest posterior; the unique root in (0,1) is
  t* = [(B-b) + sqrt((B-b)^2 + 4 B b q)] / (2B)   (t* -> q as B -> 0).
Against the Kyle MM: order y* = scale_k * (q - p + B*lambda) / (2 lambda).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from batch_amm import lmsr_np as lmsr
from batch_amm.envs import PRICE_EPS, clip_price

_LOGIT_MAX = float(lmsr.logit(1.0 - PRICE_EPS))


def manip_target_lmsr(q: np.ndarray, b: float, bounty: float) -> np.ndarray:
    """Myopic manipulator's optimal LMSR price target (vectorised)."""
    q = np.asarray(q, dtype=float)
    if bounty <= 0.0:
        return q.copy()
    a = bounty
    disc = (a - b) ** 2 + 4.0 * a * b * q
    t = ((a - b) + np.sqrt(disc)) / (2.0 * a)
    return clip_price(t)


@dataclass
class Config:
    mech: str  # "seq_lmsr" | "batch_lmsr" | "batch_kyle"
    rounds: int = 1
    b: float = 0.1
    sizing: str = "competitive"  # "full" | "competitive" (batch arms only)
    manip_seat: Optional[int] = None
    bounty: float = 0.0
    # what traders observe between batch rounds:
    #   "full"      — per-trader orders (attributed)
    #   "aggregate" — clearing price + net flow only (pseudo-anonymous)
    #   "price"     — clearing price only; identical information to
    #                 "aggregate" against a deterministic AMM (the price move
    #                 is an invertible function of the net flow) — kept as a
    #                 separate value and unit-tested equal
    disclosure: str = "full"

    def __post_init__(self):
        assert self.mech in ("seq_lmsr", "batch_lmsr", "batch_kyle")
        assert self.sizing in ("full", "competitive")
        assert self.disclosure in ("full", "aggregate", "price")


def run_market(env, cfg: Config) -> Dict[str, np.ndarray]:
    """Run one market per environment rep; return per-rep records.

    Returns dict with:
      final_price (M,), twap (M,), round_prices (R, M),
      cash (M,N), holdings (M,N), mm_cash (M,),
      slip_own (M,N)   — execution cost above the trader's ARRIVAL price
                          (turn-open price in SEQ, posted price in BATCH),
      slip_round (M,N) — execution cost above the ROUND-OPEN price,
      volume (M,N)     — sum |shares| traded.
    """
    n, m, b = env.n, env.m, cfg.b
    lo, hi = getattr(env, "target_bounds", (PRICE_EPS, 1.0 - PRICE_EPS))

    def cap(t):
        # all posted quotes respect the env's quote bounds (e.g. the 0.1/0.9
        # tabular-grid floor in the Galanis env); the market price itself may
        # exceed them in batch (netted flow of several capped quotes)
        return np.clip(t, lo, hi)

    state = env.make_state()
    p = np.full(m, 0.5)
    cash = np.zeros((m, n))
    holdings = np.zeros((m, n))
    mm_cash = np.zeros(m)
    slip_own = np.zeros((m, n))
    slip_round = np.zeros((m, n))
    volume = np.zeros((m, n))
    round_prices = np.zeros((cfg.rounds, m))

    def targets_all() -> np.ndarray:
        """(N, M) targets; manipulator seat replaced."""
        ts = np.stack([env.honest_target(i, state) for i in range(n)])
        if cfg.manip_seat is not None and cfg.mech != "batch_kyle":
            ts[cfg.manip_seat] = manip_target_lmsr(
                ts[cfg.manip_seat], b, cfg.bounty
            )
        return cap(ts)

    def disclose(implied_targets: np.ndarray, first: bool) -> None:
        """Between-round information release per cfg.disclosure."""
        if cfg.disclosure == "full":
            env.reveal_batch(implied_targets, state, first_time=first)
        else:
            # "aggregate" and "price" both reduce to the statistic
            # T = sum_i logit(implied target_i): the net flow reveals T given
            # the common-knowledge sizing rule, and the clearing price
            # reveals the net flow (deterministic AMM).
            t_total = lmsr.logit(implied_targets).sum(axis=0)
            env.reveal_batch_anon(t_total, state, first_time=first)

    for r in range(cfg.rounds):
        first = r == 0
        p_round_open = p.copy()

        if cfg.mech == "seq_lmsr":
            for i in range(n):
                q = env.honest_target(i, state)
                if cfg.manip_seat == i:
                    t = cap(manip_target_lmsr(q, b, cfg.bounty))
                else:
                    t = cap(q)
                shares = lmsr.shares_to_move(p, t, b)
                cost = lmsr.cost_to_move(p, t, b)
                cash[:, i] -= cost
                holdings[:, i] += shares
                mm_cash += cost
                slip_own[:, i] += cost - p * shares
                slip_round[:, i] += cost - p_round_open * shares
                volume[:, i] += np.abs(shares)
                env.reveal(i, t, state, first_time=first)
                p = t.copy()

        elif cfg.mech == "batch_lmsr":
            scale = 1.0 if cfg.sizing == "full" else 1.0 / n
            ts = targets_all()
            lp = lmsr.logit(p)
            x = scale * b * (lmsr.logit(ts) - lp[None, :])  # (N, M)
            x_net = x.sum(axis=0)
            lp1 = np.clip(lp + x_net / b, -_LOGIT_MAX, _LOGIT_MAX)
            x_exec_net = b * (lp1 - lp)
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.where(x_net != 0.0, x_exec_net / x_net, 1.0)
            x_exec = x * alpha[None, :]
            p1 = lmsr.sigmoid(lp1)
            dc = lmsr.cost_to_move(p, p1, b)
            small = np.abs(x_exec_net) < 1e-14
            with np.errstate(divide="ignore", invalid="ignore"):
                pi = np.where(small, p, dc / np.where(small, 1.0, x_exec_net))
            cash -= (pi[None, :] * x_exec).T
            holdings += x_exec.T
            mm_cash += dc
            slip_own += ((pi - p)[None, :] * x_exec).T
            slip_round += ((pi - p_round_open)[None, :] * x_exec).T
            volume += np.abs(x_exec).T
            disclose(ts, first)
            p = p1

        elif cfg.mech == "batch_kyle":
            scale_k = 1.0 if cfg.sizing == "full" else 2.0 / (n + 1)
            lam = p * (1.0 - p) / b  # matched local LMSR depth at posted price
            ts = np.stack([env.honest_target(i, state) for i in range(n)])
            x = scale_k * (ts - p[None, :]) / (2.0 * lam[None, :])
            if cfg.manip_seat is not None:
                i = cfg.manip_seat
                t_m = cap(ts[i] + cfg.bounty * lam)  # implied manip quote, capped
                x[i] = scale_k * (t_m - p) / (2.0 * lam)
            x_net = x.sum(axis=0)
            p1 = clip_price(p + lam * x_net)
            x_exec_net = (p1 - p) / lam
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.where(x_net != 0.0, x_exec_net / x_net, 1.0)
            x_exec = x * alpha[None, :]
            pi = p1
            cash -= (pi[None, :] * x_exec).T
            holdings += x_exec.T
            mm_cash += pi * x_exec_net
            slip_own += ((pi - p)[None, :] * x_exec).T
            slip_round += ((pi - p_round_open)[None, :] * x_exec).T
            volume += np.abs(x_exec).T
            implied = p[None, :] + 2.0 * lam[None, :] * x / scale_k
            disclose(cap(implied), first)
            p = p1

        round_prices[r] = p

    return {
        "final_price": p,
        "twap": round_prices.mean(axis=0),
        "round_prices": round_prices,
        "cash": cash,
        "holdings": holdings,
        "mm_cash": mm_cash,
        "slip_own": slip_own,
        "slip_round": slip_round,
        "volume": volume,
    }


__all__ = ["Config", "run_market", "manip_target_lmsr"]
