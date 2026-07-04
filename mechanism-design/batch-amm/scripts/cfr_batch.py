"""Stretch check: equilibrium (CFR+) play of the one-shot simultaneous-move
BATCH-LMSR game on the Galanis "Easy" structure (t3s111y2), to verify the
myopic-trader headline results are not artifacts of myopic play.

Game (R=1 batch round, full sizing, price targets on the 9-point grid
{0.1..0.9} — the same discretisation as galanis-market's sequential CFR+):

  * Nature draws omega uniform on {0,1}^3; trader i observes bit i.
  * All three traders SIMULTANEOUSLY pick a target price t_i from the grid
    and submit the market order x_i = b*(logit t_i - logit 0.5).
  * Batch clearing as in engine.py: net X = sum x_i moves the LMSR to
    logit p1 = X/b; uniform fill price pi = [C(p1)-C(0.5)]/X (pi = 0.5 at
    X = 0); pnl_i = x_i * (X(omega) - pi).
  * Optional manipulator: player 0's utility += bounty * (p1 - 0.5).

Note that with a target grid and full sizing, ORDER SIZE is endogenous to
the equilibrium: a trader who wants a small order picks a target near 0.5.
So CFR+ here searches over sizings too — no damping convention is imposed.

Solver: exact simultaneous-move CFR+ (regret matching+, alternating updates,
linearly weighted average strategy) on the full payoff tensor
(8 states x 9^3 joint actions). Infosets: 2 per player (own bit), 6 total —
versus 2 + 18 + 162 = 182 infosets for the R=1 SEQUENTIAL game at the same
grid (player i's infoset also encodes the i previous quotes), which is the
solve-cost scaling headline.

Exploitability (NashConv) is computed exactly. In 3-player general-sum games
CFR+ converges to a coarse correlated equilibrium; NashConv is reported so
the profile can be judged as a near-equilibrium (same caveat as
mechanism-design/MANIPULATION.md).

Writes results/cfr_batch.json.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from galanis_market.structures import STATES, STRUCTURES

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

B_LIQ = 0.01  # matches galanis-market CFR+ setup
GRID = np.round(np.arange(0.1, 0.91, 0.1), 6)  # 9 actions
NA = len(GRID)
NP_ = 3
ITERS = 4000


def logit(p):
    return np.log(p / (1.0 - p))


def build_tensors(bounty: float):
    """U[i][w, a0, a1, a2] utilities; plus clearing stats tensors."""
    struct = STRUCTURES["t3s111y2"]
    payout = np.array([struct.x_of(STATES[w]) for w in range(8)], dtype=float)
    lg = logit(GRID)
    x = B_LIQ * lg  # order per action (from p0 = 0.5)
    # joint net flow X[a0,a1,a2]
    Xn = x[:, None, None] + x[None, :, None] + x[None, None, :]
    l1 = Xn / B_LIQ
    p1 = 1.0 / (1.0 + np.exp(-l1))
    dc = -B_LIQ * np.log1p(-p1) + B_LIQ * np.log(0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(np.abs(Xn) < 1e-15, 0.5, dc / np.where(np.abs(Xn) < 1e-15, 1.0, Xn))
    U = []
    for i in range(NP_):
        shape = [1, 1, 1]
        shape[i] = NA
        xi = x.reshape(shape)
        # (w, a0, a1, a2)
        ui = xi[None, ...] * (payout[:, None, None, None] - pi[None, ...])
        if i == 0 and bounty > 0:
            ui = ui + bounty * (p1[None, ...] - 0.5)
        U.append(ui)
    leak = ((pi - 0.5) * Xn)[None, ...] * np.ones((8, 1, 1, 1))
    return U, p1, payout, leak


def bits(w):
    return STATES[w]


def solve(bounty: float):
    U, p1, payout, leak = build_tensors(bounty)
    # infoset = (player, own bit); sigma[(i, bit)] = distribution over actions
    regrets = {(i, c): np.zeros(NA) for i in range(NP_) for c in (0, 1)}
    strat_sum = {(i, c): np.zeros(NA) for i in range(NP_) for c in (0, 1)}

    def current_sigma():
        sig = {}
        for k, r in regrets.items():
            pos = np.maximum(r, 0.0)
            z = pos.sum()
            sig[k] = pos / z if z > 0 else np.full(NA, 1.0 / NA)
        return sig

    def sigma_for_state(sig, w):
        return [sig[(i, bits(w)[i])] for i in range(NP_)]

    t0 = time.time()
    for it in range(1, ITERS + 1):
        sig = current_sigma()
        for i in range(NP_):
            # counterfactual expected utility per own action, per own bit
            ev = {0: np.zeros(NA), 1: np.zeros(NA)}
            for w in range(8):
                s = sigma_for_state(sig, w)
                # contract U[i][w] over others' strategies, own axis first
                u = np.moveaxis(U[i][w], i, 0)  # (NA_own, ...)
                for j in (j for j in range(NP_) if j != i):
                    u = np.tensordot(u, s[j], axes=([1], [0]))
                ev[bits(w)[i]] += u / 8.0
            for c in (0, 1):
                mean_u = float(ev[c] @ sig[(i, c)])
                regrets[(i, c)] = np.maximum(
                    regrets[(i, c)] + (ev[c] - mean_u), 0.0
                )  # RM+
                strat_sum[(i, c)] += it * sig[(i, c)]  # linear averaging

    avg = {}
    for k, ssum in strat_sum.items():
        z = ssum.sum()
        avg[k] = ssum / z if z > 0 else np.full(NA, 1.0 / NA)

    # exact expected values and exploitability under avg strategy
    def expected(tensor_w, sig):
        """E over omega and joint actions of tensor (8, NA, NA, NA)."""
        tot = 0.0
        for w in range(8):
            s = sigma_for_state(sig, w)
            v = tensor_w[w]
            for j in range(NP_):
                v = np.tensordot(s[j], v, axes=([0], [0]))
            tot += float(v) / 8.0
        return tot

    nashconv = 0.0
    values = []
    for i in range(NP_):
        # current value
        vi = expected(U[i], avg)
        values.append(vi)
        # best response value: per bit, maximise counterfactual EV
        ev = {0: np.zeros(NA), 1: np.zeros(NA)}
        for w in range(8):
            s = sigma_for_state(avg, w)
            u = np.moveaxis(U[i][w], i, 0)
            rest = [j for j in range(NP_) if j != i]
            for j in rest:
                u = np.tensordot(u, s[j], axes=([1], [0]))
            ev[bits(w)[i]] += u / 8.0
        br = sum(float(ev[c].max()) for c in (0, 1))
        nashconv += max(0.0, br - vi)

    # market-quality metrics under avg strategy
    ll_t = -np.where(
        payout[:, None, None, None] > 0.5,
        np.log(np.clip(p1[None], 1e-9, 1)),
        np.log1p(-np.clip(p1[None], 0, 1 - 1e-9)),
    )
    acc_t = (
        (p1[None] >= 0.5).astype(float) == payout[:, None, None, None]
    ).astype(float)
    dpf_t = p1[None] * np.ones((8, 1, 1, 1)) - 0.5
    metrics = {
        "log_loss_final": expected(ll_t, avg),
        "decision_acc": expected(acc_t, avg),
        "leak_own": expected(leak, avg),
        "mean_final_price_minus_half": expected(dpf_t, avg),
        "player_values": values,
        "nashconv": nashconv,
        "solve_seconds": time.time() - t0,
    }
    strategy = {
        f"p{i}_bit{c}": [round(float(v), 4) for v in avg[(i, c)]]
        for i in range(NP_)
        for c in (0, 1)
    }
    return metrics, strategy


def main():
    out = {
        "grid": GRID.tolist(),
        "b": B_LIQ,
        "iters": ITERS,
        "infosets_batch": 6,
        "infosets_sequential_same_grid_R1": 2 + 2 * 9 + 2 * 81,
        "runs": [],
    }
    for bounty in (0.0, 0.05, 0.2):
        m, s = solve(bounty)
        print(
            f"bounty={bounty}: LL={m['log_loss_final']:.4f} acc={m['decision_acc']:.4f} "
            f"leak={m['leak_own']:.6f} dPf={m['mean_final_price_minus_half']:+.4f} "
            f"NashConv={m['nashconv']:.2e} ({m['solve_seconds']:.1f}s)",
            flush=True,
        )
        out["runs"].append({"bounty": bounty, "metrics": m, "avg_strategy": s})
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "cfr_batch.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
