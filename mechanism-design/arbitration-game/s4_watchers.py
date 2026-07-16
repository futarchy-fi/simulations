"""S4 (C5): watcher economics and the probability a bad YES slips through.

W watchers inspect each arriving proposal independently at the symmetric
break-even rate f solving  bad_frac * m * E[1/(1+K)] = c,  K ~ Bin(W-1, f)
(E[1/(1+K)] = (1-(1-f)^W)/(W f); the caught bond is paid to one of the
catchers). f = 0 when bad_frac*m <= c (nobody watches), f = 1 when watching
everything is still profitable. Inspection costs c are absolute (reading the
"{...}*m" grid at the reference bond m = 1): if c scaled with the swept m,
m would cancel out of the economics entirely. Per-inspection costs also make
the break-even rate lam-independent; lam only scales counts, reported as
expected bad slips per TIMEOUT. m* = c/bad_frac is the minimum bond making
watching self-financing.

Usage: python3 s4_watchers.py [--pilot]
"""

from __future__ import annotations

import argparse

import numpy as np

from game import Game, SETTLED
from report import print_table, write_json

SEED = 20260719
BAD_FRAC = 0.05


def break_even_rate(w, c, m):
    reward = BAD_FRAC * m

    def ev(f):
        return reward * (1 - (1 - f) ** w) / (w * f) - c

    if reward <= c:
        return 0.0
    if ev(1.0) > 0:
        return 1.0
    lo, hi = 1e-9, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ev(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_config(w, c, lam, m, M, seed):
    f = break_even_rate(w, c, m)
    g = Game(m=m, seed=seed)
    rng = np.random.default_rng(seed + 1)
    bads = slips = 0
    for i in range(M):
        bad = bool(rng.random() < BAD_FRAC)
        pid = g.new_proposal(quality=not bad, t=i)
        g.place_yes(i, pid, "prop", m)
        caught = rng.binomial(w, f) > 0
        if bad:
            bads += 1
            if caught:
                g.place_no(i, pid, "watch")
            elif g.finalize_by_timeout(i + g.timeout, pid):
                slips += 1
        if g.proposals[pid].state != SETTLED:  # settle the rest by timeout
            g.finalize_by_timeout(g.proposals[pid].last_change + g.timeout, pid)
    p_mc = slips / bads if bads else 0.0
    return {"W": w, "c": c, "lam": lam, "m": m, "f": f,
            "P_slip_mc": p_mc, "P_slip_an": (1 - f) ** w,
            "bad_slips_per_T": lam * BAD_FRAC * p_mc,
            "m_star": c / BAD_FRAC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        # includes an interior break-even point (W=10, c=0.01, m=1)
        ws, cs, lams, ms, M = [1, 10], [0.01, 0.1], [1, 100], [1.0, 10.0], 200
    else:
        ws, cs, lams, ms, M = [1, 2, 5, 10], [0.001, 0.01, 0.1], \
            [1, 10, 100], [0.1, 1.0, 10.0], 2000
    rows = []
    n = 0
    for w in ws:
        for c in cs:
            for lam in lams:
                for m in ms:
                    rows.append(run_config(w, c, lam, m, M, SEED + 37 * n))
                    n += 1
    path = write_json("s4_watchers",
                      {"pilot": args.pilot, "seed": SEED,
                       "bad_frac": BAD_FRAC}, rows)
    print("## S4 watcher break-even rates and P(bad YES slips to timeout)")
    print_table(rows)
    print("\nm* = c/%.2f is lam-independent under per-inspection costs; "
          "lam only scales the slip count." % BAD_FRAC)
    print("wrote", path)


if __name__ == "__main__":
    main()
