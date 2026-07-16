"""S2 (C6): is the bond stage endogenous screening or oracle passthrough?

Informed disagreement crowd: per proposal the crowd sees a signal of the
true quality (accuracy sigma), forms the common posterior, and optimists /
skeptics play the two-belief game at posterior +/- 0.25. Proposals in the
disagreement band graduate to the oracle (accuracy alpha); the rest settle
by timeout on the crowd's information. Compared against the naked oracle
deciding every proposal directly. delta < 0 means the bond stage ADDS
screening; delta ~ 0 means it INHERITS the oracle's corruption.

Usage: python3 s2_endogeneity.py [--pilot]
"""

from __future__ import annotations

import argparse

import numpy as np

from agents import run_two_belief
from game import Game
from report import print_table, write_json

SEED = 20260717
SIGMA = 0.8   # crowd signal accuracy
BAND = 0.25   # optimist/skeptic spread around the posterior


def posterior(q, signal_good):
    if signal_good:
        return q * SIGMA / (q * SIGMA + (1 - q) * (1 - SIGMA))
    return q * (1 - SIGMA) / (q * (1 - SIGMA) + (1 - q) * SIGMA)


def run_grid(alphas, qs, M):
    rows = []
    for a_i, alpha in enumerate(alphas):
        for q_i, q in enumerate(qs):
            rng = np.random.default_rng(SEED + 100 * a_i + q_i)
            wrong_sys = wrong_orc = grads = 0
            for i in range(M):
                quality = bool(rng.random() < q)
                signal = bool(rng.random() < SIGMA) == quality
                post = posterior(q, signal)
                p_o = min(post + BAND, 0.99)
                p_s = max(post - BAND, 0.01)
                g = Game(alpha=alpha, seed=SEED + 7919 * a_i + 131 * q_i + i)
                pid = g.new_proposal(quality)
                rec = run_two_belief(g, pid, p_o, p_s)
                wrong_sys += rec["accepted"] != quality
                grads += rec["graduated"]
                # naked oracle on the same proposal
                orc = quality if rng.random() < alpha else (not quality)
                wrong_orc += orc != quality
            rows.append({"alpha": alpha, "q": q, "P_grad": grads / M,
                         "wrong_system": wrong_sys / M,
                         "wrong_oracle": wrong_orc / M,
                         "delta": (wrong_sys - wrong_orc) / M})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        alphas, qs, M = [0.5, 0.75, 1.0], [0.2, 0.5, 0.8], 200
    else:
        alphas = [round(0.5 + 0.05 * k, 2) for k in range(11)]
        qs = [round(0.1 * k, 1) for k in range(1, 10)]
        M = 2000
    rows = run_grid(alphas, qs, M)
    path = write_json("s2_endogeneity",
                      {"pilot": args.pilot, "seed": SEED, "sigma": SIGMA,
                       "band": BAND}, rows)
    print("## S2 wrong-execution rate: two-stage system vs naked oracle "
          "(sigma=%.2f crowd signal)" % SIGMA)
    print_table(rows)
    # headline: mean delta per alpha
    agg = []
    for alpha in sorted(set(r["alpha"] for r in rows)):
        sub = [r["delta"] for r in rows if r["alpha"] == alpha]
        agg.append({"alpha": alpha, "mean_delta": sum(sub) / len(sub)})
    print("\n## delta vs alpha (negative = bond stage adds screening)")
    print_table(agg)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
