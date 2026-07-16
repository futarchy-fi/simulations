"""S1 (C1+C2): stalling, disagreement escalation, and purchasable delay.

A) common-belief sweep: accept-band map (accepted iff p >= 1/2: the crowd
   moves only on strict EV > 0, so at the p == 1/2 knife edge it is
   indifferent and the YES stands).
B) two-belief disagreement grid: escalation depth, graduation, win shares.
C) DelayAdversary: purchasable delay vs budget; the printed cost/window
   reference Y*(2p-1) fixes Y = m = 1 so it is valid for the B=1 rows only
   (larger budgets double the standing bond each window, so cost_per_w
   legitimately exceeds it); log2(B) window scaling until the graduation
   threshold caps it. Defender is informed (re-flips iff truly good), not
   the belief-threshold crowd -- see run_delay_adversary.

Usage: python3 s1_stall_delay.py [--pilot]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from agents import run_common_belief, run_delay_adversary, run_two_belief
from game import Game
from report import print_table, write_json

SEED = 20260716


def part_a(pilot):
    ps = [0.1, 0.3, 0.5, 0.7, 0.9] if pilot else \
        [round(0.05 * k, 2) for k in range(1, 20)]
    M = 200 if pilot else 2000
    rows = []
    for j, p in enumerate(ps):
        rng = np.random.default_rng(SEED + j)
        acc = grad = 0
        tts = wel = 0.0
        for i in range(M):
            quality = bool(rng.random() < p)  # calibrated q = p
            g = Game(seed=SEED + 1000 * j + i)
            pid = g.new_proposal(quality)
            rec = run_common_belief(g, pid, p)
            acc += rec["accepted"]
            grad += rec["graduated"]
            tts += rec["settle_time"]
            w = (1.0 if quality else -1.0) if rec["accepted"] else 0.0
            wel += w - 0.001 * rec["settle_time"]
        rows.append({"p": p, "P_accept": acc / M, "P_reject": 1 - acc / M,
                     "P_grad": grad / M, "mean_tts_h": tts / M,
                     "welfare": wel / M,
                     "expect_ok": (acc / M == float(p >= 0.5)) and grad == 0})
    return rows


def part_b(pilot):
    grid_s = [0.1, 0.3] if pilot else [0.1, 0.2, 0.3, 0.4]
    grid_o = [0.7, 0.9] if pilot else [0.7, 0.8, 0.9]
    ratios = [(1, 4), (1, 1), (4, 1)]
    total = 32.0  # * m; lean ratios bite below the 8m threshold, rich reach it
    M = 100 if pilot else 1000
    rows = []
    for p_s in grid_s:
        for p_o in grid_o:
            for ro, rs in ratios:
                b_o = total * ro / (ro + rs)
                b_s = total * rs / (ro + rs)
                cnt = {}
                depth = transfer = 0.0
                for i in range(M):
                    g = Game(seed=SEED + i)
                    pid = g.new_proposal(bool((i % 2) == 0))  # q = 0.5
                    rec = run_two_belief(g, pid, p_o, p_s, b_o, b_s)
                    cnt[rec["outcome"]] = cnt.get(rec["outcome"], 0) + 1
                    depth += rec["depth"]
                    transfer += max(rec["net_opt"], rec["net_skp"])
                rows.append({"p_s": p_s, "p_o": p_o,
                             "ratio": "%d:%d" % (ro, rs),
                             "depth": depth / M,
                             "P_grad": (cnt.get("oracle_accept", 0) +
                                        cnt.get("oracle_reject", 0)) / M,
                             "opt_win": (cnt.get("opt_timeout", 0) +
                                         cnt.get("oracle_accept", 0)) / M,
                             "skp_win": (cnt.get("skp_timeout", 0) +
                                         cnt.get("oracle_reject", 0)) / M,
                             "transfer": transfer / M})
    return rows


def part_c(pilot):
    ps = [0.5, 0.6, 0.9] if pilot else [0.5, 0.55, 0.6, 0.7, 0.9]
    budgets = [1, 8, 64] if pilot else [2 ** k for k in range(9)]
    M = 200 if pilot else 2000
    rows = []
    for p in ps:
        for b in budgets:
            rng = np.random.default_rng(SEED + int(1000 * p) + b)
            d = cost = flips = flipped = 0.0
            for i in range(M):
                g = Game(alpha=1.0, seed=SEED + 7919 * b + i)
                pid = g.new_proposal(bool(rng.random() < p))  # q = p
                rec = run_delay_adversary(g, pid, float(b) * g.m)
                d += rec["delay_windows"]
                cost += rec["realized_cost"]
                flips += rec["flips"]
                flipped += rec["outcome_flipped"]
            dw = d / M
            rows.append({"p": p, "B": b, "delay_w": dw,
                         "cost": cost / M,
                         "cost_per_w": (cost / M) / dw if dw else 0.0,
                         "refB1_Y(2p-1)": 1.0 * (2 * p - 1),  # Y=m=1: B=1 rows only
                         "log2B+1": math.log2(b) + 1,
                         "flipped": flipped / M})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    a, b, c = part_a(args.pilot), part_b(args.pilot), part_c(args.pilot)
    path = write_json("s1_stall_delay",
                      {"pilot": args.pilot, "seed": SEED}, {"A": a, "B": b, "C": c})
    print("## S1A common-belief accept band (expect: accepted iff p>=1/2, "
          "YES stands at the p=1/2 tie; no grads)")
    print_table(a)
    print("\n## S1B disagreement escalation")
    print_table(b)
    print("\n## S1C purchasable delay (informed defender re-flips iff truly "
          "good; refB1 column valid at B=1 only, bond doubles per window for "
          "B>1; grad threshold caps windows)")
    print_table(c)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
