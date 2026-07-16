"""S3 (C4): spam floods vs a capital-constrained defender pool.

Spammer activates proposals (YES bond m) at Poisson rate lam per TIMEOUT,
fraction beta bad; bad proposals are abandoned once contested (escalating a
bad proposal is -EV at alpha=1), good spam executes harmlessly. Defenders
identify bad proposals perfectly (capital, not information, is the binding
constraint here; information is S4's job) and contest while free capital
lasts; a win returns the stake doubled after TIMEOUT. Bad executions happen
when a bad YES reaches timeout while the pool is fully locked. The
congestion-guard engine variant (a design-doc idea, not part of 8cee5bc) is
NOT swept here: nothing ever graduates in this scenario, so the queue never
fills and guard on/off rows were verified identical -- zero signal.

Usage: python3 s3_spam_capital.py [--pilot]
"""

from __future__ import annotations

import argparse

import numpy as np

from game import Game
from report import print_table, write_json

SEED = 20260718


def run_config(lam, beta, d0_x, horizon_t, seed):
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1)
    d0 = d0_x * g.base_x
    horizon = horizon_t * g.timeout
    rate = lam / float(g.timeout)
    bad_open, contested, good_open = [], [], []
    bad_arrivals = bad_exec = good_exec = 0
    min_free = d0
    for t in range(int(horizon) + 1):
        # settle due timers
        for pool, lst in (("good", good_open), ("bad", bad_open),
                          ("no", contested)):
            for pid in list(lst):
                p = g.proposals[pid]
                if t >= p.last_change + g.timeout:
                    acc = g.finalize_by_timeout(t, pid)
                    lst.remove(pid)
                    if pool == "bad" and acc:
                        bad_exec += 1
                    if pool == "good" and acc:
                        good_exec += 1
        # arrivals
        for _ in range(rng.poisson(rate)):
            bad = bool(rng.random() < beta)
            pid = g.new_proposal(quality=not bad, t=t)
            g.place_yes(t, pid, "spam", g.m)
            if bad:
                bad_arrivals += 1
                bad_open.append(pid)
            else:
                good_open.append(pid)
        # defenders contest every bad YES that free capital covers
        free = d0 + g.balances.get("def", 0.0)
        for pid in list(bad_open):
            amt = g.proposals[pid].yes.amount
            if free >= amt - 1e-12:
                g.place_no(t, pid, "def")
                bad_open.remove(pid)
                contested.append(pid)
                free -= amt
        min_free = min(min_free, free)
    cycles = horizon_t  # one defender turn per TIMEOUT
    final = d0 + g.balances.get("def", 0.0)
    spend = -g.balances.get("spam", 0.0)
    return {"lam": lam, "beta": beta, "D0": d0,
            "bad_arr": bad_arrivals, "bad_exec": bad_exec,
            "good_exec": good_exec, "def_final": final,
            "min_free": min_free,
            "growth_per_cycle": (final / d0) ** (1.0 / cycles),
            "attacker_spend": spend,
            "spend_per_bad_exec": spend / bad_exec if bad_exec else float("inf")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    # spec grid maxes out at lam*beta*m = 16m locked per TIMEOUT vs D0 >= 40m,
    # so defenders are never constrained there; the appended stress rate 160
    # exhibits the overrun regime (and is labeled as beyond the spec grid)
    if args.pilot:
        lams, betas, d0s, horizon = [1, 16, 160], [1.0, 0.5], [5, 20], 20
    else:
        lams, betas, d0s, horizon = [1, 4, 16, 160], [1.0, 0.8, 0.5], \
            [5, 10, 20], 100
    rows = []
    for i, lam in enumerate(lams):
        for j, beta in enumerate(betas):
            for k, d0x in enumerate(d0s):
                rows.append(run_config(lam, beta, d0x, horizon,
                                       SEED + 1000 * i + 100 * j + 10 * k))
    path = write_json("s3_spam_capital",
                      {"pilot": args.pilot, "seed": SEED,
                       "horizon_timeouts": horizon}, rows)
    print("## S3 spam vs defender capital (horizon %d TIMEOUTs; congestion-"
          "guard variant not swept: nothing graduates, zero signal)" % horizon)
    print_table(rows)
    print("\nHeadline: growth_per_cycle == 2 only while the pool is fully "
          "deployed AND never overrun; mixed spam (beta<1) costs nothing "
          "given perfect ID, it only thins bad arrivals.")
    print("wrote", path)


if __name__ == "__main__":
    main()
