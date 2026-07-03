"""Summarise phase-2 results (entry sweep, TWAP re-solves, T2u type
uncertainty) into the tables used by MANIPULATION.md and the
results-site explainer. Tolerates missing files (prints what exists).

Usage: python phase2_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"

BONUSES = ["0.0", "0.02", "0.05", "0.2"]


def load(name):
    p = RES / name
    if not p.exists():
        return None
    return json.load(open(p))


def acc(d, cname, bonus):
    if d is None:
        return None
    rows = d["results"].get(cname)
    if rows is None:
        return None
    r = rows.get(bonus) or rows.get(str(float(bonus)))
    if r is None:
        return None
    return r["aggregate"]["decision_accuracy"]


def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def main():
    final = {t: load(f"entry_sweep_{t}.json") for t in ("T1", "T2", "T3")}
    twap_t1 = load("entry_sweep_BASE-2_T1_twap.json")
    twap_t2 = load("entry_sweep_T2_twap.json")
    twap_t3 = load("entry_sweep_REPL_T3_twap.json")
    repl_final = load("manipulator_sweep_t3s111y2.json")

    print("== TWAP vs final: decision accuracy ==")
    header = ("| bounty | T2-last final | T2-last TWAP | T1-last final | "
              "T1-last TWAP | T2-first final | T2-first TWAP | "
              "REPL final | REPL TWAP |")
    print(header)
    print("|" + "---|" * 9)
    for b in BONUSES:
        repl_f = None
        if repl_final is not None:
            key = b if b in repl_final["results"] else str(float(b))
            r = repl_final["results"].get(key)
            repl_f = r["aggregate"]["decision_accuracy"] if r else None
        cells = [
            acc(final["T2"], "T2-last", b), acc(twap_t2, "T2-last", b),
            acc(final["T1"], "T1-last", b), acc(twap_t1, "T1-last", b),
            acc(final["T2"], "T2-first", b), acc(twap_t2, "T2-first", b),
            repl_f, acc(twap_t3, "REPL", b) if b != "0.02" else None,
        ]
        print(f"| {b} | " + " | ".join(fmt(c) for c in cells) + " |")

    # T3 TWAP rows
    if twap_t3 is not None:
        print("\nT3 TWAP:", {f"{c}@{b}": fmt(acc(twap_t3, c, b))
                             for c in ("T3-first", "T3-last") for b in BONUSES})
    if twap_t1 is not None:
        print("T1-first TWAP:", {b: fmt(acc(twap_t1, "T1-first", b)) for b in BONUSES})
        print("BASE-2 TWAP:", fmt(acc(twap_t1, "BASE-2", "0")))

    # T2u
    print("\n== T2u type uncertainty (bonus 0.2) ==")
    for tag in ("first", "last"):
        d = load(f"t2u_{tag}_q0.25_0.5.json")
        if d is None:
            print(f"(t2u {tag} pending)")
            continue
        for cname, r in d["results"].items():
            bt = r["by_type"]
            h, b_ = bt["honest"]["__aggregate__"], bt["bribed"]["__aggregate__"]
            q = r["params"]["manipulator_prob"]
            mix = r["acc_mixture"]
            known_mix = (1 - q) * 1.0 + q * (0.75 if tag == "first" else 0.5)
            print(f"{cname}: acc_honest={h['decision_accuracy']:.4f} "
                  f"acc_bribed={b_['decision_accuracy']:.4f} "
                  f"mix={mix:.4f} known-type-mix={known_mix:.4f} "
                  f"nc={r['nash_conv']:.2e}")
            hp = {s: round(v["mean_stat"], 3) for s, v in bt["honest"].items()
                  if s != "__aggregate__"}
            bp = {s: round(v["mean_stat"], 3) for s, v in bt["bribed"].items()
                  if s != "__aggregate__"}
            print(f"   honest-type mean stat by state: {hp}")
            print(f"   bribed-type mean stat by state: {bp}")


if __name__ == "__main__":
    main()
