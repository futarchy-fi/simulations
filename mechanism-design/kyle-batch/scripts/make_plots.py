#!/usr/bin/env python
"""Generate the committed PNG figures for KYLE.md from results/*.json.

Palette: validated default light-mode categorical slots (dataviz skill
reference instance): blue #2a78d6, aqua #1baf7a, yellow #eda100,
violet #4a3aa7; surface #fcfcfb; text #0b0b0b / #52514e.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _logx(ax, ticks=(0.2, 0.5, 1, 2, 5)):
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(which="minor", labelbottom=False)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]  # fixed slot order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.edgecolor": GRID, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2.0, "lines.markersize": 5.5,
})


def fig_corruption():
    data = json.loads((RES / "corruption.json").read_text())["B_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    rhos = [1.0, 0.5, 0.0]
    labels = {1.0: "known (ρ=1)", 0.5: "covert ρ=0.5", 0.0: "fully covert ρ=0"}
    for ax, key, title, ylab in (
        (axes[0], "bias_present", "Price bias E[p] (manipulator present)", "bias"),
        (axes[1], "d_dq_present", "Decision-quality change vs B=0", "Δ E[v·q(p)]"),
    ):
        for i, rho in enumerate(rhos):
            rows = [r for r in data if abs(r["rho"] - max(rho, 1e-12)) < 1e-6]
            rows.sort(key=lambda r: r["B"])
            xs = [r["B"] for r in rows]
            ys = [r.get(key) or 0.0 for r in rows]
            ax.plot(xs, ys, "-o", color=SERIES[i], label=labels[rho])
            ax.annotate(labels[rho], (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(4, 0), fontsize=8.5, color=TEXT2)
        ax.set_xlabel("bounty B")
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Corruption is smooth, bias-shaped, and vanishes when the MM knows "
                 "(N=3, σ_ε=1, σ_u=1, τ=0.3)", x=0.01, ha="left", fontsize=11)
    fig.savefig(RES / "fig_corruption_smooth.png", dpi=160)
    print("wrote fig_corruption_smooth.png")


def fig_frontier():
    data = json.loads((RES / "frontier.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    Bs = [0.5, 1.0, 2.0, 5.0]

    # panel 1: price bias vs sigma_u (linear MM, rho=0)
    ax = axes[0]
    for i, B in enumerate(Bs):
        rows = sorted([r for r in data if r["mm"] == "linear" and r["rho"] == 0.0
                       and r["B"] == B], key=lambda r: r["sigma_u"])
        ax.plot([r["sigma_u"] for r in rows], [r["bias"] for r in rows],
                "-o", color=SERIES[i], label=f"B={B}")
        ax.annotate(f"B={B}", (rows[0]["sigma_u"], rows[0]["bias"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8.5,
                    color=TEXT2)
    _logx(ax)
    ax.set_xlabel("noise depth σ_u (log)")
    ax.set_ylabel("price bias E[p]")
    ax.set_title("Bias ∝ λ(σ_u): noise depth buys resistance", loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    # panel 2: subsidy cost vs sigma_u (single series; identical across B)
    ax = axes[1]
    rows = sorted([r for r in data if r["mm"] == "linear" and r["rho"] == 0.0
                   and r["B"] == 1.0], key=lambda r: r["sigma_u"])
    ax.plot([r["sigma_u"] for r in rows], [r["subsidy"] for r in rows],
            "-o", color=SERIES[0])
    _logx(ax)
    ax.set_xlabel("noise depth σ_u (log)")
    ax.set_ylabel("noise-flow expected loss λσ_u²")
    ax.set_title("…but the noise flow pays for it linearly", loc="left")

    # panel 3: the frontier -- damage vs subsidy, traced by sigma_u
    ax = axes[2]
    for i, B in enumerate(Bs):
        rows = sorted([r for r in data if r["mm"] == "linear" and r["rho"] == 0.0
                       and r["B"] == B], key=lambda r: r["sigma_u"])
        xs = [r["subsidy"] for r in rows]
        ys = [-r["d_dq"] for r in rows]
        ax.plot(xs, ys, "-o", color=SERIES[i], label=f"B={B}")
        ax.annotate(f"B={B}", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(6, 2), fontsize=8.5, color=TEXT2)
    ax.set_xlabel("liquidity subsidy (noise-flow loss λσ_u²)")
    ax.set_ylabel("decision-quality damage −ΔDQ")
    ax.set_title("The noise-budget / corruption frontier", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("The noise budget buys corruption resistance — and baseline "
                 "corr(p,v) is σ_u-invariant  (N=3, σ_ε=1, τ=0.3, fully covert)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(RES / "fig_frontier.png", dpi=160)
    print("wrote fig_frontier.png")


def fig_camouflage():
    data = json.loads((RES / "frontier.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    B = 2.0
    for i, (mm, lab) in enumerate([("linear", "linear MM (no detection)"),
                                   ("bayes", "Bayesian MM (detects pushes)")]):
        rows = sorted([r for r in data if r["mm"] == mm and r["rho"] == 0.5
                       and r["B"] == B], key=lambda r: r["sigma_u"])
        ax.plot([r["sigma_u"] for r in rows], [-r["d_dq"] for r in rows],
                "-o", color=SERIES[i], label=lab)
    _logx(ax)
    ax.set_xlabel("noise depth σ_u (log)")
    ax.set_ylabel("decision-quality damage −ΔDQ")
    ax.set_title(f"Detection does not rescue thin markets (ρ=0.5, B={B})",
                 loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.savefig(RES / "fig_camouflage.png", dpi=160)
    print("wrote fig_camouflage.png")


def fig_twap():
    data = json.loads((RES / "twap.json").read_text())
    rows = [r for r in data if r["statistic"] in ("twap", "last")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    ax = axes[0]
    for i, stat in enumerate(("last", "twap")):
        rr = sorted([r for r in rows if r["statistic"] == stat and r["B"] == 0.0],
                    key=lambda r: r["T"])
        ax.plot([r["T"] for r in rr], [r["dq"] for r in rr], "-o",
                color=SERIES[i], label=stat)
    ax.set_xlabel("batches T")
    ax.set_ylabel("baseline DQ  E[v·q(P)]")
    ax.set_title("Baseline: last batch beats TWAP", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    for i, stat in enumerate(("last", "twap")):
        rr = sorted([r for r in rows if r["statistic"] == stat and r["B"] == 2.0],
                    key=lambda r: r["T"])
        ax.plot([r["T"] for r in rr], [-r["d_dq"] for r in rr], "-o",
                color=SERIES[i], label=stat)
    ax.set_xlabel("batches T")
    ax.set_ylabel("damage −ΔDQ at B=2")
    ax.set_title("Corruption damage by decision statistic", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("TWAP-of-batches vs last batch (covert uninformed pusher, myopic MM)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(RES / "fig_twap.png", dpi=160)
    print("wrote fig_twap.png")


def fig_twap_windowed():
    data = json.loads((RES / "twap_windowed.json").read_text())["known_K"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    Ts = [4, 8, 16]

    ax = axes[0]
    for i, T in enumerate(Ts):
        rows = sorted([r for r in data if r["T"] == T and r["B"] == 0.0],
                      key=lambda r: r["K"])
        ax.plot([r["K"] for r in rows], [r["baseline_cost_vs_K1"] for r in rows],
                "-o", color=SERIES[i], label=f"T={T}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("window K (mean of last K batch prices)")
    ax.set_ylabel("DQ(K) − DQ(K=1) at B=0")
    ax.set_title("Baseline cost of remembering more prices", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax = axes[1]
    for i, T in enumerate(Ts):
        rows = sorted([r for r in data if r["T"] == T and r["B"] == 5.0],
                      key=lambda r: r["K"])
        ax.plot([r["K"] for r in rows], [r["damage"] for r in rows],
                "-o", color=SERIES[i], label=f"T={T}")
        ax.annotate(f"T={T}", (rows[-1]["K"], rows[-1]["damage"]),
                    textcoords="offset points", xytext=(5, 0), fontsize=8.5,
                    color=TEXT2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("window K")
    ax.set_ylabel("damage −ΔDQ at B=5 (log)")
    ax.set_title("…buys almost no damage reduction", loc="left")
    fig.suptitle("Windowed TWAP: the averaging window K decomposed "
                 "(N=3, σ_ε=1, σ_u=1, τ=0.3, covert uninformed pusher)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(RES / "fig_twap_windowed.png", dpi=160)
    print("wrote fig_twap_windowed.png")


def fig_subsidy():
    d = json.loads((RES / "subsidy.json").read_text())
    rows = d["rows"]
    S_star = d["config"]["kappa_equals_kyle_lambda_at_S"]
    insts = [("noise", "noise flow (E[transfer]=S)"),
             ("amm_covert", "AMM depth (worst-case=S), covert"),
             ("amm_aware", "AMM depth, honest aware")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)

    ax = axes[0]
    for i, (inst, lab) in enumerate(insts):
        rr = sorted([r for r in rows if r["instrument"] == inst],
                    key=lambda r: r["S"])
        ax.plot([r["S"] for r in rr], [r["damage"] for r in rr], "-o",
                color=SERIES[i], label=lab)
    ax.axvline(S_star, color=GRID, linewidth=1.2)
    ax.annotate("κ = Kyle λ*", (S_star, ax.get_ylim()[0]), fontsize=8,
                color=TEXT2, textcoords="offset points", xytext=(3, 12))
    _logx(ax, ticks=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2))
    ax.set_yscale("log")
    ax.set_xlabel("per-market subsidy budget S (log)")
    ax.set_ylabel("damage −ΔDQ at B=2 (log)")
    ax.set_title("Corruption resistance per dollar", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for i, (inst, lab) in enumerate(insts):
        rr = sorted([r for r in rows if r["instrument"] == inst],
                    key=lambda r: r["S"])
        ax.plot([r["S"] for r in rr], [r["dq0"] for r in rr], "-o",
                color=SERIES[i], label=lab)
    rr = sorted([r for r in rows if r["instrument"] == "noise"],
                key=lambda r: r["S"])
    ax.plot([r["S"] for r in rr], [r["dq0_frozen_beta"] for r in rr], "--s",
            color=SERIES[3], label="noise, frozen-β (behavioral bound)",
            markersize=4.5)
    ax.axvline(S_star, color=GRID, linewidth=1.2)
    _logx(ax, ticks=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2))
    ax.set_xlabel("per-market subsidy budget S (log)")
    ax.set_ylabel("baseline DQ (B=0)")
    ax.set_title("What the same dollars do at baseline", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Two subsidy instruments, one budget: noise flow vs AMM depth "
                 "(N=3, σ_ε=1, τ=0.3, covert informed manipulator, B=2)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(RES / "fig_subsidy.png", dpi=160)
    print("wrote fig_subsidy.png")


if __name__ == "__main__":
    fig_corruption()
    fig_frontier()
    fig_camouflage()
    fig_twap()
    fig_twap_windowed()
    fig_subsidy()
