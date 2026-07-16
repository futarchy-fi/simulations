#!/usr/bin/env python3
"""
S5 (claim C8): subsidy-sizing sweep for proposal-poker.

sim.py is untouched. This script imports its config/agents/posterior code and
re-implements the per-proposal loop with exactly three additions:

1. `subsidy`: a per-proposal reward pool, paid at settlement pro-rata (by
   stake) to correct-side stakers. Unpaid if no one staked the correct side.
   Externally funded (a sponsor), so it is NOT budget-balanced -- that is the
   point of C8: how much outside money must be injected.
2. Entry rule: on an agent's FIRST entry decision for a proposal, the EV adds
   p_win * subsidy (sole-deviator belief: exact when testing a deviation from
   the all-abstain equilibrium; optimistic once others also enter). Realized
   payouts always use the true pro-rata split.
3. Deviation from the engine: the per-round participation cost c_j is charged
   at settlement (wealth and PnL). sim.py uses c_j only inside the EV rule and
   never deducts it, so realized PnL there would measure costless
   participation and understate the subsidy needed for +EV.

Everything else (stake escalation, 0.5x winner payout, fee, futarchy fallback
on contested high-y stalls, default-reject) mirrors sim.run_poker_for_proposal.

Run:  PYTHONPATH=/home/kelvin/.local/lib/python3.9/site-packages python3 s5_subsidy.py
Writes results-s5-subsidy.json next to this file.
"""

import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import SimConfig, Proposal, generate_agents, agent_posterior  # noqa: E402


def prob_positive(mean: float, prec: float) -> float:
    # ponytail: erf-based normal CDF, identical to scipy norm.cdf but without
    # the per-call scipy import in sim.prob_positive (hot loop).
    return 0.5 * (1.0 + math.erf(mean * math.sqrt(prec) / math.sqrt(2.0)))


def decide(p_pos, stake, fee_rate, cost, subsidy_ev):
    """sim.agent_decides_to_bet plus the anticipated subsidy-pool term."""
    ev_yes = stake * (2 * p_pos - 1 - fee_rate) - cost + p_pos * subsidy_ev
    ev_no = stake * (1 - 2 * p_pos - fee_rate) - cost + (1 - p_pos) * subsidy_ev
    if ev_yes > 0 and ev_yes >= ev_no:
        return "YES"
    if ev_no > 0 and ev_no >= ev_yes:
        return "NO"
    return None


def run_proposal(proposal, agents, cfg, rng, subsidy):
    """Copy of sim.run_poker_for_proposal with the S5 additions marked."""
    x = proposal.x

    signals = {}
    for agent in agents:
        signals[agent.id] = x + rng.normal(0, 1.0 / np.sqrt(agent.tau))

    yes_bets, no_bets = [], []
    participating = set()
    stake = cfg.initial_stake

    for round_num in range(1, cfg.n_rounds + 1):
        round_yes, round_no = [], []
        for agent in agents:
            if agent.wealth < stake + cfg.fee_rate * stake + agent.cost:
                continue
            post_mean, post_prec = agent_posterior(signals[agent.id], agent.tau)
            p_pos = prob_positive(post_mean, post_prec)
            sub_ev = subsidy if agent.id not in participating else 0.0  # S5 (2)
            decision = decide(p_pos, stake, cfg.fee_rate, agent.cost, sub_ev)
            if decision == "YES":
                round_yes.append((agent.id, stake))
                participating.add(agent.id)
            elif decision == "NO":
                round_no.append((agent.id, stake))
                participating.add(agent.id)

        yes_bets.extend(round_yes)
        no_bets.extend(round_no)
        proposal.rounds_played = round_num

        if len(round_yes) == 0 and len(round_no) == 0:
            break
        if len(round_yes) > 0 and len(round_no) == 0:
            proposal.approved = True
            proposal.method = "poker-unopposed"
            break
        if len(round_no) > 0 and len(round_yes) == 0:
            proposal.approved = False
            proposal.method = "poker-unopposed"
            break
        stake *= cfg.stake_multiplier

    proposal.total_stakes_yes = sum(s for _, s in yes_bets)
    proposal.total_stakes_no = sum(s for _, s in no_bets)
    proposal.n_participants = len(participating)

    if proposal.method == "":
        total_yes, total_no = proposal.total_stakes_yes, proposal.total_stakes_no
        if total_yes + total_no == 0:
            proposal.approved = False
            proposal.method = "default-reject"
        elif cfg.stall_triggers_futarchy and proposal.y > cfg.futarchy_cost:
            proposal.approved = (x + rng.normal(0, cfg.futarchy_noise)) > 0
            proposal.method = "futarchy"
        else:
            proposal.approved = total_yes > total_no
            proposal.method = "poker-weight"

    # Settlement (engine logic + S5 cost charge + S5 subsidy pool)
    correct_side = "YES" if x > 0 else "NO"
    agent_map = {a.id: a for a in agents}

    for side, bets in (("YES", yes_bets), ("NO", no_bets)):
        for aid, s in bets:
            agent = agent_map[aid]
            fee = cfg.fee_rate * s
            agent.wealth -= fee + agent.cost  # S5 (3): charge c_j for real
            agent.n_bets += 1
            if side == correct_side:
                agent.wealth += s * 0.5
                agent.total_pnl += s * 0.5 - fee - agent.cost
                agent.n_correct += 1
            else:
                agent.wealth -= s
                agent.total_pnl -= s + fee + agent.cost

    # S5 (1): per-proposal reward pool, pro-rata by correct-side stake
    subsidy_paid = 0.0
    correct_bets = yes_bets if correct_side == "YES" else no_bets
    total_correct = sum(s for _, s in correct_bets)
    if subsidy > 0 and total_correct > 0:
        for aid, s in correct_bets:
            share = subsidy * s / total_correct
            agent_map[aid].wealth += share
            agent_map[aid].total_pnl += share
        subsidy_paid = subsidy

    return participating, subsidy_paid


def run_rep(cfg, subsidy, seed):
    rng = np.random.default_rng(seed)
    xs = rng.normal(0, 1, cfg.n_proposals)
    ys = rng.lognormal(cfg.mu_y, cfg.sigma_y, cfg.n_proposals)
    agents = generate_agents(cfg, rng)
    med_tau = float(np.median([a.tau for a in agents]))
    informed = {a.id for a in agents if a.tau >= med_tau}

    part_slots = 0
    informed_slots = 0
    subsidy_total = 0.0
    mech_val = 0.0
    oracle_val = 0.0
    n_correct = 0
    n_futarchy = 0

    for i in range(cfg.n_proposals):
        p = Proposal(id=i, x=xs[i], y=ys[i])
        participating, paid = run_proposal(p, agents, cfg, rng, subsidy)
        part_slots += len(participating)
        informed_slots += len(participating & informed)
        subsidy_total += paid
        if p.approved:
            mech_val += p.x * p.y
        if p.x > 0:
            oracle_val += p.x * p.y
        n_correct += int(p.approved == (p.x > 0))
        n_futarchy += int(p.method == "futarchy")

    inf_bets = sum(a.n_bets for a in agents if a.id in informed)
    inf_pnl = sum(a.total_pnl for a in agents if a.id in informed)
    all_bets = sum(a.n_bets for a in agents)
    all_pnl = sum(a.total_pnl for a in agents)

    per_agent = {a.id: (a.total_pnl / a.n_bets if a.n_bets > 0 else None)
                 for a in agents}

    return {
        "per_agent_pnl_per_bet": per_agent,
        "informed_ids": sorted(informed),
        "taus": {a.id: a.tau for a in agents},
        "participation_rate": part_slots / (cfg.n_proposals * cfg.n_agents),
        "informed_participation_rate": informed_slots / (cfg.n_proposals * len(informed)),
        "efficiency": mech_val / oracle_val if oracle_val > 0 else None,
        "accuracy": n_correct / cfg.n_proposals,
        "informed_pnl_per_bet": inf_pnl / inf_bets if inf_bets > 0 else None,
        "all_pnl_per_bet": all_pnl / all_bets if all_bets > 0 else None,
        "subsidy_paid_per_proposal": subsidy_total / cfg.n_proposals,
        "futarchy_rate": n_futarchy / cfg.n_proposals,
    }


def mean_ci(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    if len(vals) < 2:
        return {"mean": m, "ci95": [m, m], "n": len(vals)}
    se = float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
    return {"mean": m, "ci95": [m - 1.96 * se, m + 1.96 * se], "n": len(vals)}


SUBSIDIES = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 1.0, 3.0, 10.0, 30.0, 50.0, 70.0, 100.0]
SETTINGS = {
    "cheap": {"cost_per_tau": 0.02, "fee_rate": 0.005},
    "default": {"cost_per_tau": 0.1, "fee_rate": 0.01},
    "pricey": {"cost_per_tau": 1.0, "fee_rate": 0.05},
    "prohibitive": {"cost_per_tau": 2.0, "fee_rate": 0.05},
}
REPS = 30
BASE_SEED = 42


def smoke():
    cfg = SimConfig()
    r0 = run_rep(cfg, 0.0, 1)
    assert r0["subsidy_paid_per_proposal"] == 0.0  # subsidy=0 pays nothing
    r_hi = run_rep(cfg, 100.0, 1)
    assert r_hi["participation_rate"] > r0["participation_rate"]  # pool attracts entry
    cfg_pro = SimConfig(cost_per_tau=2.0, fee_rate=0.05)
    r_pro = run_rep(cfg_pro, 0.0, 1)
    assert r_pro["participation_rate"] < 0.005  # no-information equilibrium
    print("smoke ok:", {k: round(v, 4) for k, v in r0.items()
                        if isinstance(v, (int, float))})


def main():
    smoke()
    cells = []
    for name, overrides in SETTINGS.items():
        for sub in SUBSIDIES:
            cfg = SimConfig(**overrides)
            reps = [run_rep(cfg, sub, BASE_SEED + r) for r in range(REPS)]
            cell = {"setting": name, **overrides, "subsidy": sub,
                    "reps": REPS, "n_proposals": cfg.n_proposals}
            for k in reps[0]:
                if k in ("per_agent_pnl_per_bet", "informed_ids", "taus"):
                    continue
                cell[k] = mean_ci([r[k] for r in reps])
            cell["no_info_persists"] = cell["participation_rate"]["mean"] < 0.005

            # Best informed agent: "any informed agent participates with +EV"
            # requires only one such agent. Per-agent PnL/bet CI across reps;
            # need >= 10 reps with bets so a lucky pair of reps can't qualify.
            best = None
            for aid in reps[0]["informed_ids"]:
                m = mean_ci([r["per_agent_pnl_per_bet"][aid] for r in reps])
                if m is None or m["n"] < 10:
                    continue
                if best is None or m["ci95"][0] > best["pnl_per_bet"]["ci95"][0]:
                    best = {"id": aid, "tau": reps[0]["taus"][aid], "pnl_per_bet": m}
            cell["best_informed_agent"] = best
            cells.append(cell)
            eff = cell["efficiency"]
            print(f"{name:12s} sub={sub:<7g} part={cell['participation_rate']['mean']:.3f} "
                  f"eff={eff['mean'] if eff else float('nan'):.3f} "
                  f"inf_pnl={cell['informed_pnl_per_bet']['mean'] if cell['informed_pnl_per_bet'] else float('nan'):+.3f}",
                  flush=True)

    # Knee per setting:
    #   (a) min subsidy where at least one informed agent bets (>=10 reps)
    #       with realized PnL/bet CI-low > 0
    #   (b) min subsidy with efficiency CI-low > 0.05 (always-default baseline = 0)
    knees = {}
    for name in SETTINGS:
        sc = [c for c in cells if c["setting"] == name]
        knee_a = next((c["subsidy"] for c in sc
                       if c["best_informed_agent"]
                       and c["best_informed_agent"]["pnl_per_bet"]["ci95"][0] > 0), None)
        knee_b = next((c["subsidy"] for c in sc
                       if c["efficiency"] and c["efficiency"]["ci95"][0] > 0.05), None)
        knees[name] = {"knee_a_informed_plus_ev": knee_a, "knee_b_beats_default": knee_b}

    out = {
        "experiment": "S5 subsidy sizing (claim C8)",
        "engine": "sim.py @ proposal-poker",
        "extension": ("per-proposal reward pool paid pro-rata to correct-side stakers; "
                      "sole-deviator pool anticipation on first entry; "
                      "participation cost c_j charged at settlement (engine omits it)"),
        "baseline": "always-default = reject-all = value 0 = efficiency 0",
        "subsidy_grid": SUBSIDIES,
        "settings": SETTINGS,
        "reps_per_cell": REPS,
        "knees": knees,
        "cells": cells,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-s5-subsidy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("knees:", json.dumps(knees))
    print("wrote", path)


if __name__ == "__main__":
    main()
