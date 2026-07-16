"""
Proposal Poker — Mechanism Simulation

Model:
- Proposals have quality x ~ N(0,1) and importance y ~ LogNormal(mu_y, sigma_y).
- Payoff from approving = x * y. Payoff from rejecting = 0. Goal: approve iff x > 0.
- N agents with heterogeneous signal precision tau_j and participation cost c_j.
- Agents bet YES or NO in escalating rounds (like poker blinds).
- 1% fee on amount at stake discourages noise bettors.
- If the market stalls (no one bets), the mechanism either decides or escalates to futarchy.

Poker Futarchy Rounds:
  Round 1: minimum stake S_1 (small). Any agent can bet YES or NO.
  Round 2: minimum stake S_2 = 2 * S_1. Agents can raise, call, or fold.
  ...
  Round K: stakes are large. Only agents with strong signals remain.

  If at any round, one side has no bettors, the other side wins immediately.
  If both sides have bettors but the stakes reach a threshold, trigger futarchy.

The 1% fee means an agent only bets if:
    |P(x>0 | signal) - 0.5| > fee_rate / 2   (roughly)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


# ─── Parameters ───────────────────────────────────────────────────────

@dataclass
class SimConfig:
    # Proposal distribution
    n_proposals: int = 500
    mu_y: float = 1.0        # log-mean of importance
    sigma_y: float = 1.5     # log-std of importance (heavy tail)

    # Agent population
    n_agents: int = 20
    tau_min: float = 0.5     # worst signal precision
    tau_max: float = 5.0     # best signal precision
    cost_per_tau: float = 0.1  # c_j = cost_per_tau * tau_j

    # Mechanism
    fee_rate: float = 0.01   # 1% of amount at stake
    n_rounds: int = 5        # max betting rounds
    stake_multiplier: float = 2.0  # stake doubles each round
    initial_stake: float = 1.0
    futarchy_cost: float = 50.0    # cost C to run a futarchy
    futarchy_noise: float = 0.3    # noise std of futarchy signal

    # Decision thresholds
    confidence_threshold: float = 0.7  # P(x>0) needed to approve without futarchy
    stall_triggers_futarchy: bool = True

    seed: int = 42


# ─── Agent ────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    tau: float          # signal precision
    cost: float         # fixed cost to participate per round
    wealth: float = 100.0
    total_pnl: float = 0.0
    n_bets: int = 0
    n_correct: int = 0


# ─── Proposal ─────────────────────────────────────────────────────────

@dataclass
class Proposal:
    id: int
    x: float            # true quality
    y: float            # true importance
    approved: bool = False
    method: str = ""    # "poker", "futarchy", "default"
    rounds_played: int = 0
    total_stakes_yes: float = 0.0
    total_stakes_no: float = 0.0
    fees_collected: float = 0.0
    n_participants: int = 0


# ─── Core simulation ─────────────────────────────────────────────────

def generate_agents(cfg: SimConfig, rng: np.random.Generator) -> List[Agent]:
    """Generate agents with linearly spaced precision, cost proportional to precision."""
    taus = np.linspace(cfg.tau_min, cfg.tau_max, cfg.n_agents)
    agents = []
    for i, tau in enumerate(taus):
        cost = cfg.cost_per_tau * tau
        agents.append(Agent(id=i, tau=tau, cost=cost))
    return agents


def agent_posterior(signal: float, tau: float, prior_mean: float = 0.0, prior_prec: float = 1.0) -> Tuple[float, float]:
    """
    Bayesian update: prior N(prior_mean, 1/prior_prec), signal ~ N(x, 1/tau).
    Returns (posterior_mean, posterior_precision).
    """
    post_prec = prior_prec + tau
    post_mean = (prior_prec * prior_mean + tau * signal) / post_prec
    return post_mean, post_prec


def prob_positive(mean: float, prec: float) -> float:
    """P(x > 0) under N(mean, 1/prec)."""
    from scipy.stats import norm
    std = 1.0 / np.sqrt(prec)
    return float(norm.cdf(mean / std))


def agent_decides_to_bet(
    p_pos: float,
    stake: float,
    fee_rate: float,
    cost: float,
) -> Optional[str]:
    """
    Agent decides whether to bet YES, NO, or abstain.

    Expected profit from betting YES at stake S:
        p_pos * S - (1 - p_pos) * S - fee_rate * S - cost
        = S * (2*p_pos - 1 - fee_rate) - cost

    Similarly for NO:
        S * (2*(1-p_pos) - 1 - fee_rate) - cost
        = S * (1 - 2*p_pos - fee_rate) - cost
    """
    ev_yes = stake * (2 * p_pos - 1 - fee_rate) - cost
    ev_no = stake * (1 - 2 * p_pos - fee_rate) - cost

    if ev_yes > 0 and ev_yes >= ev_no:
        return "YES"
    elif ev_no > 0 and ev_no >= ev_yes:
        return "NO"
    else:
        return None


def run_poker_for_proposal(
    proposal: Proposal,
    agents: List[Agent],
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Proposal:
    """Run the poker futarchy mechanism on a single proposal."""
    x = proposal.x

    # Generate all agent signals upfront
    signals = {}
    for agent in agents:
        noise_std = 1.0 / np.sqrt(agent.tau)
        signals[agent.id] = x + rng.normal(0, noise_std)

    # Track bets per round
    yes_bets: List[Tuple[int, float]] = []  # (agent_id, stake)
    no_bets: List[Tuple[int, float]] = []
    participating_agents = set()

    stake = cfg.initial_stake

    for round_num in range(1, cfg.n_rounds + 1):
        round_yes = []
        round_no = []

        for agent in agents:
            if agent.wealth < stake + cfg.fee_rate * stake + agent.cost:
                continue  # can't afford

            # Agent computes posterior from their signal
            post_mean, post_prec = agent_posterior(signals[agent.id], agent.tau)
            p_pos = prob_positive(post_mean, post_prec)

            decision = agent_decides_to_bet(p_pos, stake, cfg.fee_rate, agent.cost)

            if decision == "YES":
                round_yes.append((agent.id, stake))
                participating_agents.add(agent.id)
            elif decision == "NO":
                round_no.append((agent.id, stake))
                participating_agents.add(agent.id)

        yes_bets.extend(round_yes)
        no_bets.extend(round_no)
        proposal.rounds_played = round_num

        # Collect fees
        round_total_stake = sum(s for _, s in round_yes) + sum(s for _, s in round_no)
        proposal.fees_collected += cfg.fee_rate * round_total_stake

        # Check if one side is empty this round — the other side wins
        if len(round_yes) == 0 and len(round_no) == 0:
            # No one bets — stall. Use current balance to decide.
            break

        if len(round_yes) > 0 and len(round_no) == 0:
            # YES side wins by default
            proposal.approved = True
            proposal.method = "poker-unopposed"
            break

        if len(round_no) > 0 and len(round_yes) == 0:
            # NO side wins by default
            proposal.approved = False
            proposal.method = "poker-unopposed"
            break

        # Both sides have bettors — escalate
        stake *= cfg.stake_multiplier

    proposal.total_stakes_yes = sum(s for _, s in yes_bets)
    proposal.total_stakes_no = sum(s for _, s in no_bets)
    proposal.n_participants = len(participating_agents)

    # If we exhausted rounds without one side folding, decide based on weight
    if proposal.method == "":
        total_yes = proposal.total_stakes_yes
        total_no = proposal.total_stakes_no

        if total_yes + total_no == 0:
            # No one participated at all — default reject
            proposal.approved = False
            proposal.method = "default-reject"
        elif cfg.stall_triggers_futarchy and proposal.y > cfg.futarchy_cost:
            # High importance + contested → futarchy
            futarchy_signal = x + rng.normal(0, cfg.futarchy_noise)
            proposal.approved = futarchy_signal > 0
            proposal.method = "futarchy"
        else:
            # Decide by weight of bets
            proposal.approved = total_yes > total_no
            proposal.method = "poker-weight"

    # Settle bets — pay winners from losers
    correct_side = "YES" if x > 0 else "NO"
    agent_map = {a.id: a for a in agents}

    for aid, s in yes_bets:
        agent = agent_map[aid]
        fee = cfg.fee_rate * s
        agent.wealth -= fee
        agent.n_bets += 1
        if correct_side == "YES":
            agent.wealth += s * 0.5  # simplified: win proportional share
            agent.total_pnl += s * 0.5 - fee
            agent.n_correct += 1
        else:
            agent.wealth -= s
            agent.total_pnl -= s + fee

    for aid, s in no_bets:
        agent = agent_map[aid]
        fee = cfg.fee_rate * s
        agent.wealth -= fee
        agent.n_bets += 1
        if correct_side == "NO":
            agent.wealth += s * 0.5
            agent.total_pnl += s * 0.5 - fee
            agent.n_correct += 1
        else:
            agent.wealth -= s
            agent.total_pnl -= s + fee

    return proposal


def run_simulation(cfg: SimConfig) -> dict:
    """Run full simulation across all proposals."""
    rng = np.random.default_rng(cfg.seed)

    # Generate proposals
    xs = rng.normal(0, 1, cfg.n_proposals)
    ys = rng.lognormal(cfg.mu_y, cfg.sigma_y, cfg.n_proposals)

    proposals = [Proposal(id=i, x=xs[i], y=ys[i]) for i in range(cfg.n_proposals)]
    agents = generate_agents(cfg, rng)

    # Run mechanism on each proposal
    for p in proposals:
        run_poker_for_proposal(p, agents, cfg, rng)

    # ── Compute metrics ──

    # Oracle: approve iff x > 0
    oracle_value = sum(p.x * p.y for p in proposals if p.x > 0)

    # Mechanism value
    mechanism_value = sum(p.x * p.y for p in proposals if p.approved)

    # Naive approve-all
    approve_all_value = sum(p.x * p.y for p in proposals)

    # Count decisions
    n_correct = sum(1 for p in proposals if (p.approved == (p.x > 0)))
    n_approved = sum(1 for p in proposals if p.approved)
    n_true_positive = sum(1 for p in proposals if p.approved and p.x > 0)
    n_false_positive = sum(1 for p in proposals if p.approved and p.x <= 0)
    n_true_negative = sum(1 for p in proposals if not p.approved and p.x <= 0)
    n_false_negative = sum(1 for p in proposals if not p.approved and p.x > 0)

    # By method
    methods = {}
    for p in proposals:
        m = p.method
        if m not in methods:
            methods[m] = {"count": 0, "correct": 0, "value": 0.0, "oracle_value": 0.0}
        methods[m]["count"] += 1
        methods[m]["correct"] += int(p.approved == (p.x > 0))
        if p.approved:
            methods[m]["value"] += p.x * p.y
        if p.x > 0:
            methods[m]["oracle_value"] += p.x * p.y

    # Futarchy costs
    n_futarchies = sum(1 for p in proposals if p.method == "futarchy")
    total_futarchy_cost = n_futarchies * cfg.futarchy_cost
    total_fees = sum(p.fees_collected for p in proposals)

    # Agent stats
    agent_stats = []
    for a in agents:
        accuracy = a.n_correct / a.n_bets if a.n_bets > 0 else 0
        agent_stats.append({
            "id": a.id,
            "tau": round(a.tau, 2),
            "cost": round(a.cost, 2),
            "n_bets": a.n_bets,
            "accuracy": round(accuracy, 3),
            "pnl": round(a.total_pnl, 2),
            "wealth": round(a.wealth, 2),
        })

    # Importance breakdown: how well do we do on high-y vs low-y?
    sorted_by_y = sorted(proposals, key=lambda p: p.y)
    quartiles = np.array_split(sorted_by_y, 4)
    importance_breakdown = []
    for i, q in enumerate(quartiles):
        q_list = list(q)
        q_correct = sum(1 for p in q_list if p.approved == (p.x > 0))
        q_value = sum(p.x * p.y for p in q_list if p.approved)
        q_oracle = sum(p.x * p.y for p in q_list if p.x > 0)
        q_y_range = (round(float(q_list[0].y), 2), round(float(q_list[-1].y), 2))
        importance_breakdown.append({
            "quartile": f"Q{i+1}",
            "y_range": q_y_range,
            "n": len(q_list),
            "accuracy": round(q_correct / len(q_list), 3),
            "value_captured": round(q_value, 1),
            "oracle_value": round(q_oracle, 1),
            "efficiency": round(q_value / q_oracle, 3) if q_oracle > 0 else None,
            "avg_rounds": round(np.mean([p.rounds_played for p in q_list]), 1),
            "avg_participants": round(np.mean([p.n_participants for p in q_list]), 1),
        })

    results = {
        "config": {
            "n_proposals": cfg.n_proposals,
            "n_agents": cfg.n_agents,
            "fee_rate": cfg.fee_rate,
            "n_rounds": cfg.n_rounds,
            "futarchy_cost": cfg.futarchy_cost,
        },
        "summary": {
            "oracle_value": round(oracle_value, 1),
            "mechanism_value": round(mechanism_value, 1),
            "approve_all_value": round(approve_all_value, 1),
            "efficiency": round(mechanism_value / oracle_value, 4) if oracle_value > 0 else None,
            "accuracy": round(n_correct / cfg.n_proposals, 4),
            "n_approved": n_approved,
            "true_positive": n_true_positive,
            "false_positive": n_false_positive,
            "true_negative": n_true_negative,
            "false_negative": n_false_negative,
        },
        "economics": {
            "total_fees_collected": round(total_fees, 1),
            "n_futarchies": n_futarchies,
            "total_futarchy_cost": round(total_futarchy_cost, 1),
            "net_mechanism_cost": round(total_futarchy_cost - total_fees, 1),
        },
        "by_method": methods,
        "by_importance_quartile": importance_breakdown,
        "agent_stats": agent_stats,
    }

    return results


def print_results(results: dict):
    """Pretty-print simulation results."""
    s = results["summary"]
    e = results["economics"]

    print("=" * 60)
    print("PROPOSAL POKER SIMULATION")
    print("=" * 60)

    print(f"\nProposals: {results['config']['n_proposals']}, "
          f"Agents: {results['config']['n_agents']}, "
          f"Fee: {results['config']['fee_rate']*100}%, "
          f"Rounds: {results['config']['n_rounds']}")

    print(f"\n--- Value Capture ---")
    print(f"Oracle value (perfect decisions):  {s['oracle_value']:>10.1f}")
    print(f"Mechanism value (poker+futarchy):  {s['mechanism_value']:>10.1f}")
    print(f"Approve-all value (no screening):  {s['approve_all_value']:>10.1f}")
    print(f"Efficiency (mechanism/oracle):     {s['efficiency']:>10.4f}")

    print(f"\n--- Accuracy ---")
    print(f"Overall accuracy: {s['accuracy']:.1%}")
    print(f"Approved: {s['n_approved']}  (TP={s['true_positive']}, FP={s['false_positive']})")
    print(f"Rejected: {results['config']['n_proposals'] - s['n_approved']}  "
          f"(TN={s['true_negative']}, FN={s['false_negative']})")

    print(f"\n--- Economics ---")
    print(f"Fees collected:     {e['total_fees_collected']:>8.1f}")
    print(f"Futarchies run:     {e['n_futarchies']:>8d}  (cost: {e['total_futarchy_cost']:.1f})")
    print(f"Net mechanism cost: {e['net_mechanism_cost']:>8.1f}")

    print(f"\n--- By Decision Method ---")
    for method, stats in results["by_method"].items():
        acc = stats["correct"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {method:<20s}  n={stats['count']:>4d}  accuracy={acc:.1%}  value={stats['value']:.1f}")

    print(f"\n--- By Importance Quartile ---")
    print(f"  {'Q':<4s} {'y range':<20s} {'acc':>6s} {'value':>8s} {'oracle':>8s} {'eff':>6s} {'rounds':>6s} {'agents':>6s}")
    for q in results["by_importance_quartile"]:
        eff_str = f"{q['efficiency']:.1%}" if q['efficiency'] is not None else "N/A"
        print(f"  {q['quartile']:<4s} {str(q['y_range']):<20s} {q['accuracy']:>6.1%} "
              f"{q['value_captured']:>8.1f} {q['oracle_value']:>8.1f} {eff_str:>6s} "
              f"{q['avg_rounds']:>6.1f} {q['avg_participants']:>6.1f}")

    print(f"\n--- Agent Performance (top 5 by PnL) ---")
    sorted_agents = sorted(results["agent_stats"], key=lambda a: a["pnl"], reverse=True)
    print(f"  {'id':>3s} {'tau':>5s} {'cost':>5s} {'bets':>5s} {'acc':>6s} {'pnl':>8s} {'wealth':>8s}")
    for a in sorted_agents[:5]:
        print(f"  {a['id']:>3d} {a['tau']:>5.2f} {a['cost']:>5.2f} {a['n_bets']:>5d} "
              f"{a['accuracy']:>6.1%} {a['pnl']:>8.1f} {a['wealth']:>8.1f}")
    print("  ...")
    for a in sorted_agents[-3:]:
        print(f"  {a['id']:>3d} {a['tau']:>5.2f} {a['cost']:>5.2f} {a['n_bets']:>5d} "
              f"{a['accuracy']:>6.1%} {a['pnl']:>8.1f} {a['wealth']:>8.1f}")


if __name__ == "__main__":
    import sys

    cfg = SimConfig()

    # Allow quick parameter overrides from CLI
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            if hasattr(cfg, key):
                field_type = type(getattr(cfg, key))
                setattr(cfg, key, field_type(val))

    results = run_simulation(cfg)
    print_results(results)
