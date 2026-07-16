"""Agent behaviours on top of the escalation engine.

Beliefs p are P(ultimately accepted). Timeout-terminal decisions use p raw;
oracle-terminal decisions use the alpha-adjusted acceptance probability
p_oracle(p, alpha) = alpha*p + (1-alpha)*(1-p).
"""

from __future__ import annotations

import math

from game import EVALUATING, NO, QUEUED, YES


def p_oracle(p, alpha):
    return alpha * p + (1.0 - alpha) * (1.0 - p)


def _finish(g, pid):
    """Finalize a standing side by timeout, or run the queue to completion."""
    p = g.proposals[pid]
    if p.state in (YES, NO):
        g.finalize_by_timeout(p.last_change + g.timeout, pid)
    elif p.state in (QUEUED, EVALUATING):
        g.advance(math.inf)


def run_common_belief(g, pid, p, proposer="yes_crowd", challenger="no_crowd",
                      t0=0):
    """Rational crowd with a single common belief p.

    Activation is exogenous: the proposer posts YES bond m. From YES the
    crowd flips NO iff Y*(1-2p) > 0 (p < 1/2); from NO it flips YES at 2N iff
    N*(3p-2) > 0 (p > 2/3). Both cannot hold, so at most one flip occurs and
    nothing graduates; at p == 1/2 exactly the crowd is indifferent and the
    YES stands.
    """
    t = t0
    g.place_yes(t, pid, proposer, g.m)
    depth = 1
    while True:
        pr = g.proposals[pid]
        if pr.state == YES and pr.yes.amount * (1 - 2 * p) > 0:
            t += 1
            g.place_no(t, pid, challenger)
        elif pr.state == NO and pr.no.amount * (3 * p - 2) > 0:
            t += 1
            g.place_yes(t, pid, proposer, max(2 * pr.no.amount, g.m))
        else:
            break
        depth += 1
    _finish(g, pid)
    pr = g.proposals[pid]
    return {"accepted": pr.accepted, "graduated": pr.graduated,
            "settle_time": pr.settle_time - t0, "depth": depth}


def run_two_belief(g, pid, p_o, p_s, budget_o=math.inf, budget_s=math.inf,
                   t0=0, opt="optimists", skp="skeptics"):
    """Optimists (belief p_o) vs skeptics (p_s < 1/2 < p_o).

    Each side moves while +EV under its own belief and within budget (budget
    caps the side's capital at risk; replaced bonds are refunded so only the
    standing bond counts). Myopic EVs; the terminal model switches to the
    oracle when a move graduates the proposal. N == 0 graduation is not a
    move to price: the contract's tryGraduate reverts on an empty NO bond
    (structural, not economic), so only contested positions reach the queue.
    """
    t = t0
    g.place_yes(t, pid, opt, g.m)
    depth = 1
    po_orc = p_oracle(p_o, g.alpha)
    while depth < 200:
        pr = g.proposals[pid]
        if pr.state == YES:
            y = pr.yes.amount
            # skeptic EV of matching, timeout-terminal (sign-equivalent to the
            # oracle-terminal EV whenever p_s < 1/2)
            if y * (1 - 2 * p_s) > 0 and y <= budget_s + 1e-12:
                t += 1
                g.place_no(t, pid, skp)
                depth += 1
                continue
            break
        elif pr.state == NO:
            n = pr.no.amount
            thr = g.grad_threshold()
            a_flip = max(2 * n, g.m)
            if a_flip + 1e-12 >= thr:  # flip auto-queues -> oracle terminal
                ev_flip = po_orc * n - (1 - po_orc) * a_flip
            else:
                ev_flip = p_o * n - (1 - p_o) * a_flip
            ev_grad = po_orc * n - (1 - po_orc) * thr
            moves = [(ev_flip, a_flip), (ev_grad, thr)]
            moves = [mv for mv in moves
                     if mv[0] > 0 and mv[1] <= budget_o + 1e-12]
            if not moves:
                break
            amount = max(moves)[1]
            t += 1
            g.place_yes(t, pid, opt, amount)
            depth += 1
        else:
            break
    _finish(g, pid)
    pr = g.proposals[pid]
    if pr.graduated:
        outcome = "oracle_accept" if pr.accepted else "oracle_reject"
    else:
        outcome = "opt_timeout" if pr.accepted else "skp_timeout"
    return {"outcome": outcome, "accepted": pr.accepted,
            "graduated": pr.graduated, "depth": depth,
            "settle_time": pr.settle_time - t0,
            "net_opt": g.balances.get(opt, 0.0),
            "net_skp": g.balances.get(skp, 0.0)}


def run_delay_adversary(g, pid, budget, adversary="adv", defender="def"):
    """Adversary postpones settlement of a target proposal.

    The only legal NO is the exact match, placed just before expiry
    (lastChange + TIMEOUT - 1) while the standing bond fits the budget (only
    the standing NO is at risk; replaced ones are refunded). The defender is
    deliberately INFORMED, conditioning on ground truth rather than an
    EV/belief rule: it re-flips at 2N iff the proposal is actually good
    (quality ~ Bern(p) in S1C, so each flip is a bet the adversary loses
    w.p. p: expected cost/window = Y*(2p-1)). The strictly EV-rational crowd
    the spec implies (flip iff p > 2/3) would abandon for every swept p and
    make that claim untestable, so S1C measures delay against an informed
    defender, not a rational crowd. Defense auto-graduates once 2N reaches
    the threshold.
    """
    g.place_yes(0, pid, defender, g.m)
    baseline = g.timeout  # uncontested settle time
    flips = 0
    while True:
        pr = g.proposals[pid]
        if pr.state == YES:
            y = pr.yes.amount
            if y <= budget + 1e-12:
                g.place_no(pr.last_change + g.timeout - 1, pid, adversary)
                flips += 1
                continue
            break
        elif pr.state == NO:
            if pr.quality:
                g.place_yes(pr.last_change + 1, pid, defender,
                            max(2 * pr.no.amount, g.m))
                continue
            break
        else:
            break
    _finish(g, pid)
    pr = g.proposals[pid]
    return {"flips": flips,
            "delay_windows": (pr.settle_time - baseline) / float(g.timeout),
            "realized_cost": -g.balances.get(adversary, 0.0),
            "outcome_flipped": not pr.accepted,
            "graduated": pr.graduated}
