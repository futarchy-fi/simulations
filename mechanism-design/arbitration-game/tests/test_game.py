"""Mechanism tests for the escalation game engine."""

import numpy as np
import pytest

from agents import run_common_belief
from game import Game, NO, QUEUED, SETTLED, YES


def test_common_belief_accept_band():
    for p in (0.1, 0.3, 0.7, 0.9):
        g = Game(seed=1)
        pid = g.new_proposal(quality=True)
        rec = run_common_belief(g, pid, p)
        assert rec["accepted"] == (p > 0.5)
        assert not rec["graduated"]
        assert g.conservation_ok()


def test_forced_alternation_doubles():
    g = Game(base_x=1e9)  # keep the graduation threshold out of the way
    pid = g.new_proposal(quality=True)
    g.place_yes(0, pid, "y", g.m)
    for k in range(1, 7):
        g.place_no(k, pid, "n")
        g.place_yes(k, pid, "y", 2 * g.proposals[pid].no.amount)
        assert g.proposals[pid].yes.amount == 2 ** k * g.m


def test_no_is_match_only():
    g = Game()
    pid = g.new_proposal(quality=True)
    g.place_yes(0, pid, "y", 3.7)
    g.place_no(1, pid, "n")
    assert g.proposals[pid].no.amount == g.proposals[pid].yes.amount == 3.7


def test_threshold_yes_graduates_past_large_no():
    g = Game()  # base_x = 8
    pid = g.new_proposal(quality=True)
    g.place_yes(0, pid, "y", 100.0)  # any amount >= m from INACTIVE
    g.place_no(1, pid, "n")          # large NO = 100
    # 8 < 2*100 but 8 >= grad_threshold: ALWAYS accepted, straight to queue
    g.place_yes(2, pid, "y2", g.grad_threshold())
    assert g.proposals[pid].state == QUEUED
    assert g.proposals[pid].graduated


def test_timers_and_early_finalize():
    g = Game()
    pid = g.new_proposal(quality=True)
    g.place_yes(0, pid, "y", 10.0)
    assert g.proposals[pid].last_change == 0
    with pytest.raises(ValueError):
        g.finalize_by_timeout(71, pid)
    # uncontested YES never graduates (contract: tryGraduate reverts on
    # empty noBond) and the attempt does not touch the timer
    assert not g.try_graduate(5, pid)
    assert g.proposals[pid].state == YES
    assert g.proposals[pid].last_change == 0
    g2 = Game()
    pid2 = g2.new_proposal(quality=True)
    g2.place_yes(0, pid2, "y", 1.0)
    g2.place_no(10, pid2, "n")  # flips reset lastChange
    assert g2.proposals[pid2].last_change == 10
    g2.advance(50)
    assert g2.proposals[pid2].last_change == 10
    with pytest.raises(ValueError):
        g2.finalize_by_timeout(81, pid2)
    assert g2.finalize_by_timeout(82, pid2) is False  # NO stands -> rejected


def test_conservation_fuzz_500_events():
    rng = np.random.default_rng(20260716)
    g = Game(alpha=1.0, seed=7)
    t = 0.0
    for ev in range(500):
        t += float(rng.integers(1, 30))
        live = [p for p in g.proposals if p.state in (0, YES, NO)]
        act = rng.integers(0, 5)
        if act == 0 or not live:
            pid = g.new_proposal(quality=bool(rng.random() < 0.4), t=t)
            g.place_yes(t, pid, "a%d" % rng.integers(5), g.m * (1 + rng.random()))
        else:
            p = live[rng.integers(len(live))]
            who = "a%d" % rng.integers(5)
            if act == 1 and p.state == YES:
                g.place_no(t, p.pid, who)
            elif act == 2 and p.state == NO:
                hi = max(2 * p.no.amount, g.m)
                amt = hi if rng.random() < 0.8 else g.grad_threshold()
                g.place_yes(t, p.pid, who, amt)
            elif act == 3 and p.state in (YES, NO):
                if t >= p.last_change + g.timeout:
                    g.finalize_by_timeout(t, p.pid)
            elif act == 4 and p.state == YES:
                g.try_graduate(t, p.pid)
        g.advance(t)
        assert g.conservation_ok(), "conservation broke at event %d" % ev
    # uncontested graduation is structurally impossible, so the address(0)
    # burn branch is never reachable and the sink stays empty
    pid = g.new_proposal(quality=False, t=t)
    g.place_yes(t, pid, "a0", g.grad_threshold())
    assert not g.try_graduate(t, pid)
    g.advance(t + 100 * g.d_eval + 1e6)
    assert g.burned == 0.0
    assert g.conservation_ok()


def test_contested_roi():
    # timeout ACCEPT: staked 2 after 1,1,2 escalation, wins the NO bond 1: +50%
    g = Game()
    pid = g.new_proposal(quality=True)
    g.place_yes(0, pid, "y", 1.0)
    g.place_no(1, pid, "n")
    g.place_yes(2, pid, "y", 2.0)
    assert g.finalize_by_timeout(74, pid) is True
    assert g.balances["y"] == pytest.approx(0.5 * 2.0)
    # oracle ACCEPT: escalate 1,1,2,2,4,4 then 8 >= threshold graduates;
    # YES staked 8, wins the standing NO bond 4: +50%
    g = Game(alpha=1.0)
    pid = g.new_proposal(quality=True)
    g.place_yes(0, pid, "y", 1.0)
    for k in (1, 2):
        g.place_no(2 * k - 1, pid, "n")
        g.place_yes(2 * k, pid, "y", 2 * g.proposals[pid].no.amount)
    g.place_no(5, pid, "n")
    g.place_yes(6, pid, "y", 8.0)
    assert g.proposals[pid].state == QUEUED
    g.advance(1000)
    assert g.proposals[pid].accepted
    assert g.balances["y"] == pytest.approx(0.5 * 8.0)
    # timeout REJECT: NO staked 1, wins the YES bond 1: +100%
    g = Game()
    pid = g.new_proposal(quality=False)
    g.place_yes(0, pid, "y", 1.0)
    g.place_no(1, pid, "n")
    assert g.finalize_by_timeout(73, pid) is False
    assert g.balances["n"] == pytest.approx(1.0)


def test_try_graduate_requires_standing_no():
    # uncontested YES at threshold: contract reverts (sim: returns False)
    g = Game(alpha=1.0)
    pid = g.new_proposal(quality=False)
    g.place_yes(0, pid, "y", 8.0)
    assert not g.try_graduate(0, pid)
    assert g.proposals[pid].state == YES and g.burned == 0.0
    # contested path: flip lands below an inflated threshold, queue drains,
    # then tryGraduate succeeds without touching the timer
    g2 = Game(alpha=1.0)
    blk = g2.new_proposal(quality=True)
    g2.place_yes(0, blk, "y", 1.0)
    g2.place_no(0, blk, "n")
    g2.place_yes(0, blk, "y", g2.grad_threshold())  # queued, threshold -> 16
    pid2 = g2.new_proposal(quality=True)
    g2.place_yes(0, pid2, "y", 4.0)
    g2.place_no(1, pid2, "n")
    g2.place_yes(2, pid2, "y", 8.0)  # >= 2N but < 16: stays YES
    assert g2.proposals[pid2].state == YES
    g2.advance(100)  # blocker settles, queue empty, threshold back to 8
    assert g2.try_graduate(100, pid2)
    assert g2.proposals[pid2].state == QUEUED
    assert g2.proposals[pid2].last_change == 2  # graduation never resets timer
    g2.advance(1000)
    assert g2.proposals[pid2].state == SETTLED and g2.proposals[pid2].accepted
    assert g2.burned == 0.0
    assert g2.conservation_ok()


def test_queue_cap_16():
    # contract MAX_QUEUE: 17th graduation reverts QueueFull
    g = Game()
    for k in range(17):
        pid = g.new_proposal(quality=True)
        g.place_yes(0, pid, "y", 1.0)
        g.place_no(0, pid, "n")
        if k < 16:
            g.place_yes(0, pid, "y", g.grad_threshold())
            assert g.proposals[pid].state == QUEUED
        else:
            with pytest.raises(ValueError):
                g.place_yes(0, pid, "y", g.grad_threshold())
    assert len(g.queue) == 16
    assert g.conservation_ok()
