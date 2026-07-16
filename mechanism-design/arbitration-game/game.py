"""Escalation-game engine for the FAO futarchy arbitration mechanism.

Mirrors futarchy-fi/FAO main @ 8cee5bc src/FutarchyArbitration.sol plus the
2026-02-05 Hanson draft. All token movements run through one ledger so that
sum(agent balances) + escrow + burn sink is invariant (== 0, balances start
at 0 and may go negative; only net flows matter).

Time is abstract hours. No wall-clock, all randomness via the engine rng.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

INACTIVE, YES, NO, QUEUED, EVALUATING, SETTLED = range(6)
EPS = 1e-9
MAX_QUEUE = 16  # contract: _tryGraduate reverts QueueFull at this occupancy

Bond = namedtuple("Bond", "bidder amount")


class Proposal(object):
    def __init__(self, pid, quality, created):
        self.pid = pid
        self.quality = bool(quality)
        self.created = created
        self.state = INACTIVE
        self.yes = None  # Bond or None
        self.no = None
        self.last_change = created
        self.settle_time = None
        self.accepted = None
        self.graduated = False


class Game(object):
    def __init__(self, timeout=72, d_eval=72, m=1.0, base_x=8.0, alpha=1.0,
                 congestion_guard=False, queue_cap=4, seed=0):
        self.timeout = timeout
        self.d_eval = d_eval
        self.m = m
        self.base_x = base_x
        self.alpha = alpha
        self.congestion_guard = congestion_guard
        self.queue_cap = queue_cap
        self.rng = np.random.default_rng(seed)
        self.proposals = []
        self.queue = []          # FIFO of (pid, t_queued)
        self.evaluating = None   # (pid, end_time)
        self._eval_free_at = 0.0
        self.balances = {}       # agent -> net flow (starts 0, may go negative)
        self.burned = 0.0        # address(0) sink

    # ledger -------------------------------------------------------------
    def _credit(self, agent, amount):
        self.balances[agent] = self.balances.get(agent, 0.0) + amount

    def _debit(self, agent, amount):
        self._credit(agent, -amount)

    def escrow(self):
        tot = 0.0
        for p in self.proposals:
            if p.yes is not None:
                tot += p.yes.amount
            if p.no is not None:
                tot += p.no.amount
        return tot

    def conservation_ok(self):
        return abs(sum(self.balances.values()) + self.escrow() + self.burned) < 1e-6

    # mechanism ----------------------------------------------------------
    def new_proposal(self, quality, t=0):
        p = Proposal(len(self.proposals), quality, t)
        self.proposals.append(p)
        return p.pid

    def grad_threshold(self):
        # doubles per proposal currently QUEUED (the one EVALUATING not counted)
        return self.base_x * 2 ** len(self.queue)

    def _queue_full(self):
        # contract: queuedLen + activeEvaluation >= MAX_QUEUE -> QueueFull
        return len(self.queue) + (self.evaluating is not None) >= MAX_QUEUE

    def place_yes(self, t, pid, bidder, amount):
        p = self.proposals[pid]
        flip = False
        if p.state == INACTIVE:
            if amount + EPS < self.m:
                raise ValueError("activation below m")
        elif p.state == NO:
            flip = True
            if amount + EPS < self.grad_threshold() and \
               amount + EPS < max(2 * p.no.amount, self.m):
                raise ValueError("YES bid below doubling/graduation threshold")
            if amount + EPS >= self.grad_threshold() and self._queue_full():
                raise ValueError("QueueFull")  # whole bid reverts, no ledger move
        else:
            raise ValueError("place_yes only from INACTIVE or NO")
        if p.yes is not None:  # replaced YES bond refunded to its bidder
            self._credit(p.yes.bidder, p.yes.amount)
        self._debit(bidder, amount)
        p.yes = Bond(bidder, amount)
        p.last_change = t
        if flip and amount + EPS >= self.grad_threshold():
            p.state = QUEUED  # amount >= grad threshold: straight to the queue
            p.graduated = True
            self.queue.append((pid, t))
        else:
            p.state = YES

    def place_no(self, t, pid, bidder):
        # match-only: the NO bond EXACTLY equals the standing YES bond
        p = self.proposals[pid]
        if p.state != YES:
            raise ValueError("place_no only from YES")
        if p.no is not None:  # replaced NO bond refunded
            self._credit(p.no.bidder, p.no.amount)
        amount = p.yes.amount
        self._debit(bidder, amount)
        p.no = Bond(bidder, amount)
        p.state = NO
        p.last_change = t

    def try_graduate(self, t, pid):
        p = self.proposals[pid]
        # contract: tryGraduate reverts InvalidState when noBond.amount == 0,
        # so an uncontested YES can never graduate
        if p.state != YES or p.no is None or \
           p.yes.amount + EPS < self.grad_threshold():
            return False
        if self._queue_full():
            raise ValueError("QueueFull")
        p.state = QUEUED
        p.graduated = True
        self.queue.append((pid, t))
        return True

    def finalize_by_timeout(self, t, pid):
        p = self.proposals[pid]
        if p.state not in (YES, NO):
            raise ValueError("nothing standing to finalize")
        if t + EPS < p.last_change + self.timeout:
            raise ValueError("timer not expired")
        if p.state == YES and self.congestion_guard and \
           len(self.queue) >= self.queue_cap:
            # design-doc §5 variant: no YES-by-timeout while eval queue is full
            raise ValueError("congestion guard: eval queue full")
        accepted = p.state == YES
        winner = p.yes.bidder if accepted else p.no.bidder
        self._settle(p, winner, accepted, t)
        return accepted

    def _settle(self, p, winner, accepted, t):
        pot = p.yes.amount + (p.no.amount if p.no is not None else 0.0)
        if winner is None:
            self.burned += pot  # address(0)
        else:
            self._credit(winner, pot)
        p.yes = p.no = None
        p.state = SETTLED
        p.accepted = accepted
        p.settle_time = t

    def advance(self, t):
        """Process the evaluation pipeline up to time t (one slot, FIFO)."""
        while True:
            if self.evaluating is None:
                if not self.queue:
                    return
                pid, tq = self.queue.pop(0)
                start = max(tq, self._eval_free_at)
                self.proposals[pid].state = EVALUATING
                self.evaluating = (pid, start + self.d_eval)
            pid, end = self.evaluating
            if end > t:
                return
            p = self.proposals[pid]
            correct = self.rng.random() < self.alpha
            accepted = p.quality if correct else (not p.quality)
            if accepted:
                winner = p.yes.bidder
            else:
                # address(0) burn on empty-NO rejection: defensive dead code
                # mirroring the contract -- graduation (flip or tryGraduate)
                # always carries a standing NO bond, so p.no is never None here
                winner = p.no.bidder if p.no is not None else None
            self._settle(p, winner, accepted, end)
            self._eval_free_at = end
            self.evaluating = None
