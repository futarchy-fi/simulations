"""Environments: hidden state, signals, myopic-Bayes posteriors, belief updates.

Two environments, one interface:

* ``GaussianEnv`` — v ~ N(0,1), signals s_i = v + eps_i with eps_i ~ N(0, sigma_eps^2).
  The traded security pays 1{v > 0}. M Monte Carlo replications are carried as
  a leading array axis; every operation is closed-form, so the whole engine is
  vectorised over reps.

* ``GalanisEnv`` — the discrete 3-bit "Easy" structure ``t3s111y2`` from
  galanis-market: omega uniform on {0,1}^3, trader i observes bit i, the
  security pays 1{>=2 bits set}. The 8 states are enumerated exactly
  (M = 8 "reps" with weights 1/8), so all reported numbers are exact
  expectations, no Monte Carlo error.

Behavioral model (both envs) — **fully-Bayesian myopic traders**, the same
assumption as galanis-market's ``myopic.py``: each trader believes prices are
set by myopic-Bayes play and that this is common knowledge. Trades are
therefore *inverted* by the public into the signal that would rationalise
them, and that inferred (pseudo-)signal enters the public belief. Honest
traders' trades invert exactly; a manipulator's distorted trade injects a
*wrong* pseudo-signal — the public does not discount (no suspicion), which is
the deliberate behavioral contrast with the CFR+ equilibrium results in
mechanism-design/MANIPULATION.md.

Belief-update conventions (documented deviations):

* Gaussian: signals are revealed (inverted) only on each trader's FIRST trade
  (round 1). Under the common-knowledge myopic protocol later trades carry no
  new information about the trader's signal, so the public ignores them for
  belief purposes. (A manipulator's later-round pushes therefore move the
  price but not the belief, and honest traders pull the price back — the
  behavioral analogue of "correction".)
* Galanis: beliefs update on every trade via the cell-consistency inversion of
  ``myopic.py``; if NO cell is consistent with an observed trade (only possible
  under manipulation), the belief is left unchanged (myopic.py's fall-back to
  the true cell would leak the manipulator's actual signal to the public,
  which is unrealistic).

Each env exposes:
  n, m, weights, payout          — sizes, rep weights, security payoff (M,)
  make_state()                   — fresh public-belief state for one run
  honest_target(i, state)        — trader i's posterior price (M,)
  reveal(i, implied_target, state, first_time)          — SEQ update
  reveal_batch(implied_targets, state, first_time)      — BATCH update, FULL
                                                           disclosure (N, M)
  reveal_batch_anon(t_total, state, first_time)         — BATCH update under
                                                           ANONYMOUS disclosure
  full_info_price()              — posterior given ALL signals (M,)

Disclosure regimes (see Config.disclosure in engine.py): "full" publishes
per-trader orders between rounds (reveal_batch); "aggregate" publishes only
the clearing price + net flow, and "price" only the clearing price. Against a
deterministic AMM the clearing price pins down the net flow exactly (the
price move is an invertible function of it), so "aggregate" and "price" are
informationally identical and share reveal_batch_anon; the only knob with
bite is attribution. Observers under anonymity see T = sum_i logit(implied
target_i), recoverable from the net flow given the common-knowledge sizing
rule.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
from scipy.special import expit as _sigmoid
from scipy.special import logit as _logit
from scipy.stats import norm

from batch_amm.clearing import clear_limit_batch
from galanis_market.structures import STATES, STRUCTURES, Structure

PRICE_EPS = 1e-4  # global price clip: p in [PRICE_EPS, 1 - PRICE_EPS]


def clip_price(p: np.ndarray) -> np.ndarray:
    return np.clip(p, PRICE_EPS, 1.0 - PRICE_EPS)


# --------------------------------------------------------------------------- #
# Gaussian environment
# --------------------------------------------------------------------------- #


class GaussianEnv:
    """v ~ N(0,1); s_i = v + eps_i; security pays 1{v > 0}."""

    exact = False
    target_bounds = (PRICE_EPS, 1.0 - PRICE_EPS)

    def __init__(self, n: int, m: int, sigma_eps: float = 1.0, seed: int = 0):
        self.n = n
        self.m = m
        self.sigma_eps = sigma_eps
        rng = np.random.default_rng(seed)
        self.v = rng.standard_normal(m)
        self.signals = self.v[:, None] + sigma_eps * rng.standard_normal((m, n))
        self.weights = np.full(m, 1.0 / m)
        self.payout = (self.v > 0).astype(float)
        self._tau_sig = 1.0 / sigma_eps**2

    # -- posterior machinery -------------------------------------------------
    def _posterior_price(self, s_sum: np.ndarray, k) -> np.ndarray:
        """P(v>0 | k signals summing to s_sum). k scalar or (M,)."""
        tau = 1.0 + np.asarray(k, dtype=float) * self._tau_sig
        mu = (s_sum * self._tau_sig) / tau
        return clip_price(norm.cdf(mu * np.sqrt(tau)))

    def make_state(self) -> Dict:
        return {
            "sum": np.zeros(self.m),
            "count": 0,
            "pseudo": np.zeros((self.m, self.n)),
            "revealed": np.zeros(self.n, dtype=bool),
        }

    def honest_target(self, i: int, state: Dict) -> np.ndarray:
        """Posterior price given public belief + trader i's true signal.

        If trader i's (pseudo-)signal is already in the public pool, it is
        replaced by their true signal (identical for honest traders; a
        manipulator knows their own revealed pseudo-signal is fake).
        """
        if state["revealed"][i]:
            s_sum = state["sum"] - state["pseudo"][:, i] + self.signals[:, i]
            k = state["count"]
        else:
            s_sum = state["sum"] + self.signals[:, i]
            k = state["count"] + 1
        return self._posterior_price(s_sum, k)

    def _invert(self, implied_target: np.ndarray, state: Dict) -> np.ndarray:
        """Pseudo-signal that rationalises `implied_target` given round-open belief."""
        t = clip_price(implied_target)
        tau1 = 1.0 + (state["count"] + 1) * self._tau_sig
        mu1 = norm.ppf(t) / np.sqrt(tau1)
        return (tau1 * mu1 - state["sum"] * self._tau_sig) / self._tau_sig

    def reveal(self, i: int, implied_target: np.ndarray, state: Dict, first_time: bool) -> None:
        if not first_time:
            return  # later trades carry no signal information (see module doc)
        s_hat = self._invert(implied_target, state)
        state["pseudo"][:, i] = s_hat
        state["sum"] = state["sum"] + s_hat
        state["count"] += 1
        state["revealed"][i] = True

    def reveal_batch(self, implied_targets: np.ndarray, state: Dict, first_time: bool) -> None:
        """All N orders are disclosed post-clearing; each is inverted against
        the SAME round-open belief (each trader conditioned on it + own signal)."""
        if not first_time:
            return
        s_hats = np.stack(
            [self._invert(implied_targets[i], state) for i in range(self.n)], axis=1
        )  # inversion uses round-open state; do not mutate until all inverted
        state["pseudo"][:, :] = s_hats
        state["sum"] = state["sum"] + s_hats.sum(axis=1)
        state["count"] += self.n
        state["revealed"][:] = True

    def reveal_batch_anon(self, t_total: np.ndarray, state: Dict, first_time: bool) -> None:
        """ANONYMOUS disclosure: mean-field inversion of the aggregate.

        Observers see only T = sum_i logit(implied target_i). Treat T as N
        identical "average" traders: q_bar = sigmoid(T/N), invert q_bar into
        ONE pseudo-signal s_bar against the round-open belief, enter N copies
        of s_bar into the pool. Each trader's own pool copy is then swapped
        for their exact signal by honest_target (they know their own order;
        to first order that subtracts their own contribution from the
        aggregate). Exact when all signals are equal (unit-tested); the
        Jensen gap of s -> logit(posterior(s)) is the information anonymity
        destroys. As with FULL disclosure, only round 1's aggregate is
        treated as informative.
        """
        if not first_time:
            return
        q_bar = clip_price(_sigmoid(t_total / self.n))
        s_bar = self._invert(q_bar, state)
        state["pseudo"][:, :] = s_bar[:, None]  # each seat holds one s_bar copy
        state["sum"] = state["sum"] + self.n * s_bar
        state["count"] += self.n
        state["revealed"][:] = True

    def full_info_price(self) -> np.ndarray:
        return self._posterior_price(self.signals.sum(axis=1), self.n)

    def public_price(self, state: Dict) -> np.ndarray:
        return self._posterior_price(state["sum"], state["count"])

    def with_trader_order(self, perm) -> "GaussianEnv":
        """Same draws, seat k now held by original trader perm[k] (CRN seat sweeps)."""
        out = copy.copy(self)
        out.signals = self.signals[:, list(perm)]
        return out


# --------------------------------------------------------------------------- #
# Galanis "Easy" (t3s111y2) environment — exact enumeration
# --------------------------------------------------------------------------- #

_CONSISTENCY_TOL = 1e-6


class GalanisEnv:
    """3 traders, 1 bit each, X = 1{>=2 bits}; 8 states enumerated exactly.

    ``cap`` bounds every posted quote to [cap, 1-cap] (default 0.1/0.9),
    mirroring the 9-point tabular grid floor of the galanis-market CFR+
    results (mechanism-design/MANIPULATION.md), so log-loss and PnL scales
    are directly comparable (their baseline log err 0.105 = ln(1/0.9)).
    """

    exact = True

    def __init__(self, structure: Structure = None, cap: float = 0.1):
        self.cap = cap
        self.target_bounds = (cap, 1.0 - cap)
        self.structure = structure or STRUCTURES["t3s111y2"]
        self.n = 3
        self.m = 8
        self.weights = np.full(8, 1.0 / 8.0)
        self.payout = np.array(
            [float(self.structure.x_of(STATES[w])) for w in range(8)]
        )
        self.cells = np.array(
            [
                [self.structure.cell_of(t, STATES[w]) for t in range(3)]
                for w in range(8)
            ]
        )  # (8 reps, 3 traders)

    def make_state(self) -> Dict:
        # "jams": count of (rep, round) anonymous updates rejected as
        # unexplainable (denial-of-aggregation instrumentation, BATCH.md §9)
        return {"belief": np.full((8, 8), 1.0 / 8.0), "jams": 0}  # (rep, omega)

    def _target_given_cell(self, i: int, cell: int, belief_row: np.ndarray) -> float:
        mask = self.cells[:, i] == cell  # cell_of depends only on omega
        z = belief_row[mask].sum()
        if z <= 0.0:
            return 0.5
        return float((belief_row[mask] * self.payout[mask]).sum() / z)

    def _cap(self, p):
        return np.clip(p, self.target_bounds[0], self.target_bounds[1])

    def honest_target(self, i: int, state: Dict) -> np.ndarray:
        out = np.empty(8)
        for rep in range(8):
            out[rep] = self._target_given_cell(
                i, self.cells[rep, i], state["belief"][rep]
            )
        return self._cap(out)

    def _consistent_mask(
        self, i: int, implied: float, belief_row: np.ndarray
    ) -> np.ndarray:
        """Omega-mask of states whose trader-i cell rationalises `implied`."""
        ok_cells = [
            c
            for c in range(self.structure.cells_per_trader)
            # compare in observable (capped-price) space: targets of 0/1 are
            # posted at the cap, so cap before matching; if two cells collide
            # at the cap both stay consistent (partial revelation, like the
            # tabular grid floor)
            if abs(
                float(self._cap(self._target_given_cell(i, c, belief_row)))
                - implied
            )
            < _CONSISTENCY_TOL
        ]
        if not ok_cells:
            return np.ones(8, dtype=bool)  # unexplainable trade: no update
        return np.isin(self.cells[:, i], ok_cells)

    def reveal(self, i: int, implied_target: np.ndarray, state: Dict, first_time: bool) -> None:
        del first_time  # Galanis updates on every trade (see module doc)
        for rep in range(8):
            row = state["belief"][rep]
            mask = self._consistent_mask(i, float(implied_target[rep]), row)
            new = row * mask
            z = new.sum()
            if z > 0:
                state["belief"][rep] = new / z

    def reveal_batch(self, implied_targets: np.ndarray, state: Dict, first_time: bool) -> None:
        del first_time
        for rep in range(8):
            row = state["belief"][rep]
            mask = np.ones(8, dtype=bool)
            for i in range(self.n):  # all inverted vs the same round-open belief
                mask &= self._consistent_mask(i, float(implied_targets[i, rep]), row)
            new = row * mask
            z = new.sum()
            if z > 0:
                state["belief"][rep] = new / z

    def reveal_batch_anon(self, t_total: np.ndarray, state: Dict, first_time: bool) -> None:
        """ANONYMOUS disclosure: exact Bayesian update on the aggregate.

        Observers see only T = sum_i logit(capped target_i). Because the state
        space is discrete, the update is exact: keep the omegas whose honest
        myopic quote profile (given the round-open belief) predicts the
        observed T. Anonymity coarsens the round-1 partition from "which
        trader holds which bit" to "how many bits are set" — which is exactly
        sufficient for the symmetric payoff X = 1{>=2 bits} (unit-tested).
        An unexplainable T (manipulation) leaves the belief unchanged, as in
        the attributed fall-back.
        """
        del first_time  # Galanis updates on every round (see module doc)
        for rep in range(8):
            row = state["belief"][rep]
            support = row > 0.0
            t_pred = np.full(8, np.nan)
            for w in range(8):
                if not support[w]:
                    continue
                t_pred[w] = sum(
                    float(
                        _logit(
                            self._cap(self._target_given_cell(i, self.cells[w, i], row))
                        )
                    )
                    for i in range(self.n)
                )
            mask = support & (np.abs(t_pred - float(t_total[rep])) < 1e-6)
            new = row * mask
            z = new.sum()
            if z > 0:
                state["belief"][rep] = new / z
            else:
                state["jams"] += 1

    def reveal_batch_anon_limit(
        self,
        p_open: np.ndarray,
        x_obs: np.ndarray,
        state: Dict,
        b: float,
        scale: float,
        slack: float,
        first_time: bool,
    ) -> None:
        """ANONYMOUS price-only disclosure under LIMIT orders: exact update.

        Observers see only the clearing price, which inverts to the EXECUTED
        net flow X; submitted-but-unfilled quantities are invisible to them.
        Keep the omegas whose honest limit-order profile (given the
        round-open belief, mirroring the engine's sizing/limit arithmetic)
        clears to the observed X; an unexplainable X (manipulation) leaves
        the belief unchanged and counts a jam, as in reveal_batch_anon.
        Consistency is checked in flow units at scale*b*1e-6 — the image of
        reveal_batch_anon's 1e-6 aggregate-logit tolerance.
        """
        del first_time  # Galanis updates on every round (see module doc)
        tol = scale * b * 1e-6
        for rep in range(8):
            row = state["belief"][rep]
            support = np.where(row > 0.0)[0]
            p_vec = np.full(len(support), float(p_open[rep]))
            ts = np.empty((self.n, len(support)))
            for j, w in enumerate(support):
                for i in range(self.n):
                    ts[i, j] = self._cap(
                        self._target_given_cell(i, self.cells[w, i], row)
                    )
            x = scale * b * (_logit(ts) - _logit(p_vec)[None, :])
            if np.isinf(slack):
                lims = np.where(x >= 0.0, np.inf, -np.inf)
            else:
                lims = np.where(
                    x >= 0.0,
                    np.minimum(ts + slack, 1.0),
                    np.maximum(ts - slack, 0.0),
                )
            x_pred = clear_limit_batch(p_vec, x, lims, b)["x_exec"].sum(axis=0)
            mask = np.zeros(8, dtype=bool)
            mask[support] = np.abs(x_pred - float(x_obs[rep])) < tol
            new = row * mask
            z = new.sum()
            if z > 0:
                state["belief"][rep] = new / z
            else:
                state["jams"] += 1

    def full_info_price(self) -> np.ndarray:
        return self._cap(self.payout.copy())  # all 3 bits determine X exactly

    def with_trader_order(self, perm) -> "GalanisEnv":
        """Same states, seat k now observes original trader perm[k]'s bit."""
        out = copy.copy(self)
        out.cells = self.cells[:, list(perm)]
        return out

    def public_price(self, state: Dict) -> np.ndarray:
        return clip_price((state["belief"] * self.payout[None, :]).sum(axis=1))


def copy_state(state: Dict) -> Dict:
    return copy.deepcopy(state)


__all__ = ["GaussianEnv", "GalanisEnv", "clip_price", "copy_state", "PRICE_EPS"]
