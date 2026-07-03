"""Example Bayesian threshold agent."""

from __future__ import annotations

import math

from proposal_poker.interfaces import AgentBase
from proposal_poker.types import Contribution


class BayesianThresholdAgent(AgentBase):
    """Stake once when posterior confidence clears a net-utility threshold."""

    agent_id = "bayesian_threshold"

    def __init__(
        self,
        min_stake: float = 0.5,
        max_stake: float | None = None,
        confidence_threshold: float = 0.55,
        precision_ratio: float = 2.0,
        avoid_likely_minority: bool = True,
        phi: float = 0.01,
        fee_rate: float = 0.01,
        search_points: int = 96,
        **params: object,
    ) -> None:
        super().__init__(**params)
        if min_stake <= 0:
            raise ValueError("min_stake must be positive")
        if max_stake is not None and max_stake < min_stake:
            raise ValueError("max_stake must be >= min_stake")
        if not 0.5 <= confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be in [0.5, 1.0)")
        if precision_ratio <= 0:
            raise ValueError("precision_ratio must be positive")
        if phi < 0:
            raise ValueError("phi must be non-negative")
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        if search_points < 4:
            raise ValueError("search_points must be at least 4")

        self.min_stake = float(min_stake)
        self.max_stake = None if max_stake is None else float(max_stake)
        self.confidence_threshold = float(confidence_threshold)
        self.precision_ratio = float(precision_ratio)
        self.avoid_likely_minority = bool(avoid_likely_minority)
        self.phi = float(phi)
        self.fee_rate = float(fee_rate)
        self.search_points = int(search_points)

    def act(
        self,
        wealth: float,
        signal: float,
        y: float,
        public_history: list[object],
        my_past: list[Contribution],
    ) -> Contribution | None:
        if my_past:
            return None

        tau = max(1e-12, self.precision_ratio * wealth)
        posterior_precision = 1.0 + tau
        posterior_mean = (tau / posterior_precision) * signal
        posterior_std = 1.0 / math.sqrt(posterior_precision)

        z_score = posterior_mean / posterior_std
        prob_approve = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        confidence = max(prob_approve, 1.0 - prob_approve)

        if confidence < self.confidence_threshold:
            return None

        side = "approve" if posterior_mean > 0.0 else "reject"
        stake, expected_utility = self._optimal_stake(
            side=side,
            wealth=wealth,
            y=y,
            win_probability=confidence,
            public_history=public_history,
        )
        if stake <= 0.0 or expected_utility <= 0.0:
            return None

        return Contribution(amount=stake, data={"side": side})

    def _optimal_stake(
        self,
        side: str,
        wealth: float,
        y: float,
        win_probability: float,
        public_history: list[object],
    ) -> tuple[float, float]:
        max_affordable_stake = self._max_affordable_stake(wealth=wealth, y=y)
        if max_affordable_stake < self.min_stake:
            return 0.0, 0.0

        max_search_stake = max_affordable_stake
        if self.max_stake is not None:
            max_search_stake = min(max_search_stake, self.max_stake)
        if max_search_stake < self.min_stake:
            return 0.0, 0.0

        best_stake = 0.0
        best_utility = 0.0
        candidate_stakes = {self.min_stake, max_search_stake}
        for idx in range(1, self.search_points + 1):
            fraction = idx / self.search_points
            candidate_stakes.add(self.min_stake + fraction * (max_search_stake - self.min_stake))
            candidate_stakes.add(
                self.min_stake + (fraction * fraction) * (max_search_stake - self.min_stake)
            )

        for stake in sorted(candidate_stakes):
            if self.avoid_likely_minority and self._would_likely_be_minority(side, stake, public_history):
                continue
            expected_utility = self._expected_trade_utility(
                side=side,
                stake=stake,
                wealth=wealth,
                y=y,
                win_probability=win_probability,
                public_history=public_history,
            )
            if expected_utility > best_utility:
                best_stake = stake
                best_utility = expected_utility

        return best_stake, best_utility

    def _max_affordable_stake(self, wealth: float, y: float) -> float:
        participation_cost = self.phi * wealth * math.sqrt(y)
        spendable_wealth = wealth - participation_cost
        if spendable_wealth <= 0.0:
            return 0.0

        # Keep losing-state terminal wealth strictly positive for log utility.
        return max(0.0, math.nextafter(spendable_wealth / (1.0 + self.fee_rate), 0.0))

    def _would_likely_be_minority(self, side: str, stake: float, public_history: list[object]) -> bool:
        approve_stake, reject_stake, _winner_subsidy = self._current_market(public_history)

        if side == "approve":
            # Approve loses on tie, so it needs a strict lead.
            return approve_stake + stake <= reject_stake

        # Reject wins on tie in this mechanism.
        return reject_stake + stake < approve_stake

    def _expected_trade_utility(
        self,
        side: str,
        stake: float,
        wealth: float,
        y: float,
        win_probability: float,
        public_history: list[object],
    ) -> float:
        approve_stake, reject_stake, winner_subsidy = self._current_market(public_history)

        if side == "approve":
            approve_stake += stake
        else:
            reject_stake += stake

        total_stake = approve_stake + reject_stake
        winning_pool = approve_stake if side == "approve" else reject_stake
        payout_if_win = stake * (total_stake + winner_subsidy) / winning_pool

        participation_cost = self.phi * wealth * math.sqrt(y)
        fee_cost = self.fee_rate * stake

        wealth_if_win = wealth + payout_if_win - stake - participation_cost - fee_cost
        wealth_if_lose = wealth - stake - participation_cost - fee_cost

        if wealth_if_win <= 0.0 or wealth_if_lose <= 0.0:
            return float("-inf")

        lose_probability = 1.0 - win_probability
        return (
            win_probability * math.log(wealth_if_win)
            + lose_probability * math.log(wealth_if_lose)
            - math.log(wealth)
        )

    def _current_market(self, public_history: list[object]) -> tuple[float, float, float]:
        if not public_history:
            return 0.0, 0.0, 0.0

        last_message = public_history[-1]
        if not isinstance(last_message, dict):
            return 0.0, 0.0, 0.0

        approve_stake = float(last_message.get("approve_stake", 0.0))
        reject_stake = float(last_message.get("reject_stake", 0.0))
        winner_subsidy = float(last_message.get("winner_subsidy", 0.0))
        return approve_stake, reject_stake, winner_subsidy
