from __future__ import annotations

import math
from typing import Literal

import pytest
from pydantic import BaseModel

from proposal_poker.discovery import SubmissionRegistry
from proposal_poker.interfaces import AgentBase, MechanismBase
from proposal_poker.scenario import ScenarioConfig
from proposal_poker.simulator import run_simulation
from proposal_poker.types import Contribution, Receipt


class _SideData(BaseModel):
    side: Literal["approve", "reject"]


class FixedApproveAgent(AgentBase):
    agent_id = "fixed_approve"

    def __init__(self, amount: float = 1.0, **params: object) -> None:
        super().__init__(**params)
        self.amount = amount

    def act(self, wealth, signal, y, public_history, my_past):
        del wealth, signal, y, public_history
        if my_past:
            return None
        return Contribution(amount=self.amount, data={"side": "approve"})


class FixedRejectAgent(AgentBase):
    agent_id = "fixed_reject"

    def __init__(self, amount: float = 1.0, **params: object) -> None:
        super().__init__(**params)
        self.amount = amount

    def act(self, wealth, signal, y, public_history, my_past):
        del wealth, signal, y, public_history
        if my_past:
            return None
        return Contribution(amount=self.amount, data={"side": "reject"})


class AllInAgent(AgentBase):
    agent_id = "all_in"

    def __init__(self, amount: float = 1_000_000.0, **params: object) -> None:
        super().__init__(**params)
        self.amount = amount

    def act(self, wealth, signal, y, public_history, my_past):
        del wealth, signal, y, public_history
        if my_past:
            return None
        return Contribution(amount=self.amount, data={"side": "approve"})


class OneRoundParimutuel(MechanismBase):
    mechanism_id = "one_round"

    def init(self):
        return {"approve": 0.0, "reject": 0.0}

    def publish(self, state):
        return dict(state)

    def on_contribution(self, state, contribution):
        next_state = dict(state)
        side = contribution.data["side"]
        next_state[side] += contribution.amount
        receipt = Receipt(
            id=f"r-{side}",
            amount=contribution.amount,
            data=contribution.data,
            state_at_entry=state,
        )
        return next_state, receipt

    def on_round_end(self, state):
        return state, True

    def outcome(self, state):
        approve = state["approve"]
        reject = state["reject"]
        total = approve + reject
        decision = "approve" if approve > reject else "reject"

        def payout_fn(receipt, settlement=None):
            winner = decision if settlement is None else settlement.final_decision
            winner_pool = approve if winner == "approve" else reject
            if winner_pool <= 0:
                return 0.0
            if receipt.data["side"] != winner:
                return 0.0
            return receipt.amount * total / winner_pool

        return decision, payout_fn, False

    def valid_data(self):
        return _SideData


class SubsidizedApproveMechanism(MechanismBase):
    mechanism_id = "subsidized_approve"

    def __init__(self, winner_subsidy: float = 2.0, **params: object) -> None:
        super().__init__(**params)
        self.winner_subsidy = winner_subsidy

    def init(self):
        return {"approve": 0.0}

    def publish(self, state):
        return {"approve_stake": state["approve"], "reject_stake": 0.0, "winner_subsidy": self.winner_subsidy}

    def on_contribution(self, state, contribution):
        next_state = dict(state)
        next_state["approve"] += contribution.amount
        receipt = Receipt(
            id="subsidized",
            amount=contribution.amount,
            data=contribution.data,
            state_at_entry=state,
        )
        return next_state, receipt

    def on_round_end(self, state):
        return state, True

    def outcome(self, state):
        total = state["approve"]

        def payout_fn(receipt, settlement=None):
            del settlement
            if total <= 0:
                return 0.0
            return receipt.amount * (total + self.winner_subsidy) / total

        return "approve", payout_fn, False

    def external_funding(self, state, settlement):
        del settlement
        return self.winner_subsidy if state["approve"] > 0 else 0.0

    def valid_data(self):
        return _SideData


def _base_config(agent_specs):
    return ScenarioConfig.model_validate(
        {
            "seed": 13,
            "num_proposals": 1,
            "round_cap": 5,
            "environment": {
                "mu_W": 0.0,
                "sigma_W": 0.0,
                "phi": 0.000001,
                "alpha": 0.005,
                "fee_rate": 0.01,
                "C": 50.0,
                "tau_F": 10.0,
            },
            "mechanism": {"id": "one_round", "params": {}},
            "agents": agent_specs,
        }
    )


def test_accounting_identities_and_tie_reject() -> None:
    registry = SubmissionRegistry(
        agents={
            "fixed_approve": FixedApproveAgent,
            "fixed_reject": FixedRejectAgent,
        },
        mechanisms={"one_round": OneRoundParimutuel},
    )

    config = _base_config(
        [
            {"id": "fixed_approve", "count": 1, "params": {"amount": 0.5}},
            {"id": "fixed_reject", "count": 1, "params": {"amount": 0.5}},
        ]
    )

    report = run_simulation(config, registry=registry)
    proposal = report.per_proposal[0]

    assert proposal.final_decision == "reject"
    assert proposal.decision_pre_oracle == "reject"
    assert proposal.contribution_total == pytest.approx(1.0)
    assert proposal.payout_total == pytest.approx(1.0)
    assert proposal.mechanism_net_profit == pytest.approx(0.0)
    assert proposal.proposal_utility == pytest.approx(0.0)
    assert len(proposal.agent_reports) == 2
    assert all(agent_report.attempts for agent_report in proposal.agent_reports)
    assert all(agent_report.attempts[0].accepted for agent_report in proposal.agent_reports)

    assert report.aggregates.regret == pytest.approx(
        report.aggregates.oracle_optimal_total - report.aggregates.proposal_utility_total
    )
    expected_entry_cost = config.environment.phi * math.sqrt(proposal.y) * sum(
        row.wealth for row in report.per_agent
    )
    expected_fee_cost = config.environment.fee_rate * proposal.contribution_total
    assert sum(row.total_transfer for row in report.per_agent) == pytest.approx(
        -(expected_entry_cost + expected_fee_cost)
    )


def test_insufficient_wealth_rejects_contribution() -> None:
    registry = SubmissionRegistry(
        agents={"all_in": AllInAgent},
        mechanisms={"one_round": OneRoundParimutuel},
    )

    config = _base_config(
        [
            {"id": "all_in", "count": 1, "params": {"amount": 1_000_000.0}},
        ]
    )

    report = run_simulation(config, registry=registry)
    proposal = report.per_proposal[0]
    agent_row = report.per_agent[0]

    assert proposal.contribution_total == pytest.approx(0.0)
    assert proposal.payout_total == pytest.approx(0.0)
    assert proposal.agent_reports[0].attempts[0].rejection_reason == "insufficient_wealth"
    assert agent_row.participation_count == 0
    assert agent_row.total_utility == pytest.approx(0.0)


def test_subsidized_mechanism_can_pay_out_more_than_it_collects() -> None:
    registry = SubmissionRegistry(
        agents={"fixed_approve": FixedApproveAgent},
        mechanisms={"subsidized_approve": SubsidizedApproveMechanism},
    )

    config = _base_config(
        [
            {"id": "fixed_approve", "count": 1, "params": {"amount": 0.5}},
        ]
    )
    config = ScenarioConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "mechanism": {"id": "subsidized_approve", "params": {"winner_subsidy": 2.0}},
        }
    )

    report = run_simulation(config, registry=registry)
    proposal = report.per_proposal[0]

    assert proposal.contribution_total == pytest.approx(0.5)
    assert proposal.payout_total == pytest.approx(2.5)
    assert proposal.external_funding == pytest.approx(2.0)
    assert proposal.mechanism_net_profit == pytest.approx(-2.0)
