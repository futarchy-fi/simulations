"""Core data models for Proposal Poker."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["approve", "reject"]


class Contribution(BaseModel):
    """Money and payload sent to a mechanism."""

    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0)
    data: Any


class Receipt(BaseModel):
    """Receipt issued by a mechanism for accepted contributions."""

    model_config = ConfigDict(extra="forbid")

    id: str
    amount: float = Field(gt=0)
    data: Any
    state_at_entry: Any


class SettlementContext(BaseModel):
    """Context exposed to payout functions at settlement time."""

    model_config = ConfigDict(extra="forbid")

    final_decision: Decision
    oracle_used: bool
    oracle_signal: float | None = None


class Metadata(BaseModel):
    """Metadata associated with a simulation run."""

    model_config = ConfigDict(extra="forbid")

    scenario_hash: str
    seed: int | None
    duration_seconds: float
    discovered_agents: list[str]
    discovered_mechanisms: list[str]


class AggregateMetrics(BaseModel):
    """Top-level aggregate report values."""

    model_config = ConfigDict(extra="forbid")

    proposal_count: int
    approval_count: int
    futarchy_count: int
    proposal_utility_total: float
    oracle_optimal_total: float
    regret: float
    mechanism_net_profit_total: float
    mechanism_net_profit_mean: float
    utility_mean: float
    utility_min: float
    utility_max: float
    utility_std: float


class AgentReport(BaseModel):
    """Per-agent aggregated outcome."""

    model_config = ConfigDict(extra="forbid")

    agent_instance_id: str
    agent_type_id: str
    wealth: float
    total_utility: float
    mean_utility: float
    total_stake: float
    total_transfer: float
    participation_count: int


class ProposalReport(BaseModel):
    """Per-proposal outcome and accounting."""

    model_config = ConfigDict(extra="forbid")

    index: int
    x: float
    y: float
    decision_pre_oracle: Decision
    final_decision: Decision
    use_futarchy: bool
    oracle_signal: float | None
    contribution_total: float
    payout_total: float
    external_funding: float
    mechanism_net_profit: float
    proposal_utility: float
    oracle_optimal_value: float
    forced_termination: bool


class SimulationReport(BaseModel):
    """Complete simulation output payload."""

    model_config = ConfigDict(extra="forbid")

    metadata: Metadata
    aggregates: AggregateMetrics
    per_agent: list[AgentReport]
    per_proposal: list[ProposalReport]
