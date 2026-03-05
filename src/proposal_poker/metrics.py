"""Aggregate metric helpers."""

from __future__ import annotations

import numpy as np

from .types import AgentReport, AggregateMetrics, ProposalReport


def build_aggregates(per_proposal: list[ProposalReport], per_agent: list[AgentReport]) -> AggregateMetrics:
    """Build aggregate metrics from per-proposal and per-agent rows."""

    proposal_count = len(per_proposal)
    approval_count = sum(1 for row in per_proposal if row.final_decision == "approve")
    futarchy_count = sum(1 for row in per_proposal if row.use_futarchy)

    proposal_utility_total = float(sum(row.proposal_utility for row in per_proposal))
    oracle_optimal_total = float(sum(row.oracle_optimal_value for row in per_proposal))
    regret = oracle_optimal_total - proposal_utility_total

    mechanism_net_profit_total = float(sum(row.mechanism_net_profit for row in per_proposal))
    mechanism_net_profit_mean = mechanism_net_profit_total / proposal_count

    utility_values = np.array([row.total_utility for row in per_agent], dtype=float)
    utility_mean = float(np.mean(utility_values))
    utility_min = float(np.min(utility_values))
    utility_max = float(np.max(utility_values))
    utility_std = float(np.std(utility_values))

    return AggregateMetrics(
        proposal_count=proposal_count,
        approval_count=approval_count,
        futarchy_count=futarchy_count,
        proposal_utility_total=proposal_utility_total,
        oracle_optimal_total=oracle_optimal_total,
        regret=regret,
        mechanism_net_profit_total=mechanism_net_profit_total,
        mechanism_net_profit_mean=mechanism_net_profit_mean,
        utility_mean=utility_mean,
        utility_min=utility_min,
        utility_max=utility_max,
        utility_std=utility_std,
    )
