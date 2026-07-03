from __future__ import annotations

import math
from pathlib import Path

import pytest

from proposal_poker.discovery import discover_submissions
from proposal_poker.scenario import ScenarioConfig
from proposal_poker.simulator import run_simulation


ROOT = Path(__file__).resolve().parents[1]
BUILT_INS = ROOT / "mechanism-design/proposal-evaluation"


def _scenario(seed: int) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "seed": seed,
            "num_proposals": 40,
            "round_cap": 20,
            "environment": {
                "mu_W": 3.0,
                "sigma_W": 1.5,
                "phi": 0.01,
                "alpha": 0.005,
                "fee_rate": 0.01,
                "C": 50.0,
                "tau_F": 10.0,
            },
            "mechanism": {
                "id": "binary_staking_market",
                "params": {
                    "max_rounds": 1,
                    "oracle_margin_threshold": 0.1,
                },
            },
            "agents": [
                {
                    "id": "bayesian_threshold",
                    "count": 8,
                    "params": {
                        "min_stake": 0.5,
                        "max_stake": 3.0,
                        "confidence_threshold": 0.55,
                        "precision_ratio": 2.0,
                    },
                }
            ],
        }
    )


def test_stochastic_invariants_hold_across_seeds() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])

    for seed in range(15):
        config = _scenario(seed)
        report = run_simulation(config, registry=registry)

        for proposal in report.per_proposal:
            assert proposal.payout_total <= proposal.contribution_total + proposal.external_funding + 1e-9
            expected_profit = (
                proposal.contribution_total
                - proposal.payout_total
                - (config.environment.C if proposal.use_futarchy else 0.0)
            )
            assert proposal.mechanism_net_profit == pytest.approx(expected_profit)

            expected_utility = proposal.x * proposal.y if proposal.final_decision == "approve" else 0.0
            assert proposal.proposal_utility == pytest.approx(expected_utility)

        assert report.aggregates.regret == pytest.approx(
            report.aggregates.oracle_optimal_total - report.aggregates.proposal_utility_total
        )

        for agent in report.per_agent:
            assert math.isfinite(agent.total_utility)
            assert agent.participation_count >= 0
