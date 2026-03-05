from __future__ import annotations

from pathlib import Path

import pytest

from proposal_poker.discovery import discover_submissions
from proposal_poker.scenario import ScenarioConfig
from proposal_poker.simulator import run_simulation


ROOT = Path(__file__).resolve().parents[1]
BUILT_INS = ROOT / "mechanism-design/proposal-evaluation"


def test_seeded_regression_snapshot() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])
    config = ScenarioConfig.model_validate(
        {
            "seed": 12345,
            "num_proposals": 25,
            "round_cap": 20,
            "stake_cap_fraction": 0.99,
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
                    "count": 6,
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

    report = run_simulation(config, registry=registry)
    agg = report.aggregates

    assert report.metadata.scenario_hash == "3888c4ea6dc463e82482d1b5b51041a735eb1a62fe4a2a6ed92493a579a95d34"

    assert agg.approval_count == 0
    assert agg.futarchy_count == 0
    assert agg.proposal_utility_total == pytest.approx(0.0)
    assert agg.oracle_optimal_total == pytest.approx(51.58878936524919)
    assert agg.regret == pytest.approx(51.58878936524919)
    assert agg.mechanism_net_profit_total == pytest.approx(0.0, abs=1e-12)
    assert agg.mechanism_net_profit_mean == pytest.approx(0.0, abs=1e-12)
    assert agg.utility_mean == pytest.approx(0.0)
    assert agg.utility_min == pytest.approx(0.0)
    assert agg.utility_max == pytest.approx(0.0)
    assert agg.utility_std == pytest.approx(0.0)

    assert [row.final_decision for row in report.per_proposal[:3]] == ["reject", "reject", "reject"]
    assert [row.mechanism_net_profit for row in report.per_proposal[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [row.proposal_utility for row in report.per_proposal[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [row.contribution_total for row in report.per_proposal[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [row.agent_reports[0].attempts[0].rejection_reason for row in report.per_proposal[:3]] == [
        "abstain",
        "abstain",
        "abstain",
    ]
