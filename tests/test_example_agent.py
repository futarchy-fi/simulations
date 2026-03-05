from __future__ import annotations

from pathlib import Path

from proposal_poker.discovery import discover_submissions


ROOT = Path(__file__).resolve().parents[1]
BUILT_INS = ROOT / "mechanism-design/proposal-evaluation"


def test_bayesian_threshold_agent_uses_costs_in_trade_decision() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])

    costly_agent = registry.create_agent(
        "bayesian_threshold",
        {
            "min_stake": 1.0,
            "max_stake": 1.0,
            "confidence_threshold": 0.5,
            "precision_ratio": 2.0,
            "avoid_likely_minority": False,
            "phi": 0.01,
            "fee_rate": 0.01,
        },
    )
    free_agent = registry.create_agent(
        "bayesian_threshold",
        {
            "min_stake": 1.0,
            "max_stake": 1.0,
            "confidence_threshold": 0.5,
            "precision_ratio": 2.0,
            "avoid_likely_minority": False,
            "phi": 0.0,
            "fee_rate": 0.0,
        },
    )

    assert costly_agent.act(
        wealth=10.0,
        signal=0.2,
        y=25.0,
        public_history=[
            {
                "approve_stake": 0.0,
                "reject_stake": 0.8,
                "total_stake": 0.8,
            }
        ],
        my_past=[],
    ) is None

    contribution = free_agent.act(
        wealth=10.0,
        signal=0.2,
        y=25.0,
        public_history=[
            {
                "approve_stake": 0.0,
                "reject_stake": 0.8,
                "total_stake": 0.8,
            }
        ],
        my_past=[],
    )
    assert contribution is not None
    assert contribution.data["side"] == "approve"
    assert contribution.amount == 1.0


def test_bayesian_threshold_agent_uses_public_subsidy_in_trade_decision() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])
    agent = registry.create_agent(
        "bayesian_threshold",
        {
            "min_stake": 1.0,
            "max_stake": 1.0,
            "confidence_threshold": 0.5,
            "precision_ratio": 2.0,
            "avoid_likely_minority": False,
            "phi": 0.01,
            "fee_rate": 0.01,
        },
    )

    contribution = agent.act(
        wealth=10.0,
        signal=0.2,
        y=25.0,
        public_history=[
            {
                "approve_stake": 0.0,
                "reject_stake": 0.8,
                "total_stake": 0.8,
                "winner_subsidy": 20.0,
            }
        ],
        my_past=[],
    )

    assert contribution is not None
    assert contribution.data["side"] == "approve"
    assert contribution.amount == 1.0
