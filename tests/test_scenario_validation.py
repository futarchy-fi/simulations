from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from proposal_poker.scenario import ScenarioConfig, load_scenario


def test_scenario_defaults_are_applied() -> None:
    config = ScenarioConfig.model_validate(
        {
            "mechanism": {"id": "binary_staking_market", "params": {}},
            "agents": [{"id": "bayesian_threshold", "count": 1, "params": {}}],
        }
    )

    assert config.num_proposals == 500
    assert config.round_cap == 20
    assert config.stake_cap_fraction == pytest.approx(0.99)
    assert config.environment.mu_W == pytest.approx(3.0)


def test_invalid_agent_count_fails_validation() -> None:
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(
            {
                "mechanism": {"id": "binary_staking_market", "params": {}},
                "agents": [{"id": "bayesian_threshold", "count": 0, "params": {}}],
            }
        )


def test_load_scenario_raises_value_error_on_invalid_payload(tmp_path) -> None:
    scenario_path = tmp_path / "bad.json"
    scenario_path.write_text(
        json.dumps(
            {
                "num_proposals": -1,
                "mechanism": {"id": "m", "params": {}},
                "agents": [{"id": "a", "count": 1, "params": {}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_scenario(scenario_path)
