"""Scenario configuration models and loaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EnvironmentConfig(BaseModel):
    """Environment parameters from MODEL.md."""

    model_config = ConfigDict(extra="forbid")

    mu_W: float = 3.0
    sigma_W: float = 1.5
    phi: float = Field(default=0.01, ge=0)
    alpha: float = Field(default=0.005, gt=0)
    fee_rate: float = Field(default=0.01, ge=0)
    C: float = Field(default=50.0, ge=0)
    tau_F: float = Field(default=10.0, gt=0)


class MechanismSelection(BaseModel):
    """Mechanism selection in scenario files."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class AgentSelection(BaseModel):
    """Agent selection in scenario files."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    count: int = Field(ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    """Root scenario configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    num_proposals: int = Field(default=500, ge=1)
    round_cap: int = Field(default=20, ge=1)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    mechanism: MechanismSelection
    agents: list[AgentSelection] = Field(min_length=1)


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario JSON file."""

    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        return ScenarioConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid scenario file {scenario_path}: {exc}") from exc


def scenario_hash(config: ScenarioConfig) -> str:
    """Build a stable hash from validated scenario content."""

    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
