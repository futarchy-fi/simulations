from __future__ import annotations

from pathlib import Path

import pytest

from proposal_poker.discovery import discover_submissions
from proposal_poker.errors import DuplicateSubmissionError, InvalidSubmissionError


ROOT = Path(__file__).resolve().parents[1]
BUILT_INS = ROOT / "mechanism-design/proposal-evaluation"


def test_builtin_discovery_finds_example_submissions() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])

    assert "bayesian_threshold" in registry.agents
    assert "binary_staking_market" in registry.mechanisms


def test_duplicate_ids_raise_error(tmp_path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "duplicate.py").write_text(
        """
class DuplicateAgent:
    agent_id = \"bayesian_threshold\"
    def __init__(self, **params):
        self.params = params
    def act(self, wealth, signal, y, public_history, my_past):
        return None
""",
        encoding="utf-8",
    )

    with pytest.raises(DuplicateSubmissionError):
        discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[tmp_path])


def test_invalid_submission_class_is_rejected(tmp_path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "invalid.py").write_text(
        """
class NotAnAgent:
    pass
""",
        encoding="utf-8",
    )

    with pytest.raises(InvalidSubmissionError):
        discover_submissions(repo_dirs=[], extension_dirs=[tmp_path])
