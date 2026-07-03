from __future__ import annotations

from pathlib import Path

from proposal_poker.discovery import discover_submissions
from proposal_poker.scenario import ScenarioConfig
from proposal_poker.simulator import run_simulation


ROOT = Path(__file__).resolve().parents[1]
BUILT_INS = ROOT / "mechanism-design/proposal-evaluation"


def _config(num_proposals: int, offset: int, max_rounds: int) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "seed": 777,
            "num_proposals": num_proposals,
            "round_cap": 5,
            "deterministic_env": True,
            "proposal_offset": offset,
            "mechanism": {
                "id": "binary_staking_market",
                "params": {"max_rounds": max_rounds, "oracle_margin_threshold": 0.1},
            },
            "agents": [{"id": "bayesian_threshold", "count": 3, "params": {}}],
        }
    )


def test_env_draws_identical_across_round_counts() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])
    report_a = run_simulation(_config(8, 0, max_rounds=1), registry)
    report_b = run_simulation(_config(8, 0, max_rounds=3), registry)

    for row_a, row_b in zip(report_a.per_proposal, report_b.per_proposal):
        assert row_a.x == row_b.x
        assert row_a.y == row_b.y
        signals_a = [r.signal for r in row_a.agent_reports]
        signals_b = [r.signal for r in row_b.agent_reports]
        assert signals_a == signals_b


def test_sharded_runs_match_single_run_env() -> None:
    registry = discover_submissions(repo_dirs=[BUILT_INS], extension_dirs=[])
    full = run_simulation(_config(8, 0, max_rounds=1), registry)
    shard_0 = run_simulation(_config(4, 0, max_rounds=1), registry)
    shard_1 = run_simulation(_config(4, 4, max_rounds=1), registry)

    merged = shard_0.per_proposal + shard_1.per_proposal
    assert [row.index for row in merged] == list(range(8))
    for row_full, row_shard in zip(full.per_proposal, merged):
        assert row_full.x == row_shard.x
        assert row_full.y == row_shard.y
        # Same master seed => same wealth draws => same signal draws.
        assert [r.signal for r in row_full.agent_reports] == [
            r.signal for r in row_shard.agent_reports
        ]
