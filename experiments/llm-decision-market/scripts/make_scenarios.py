"""Generate scenario JSONs for the LLM decision-market experiment.

Arms:
  A: bayesian_threshold agents, full 150 proposals, single process.
  B: llm_market (haiku) agents, 150 proposals sharded 8 ways.
  B-sonnet: llm_market (sonnet) agents, first 30 proposals, 4 shards.

All arms share seed 777 and deterministic_env, so every arm sees identical
(x, y, wealths, signals) per global proposal index.
"""

from __future__ import annotations

import json
from pathlib import Path

SEED = 777
NUM_PROPOSALS = 150
NUM_AGENTS = 5
MAX_ROUNDS = 3
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios"


def base(num_proposals: int, offset: int, agents: list[dict]) -> dict:
    return {
        "seed": SEED,
        "num_proposals": num_proposals,
        "round_cap": MAX_ROUNDS,
        "deterministic_env": True,
        "proposal_offset": offset,
        "mechanism": {
            "id": "binary_staking_market",
            "params": {
                "max_rounds": MAX_ROUNDS,
                "oracle_margin_threshold": 0.10,
                "winner_subsidy": 10.0,
            },
        },
        "agents": agents,
    }


def llm_agents(model: str, shard_id: str, log_dir: str) -> list[dict]:
    return [
        {
            "id": "llm_market",
            "count": NUM_AGENTS,
            "params": {
                "model": model,
                "shard_id": shard_id,
                "log_dir": log_dir,
                "max_rounds": MAX_ROUNDS,
            },
        }
    ]


def shard_bounds(total: int, num_shards: int) -> list[tuple[int, int]]:
    per = (total + num_shards - 1) // num_shards
    bounds = []
    start = 0
    while start < total:
        n = min(per, total - start)
        bounds.append((start, n))
        start += n
    return bounds


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = str(ROOT / "logs")

    # Arm A
    arm_a = base(
        NUM_PROPOSALS,
        0,
        [{"id": "bayesian_threshold", "count": NUM_AGENTS, "params": {}}],
    )
    (SCENARIO_DIR / "arm_a.json").write_text(json.dumps(arm_a, indent=2))

    # Arm B: 8 shards
    for i, (offset, n) in enumerate(shard_bounds(NUM_PROPOSALS, 8)):
        shard_id = f"b{i}"
        scenario = base(n, offset, llm_agents(HAIKU, shard_id, log_dir))
        (SCENARIO_DIR / f"arm_b_shard{i}.json").write_text(json.dumps(scenario, indent=2))

    # Sonnet subsample: first 30 proposals, 4 shards
    for i, (offset, n) in enumerate(shard_bounds(30, 4)):
        shard_id = f"sonnet{i}"
        scenario = base(n, offset, llm_agents(SONNET, shard_id, log_dir))
        (SCENARIO_DIR / f"arm_b_sonnet_shard{i}.json").write_text(json.dumps(scenario, indent=2))

    # Smoke test: 5 proposals, 2 agents, 2 rounds
    smoke = base(5, 0, llm_agents(HAIKU, "smoke", log_dir))
    smoke["round_cap"] = 2
    smoke["mechanism"]["params"]["max_rounds"] = 2
    smoke["agents"][0]["count"] = 2
    smoke["agents"][0]["params"]["max_rounds"] = 2
    (SCENARIO_DIR / "smoke.json").write_text(json.dumps(smoke, indent=2))

    print(f"Wrote scenarios to {SCENARIO_DIR}")


if __name__ == "__main__":
    main()
