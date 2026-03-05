# Simulations

Mechanism design research and simulations for Futarchy.

## Proposal Poker Engine (v1)

This repository now includes a Python simulation engine implementing the formal model in [`mechanism-design/proposal-evaluation/MODEL.md`](mechanism-design/proposal-evaluation/MODEL.md).

It supports:
- Pluggable mechanisms and agents discovered from folders.
- Scenario-driven runs from JSON.
- Per-proposal and aggregate reporting for:
  - agent utility,
  - mechanism `net_profit`,
  - proposal utility (`x * y` for approved proposals).

## Setup

```bash
python3 -m pip install -e '.[dev]'
```

## Run A Simulation

```bash
python3 -m proposal_poker.simulate \
  --scenario examples/scenarios/basic.json \
  --output /tmp/proposal-poker-report.json
```

Optional extension submissions:

```bash
python3 -m proposal_poker.simulate \
  --scenario examples/scenarios/basic.json \
  --extensions-dir /path/to/my-submissions \
  --output /tmp/report.json
```

`--extensions-dir` can be repeated.

## Submission Convention

Each submissions root can contain:

- `agents/*.py`
- `mechanisms/*.py`

Discovery is automatic. Duplicate IDs are rejected.

### Agent Requirements

A valid agent class must expose:
- `agent_id: str`
- `act(wealth, signal, y, public_history, my_past) -> Contribution | None`

### Mechanism Requirements

A valid mechanism class must expose:
- `mechanism_id: str`
- `init()`
- `publish(state)`
- `on_contribution(state, contribution) -> (state, receipt | None)`
- `on_round_end(state) -> (state, done)`
- `outcome(state) -> (decision, payout_fn, use_futarchy)`
- `valid_data() -> pydantic BaseModel class | None`

Built-in examples live in:
- [`mechanism-design/proposal-evaluation/agents/bayesian_threshold.py`](mechanism-design/proposal-evaluation/agents/bayesian_threshold.py)
- [`mechanism-design/proposal-evaluation/mechanisms/binary_staking_market.py`](mechanism-design/proposal-evaluation/mechanisms/binary_staking_market.py)

## Scenario JSON

Top-level keys:
- `seed` (optional)
- `num_proposals` (default `500`)
- `round_cap` (default `20`)
- `stake_cap_fraction` (default `0.99`)
- `environment`
- `mechanism`
- `agents`

Reference scenario:
- [`examples/scenarios/basic.json`](examples/scenarios/basic.json)

## Output JSON

Report includes:
- `metadata`: scenario hash, seed, duration, discovered plugin IDs
- `aggregates`: proposal/approval counts, proposal utility totals, oracle-optimal benchmark, regret, mechanism net profit totals/means, utility summary stats
- `per_agent`: wealth, total/mean utility, stake, transfer, participation count
- `per_proposal`: `x`, `y`, decisions, oracle fields, contribution and payout totals, mechanism net profit, proposal utility, forced termination flag

## Test

```bash
pytest -q
```

## mechanism-design/proposal-evaluation

A framework for studying how agents with private signals can collectively evaluate proposals. Each proposal has an unobservable quality and a public importance. Agents stake money to express their beliefs, and a mechanism aggregates these into approve/reject decisions.

The model separates the **environment** (proposals, agents, signals, utilities) from the **mechanism** (rules of the game), enabling systematic search over both agent strategies and mechanism designs.

See [mechanism-design/proposal-evaluation/MODEL.md](mechanism-design/proposal-evaluation/MODEL.md) for the formal specification.
