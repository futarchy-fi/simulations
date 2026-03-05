# Simulations

Mechanism design research and simulations for Futarchy.

## Proposal Poker Engine (v1)

This repository now includes a Python simulation engine implementing the formal model in [`mechanism-design/proposal-evaluation/MODEL.md`](mechanism-design/proposal-evaluation/MODEL.md).

It supports:
- Pluggable mechanisms and agents discovered from folders.
- Scenario-driven runs from JSON.
- The formal model's two monetary frictions: a deadweight participation cost on first accepted contribution and a deadweight fee proportional to total stake.
- Mechanism-defined external funding, including public winner subsidies.
- Utility is reported as excess log-wealth relative to abstaining, so `0` means break-even versus `log(W)` and negative means worse than abstaining.
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
- `external_funding(state, settlement) -> float` (optional; defaults to `0`)
- `valid_data() -> pydantic BaseModel class | None`

Built-in examples live in:
- [`mechanism-design/proposal-evaluation/agents/bayesian_threshold.py`](mechanism-design/proposal-evaluation/agents/bayesian_threshold.py)
- [`mechanism-design/proposal-evaluation/mechanisms/binary_staking_market.py`](mechanism-design/proposal-evaluation/mechanisms/binary_staking_market.py)

## Scenario JSON

Top-level keys:
- `seed` (optional)
- `num_proposals` (default `500`)
- `round_cap` (default `20`)
- `environment`
- `mechanism`
- `agents`

There is no artificial stake cap in the simulator. Contributions are rejected only if losing that stake would make the agent's terminal wealth non-positive, which would make log utility undefined.

Reference scenario:
- [`examples/scenarios/basic.json`](examples/scenarios/basic.json)

## Output JSON

Report includes:
- `metadata`: scenario hash, seed, duration, discovered plugin IDs
- `aggregates`: proposal/approval counts, proposal utility totals, oracle-optimal benchmark, regret, mechanism net profit totals/means, utility summary stats
- `per_agent`: wealth, total/mean excess utility, stake, transfer, participation count
- `per_proposal`: `x`, `y`, decisions, oracle fields, contribution and payout totals, external funding, mechanism net profit, proposal utility, forced termination flag, and per-agent action logs

## Test

```bash
pytest -q
```

## mechanism-design/proposal-evaluation

A framework for studying how agents with private signals can collectively evaluate proposals. Each proposal has an unobservable quality and a public importance. Agents stake money to express their beliefs, and a mechanism aggregates these into approve/reject decisions.

The model separates the **environment** (proposals, agents, signals, utilities) from the **mechanism** (rules of the game), enabling systematic search over both agent strategies and mechanism designs.

See [mechanism-design/proposal-evaluation/MODEL.md](mechanism-design/proposal-evaluation/MODEL.md) for the formal specification.
