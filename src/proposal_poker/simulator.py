"""Simulation runtime for Proposal Poker."""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from .discovery import SubmissionRegistry, discover_submissions
from .errors import InvalidSubmissionError, SimulationError, SybilViolationError
from .metrics import build_aggregates
from .scenario import ScenarioConfig, scenario_hash
from .types import AgentReport, Contribution, ProposalReport, Receipt, SettlementContext, SimulationReport

_EPS = 1e-9


@dataclass
class _AgentRuntime:
    instance_id: str
    type_id: str
    strategy: Any
    wealth: float
    precision: float
    total_utility: float = 0.0
    total_stake: float = 0.0
    total_transfer: float = 0.0
    participation_count: int = 0


def run_simulation(
    config: ScenarioConfig,
    registry: SubmissionRegistry | None = None,
) -> SimulationReport:
    """Run a full simulation according to the formal model."""

    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)

    if registry is None:
        registry = discover_submissions([_default_submission_dir()], [])

    runtimes = _instantiate_agent_population(config, registry, rng)
    proposal_rows: list[ProposalReport] = []

    for proposal_idx in range(config.num_proposals):
        x = float(rng.normal(0.0, 1.0))
        y = float(rng.lognormal(0.0, 2.0))

        mechanism = registry.create_mechanism(config.mechanism.id, config.mechanism.params)
        state = mechanism.init()
        data_schema = mechanism.valid_data()

        receipts_by_agent: list[list[Receipt]] = [[] for _ in runtimes]
        my_past_by_agent: list[list[Contribution]] = [[] for _ in runtimes]
        stake_by_agent = np.zeros(len(runtimes), dtype=float)
        entry_cost_by_agent = np.zeros(len(runtimes), dtype=float)
        payout_by_agent = np.zeros(len(runtimes), dtype=float)

        if len(runtimes) > 0:
            signal_std = np.array([1.0 / math.sqrt(agent.precision) for agent in runtimes], dtype=float)
            signals = x + rng.normal(0.0, signal_std)
        else:
            signals = np.array([], dtype=float)

        public_history: list[Any] = []
        forced_termination = True
        receipt_counter = 0

        for _round in range(config.round_cap):
            for agent_index in rng.permutation(len(runtimes)):
                runtime = runtimes[agent_index]

                # Participation constraint from the model.
                if config.environment.phi * math.sqrt(y) >= 1.0:
                    continue

                message = mechanism.publish(state)
                public_history.append(message)

                raw_contribution = runtime.strategy.act(
                    wealth=runtime.wealth,
                    signal=float(signals[agent_index]),
                    y=y,
                    public_history=list(public_history),
                    my_past=list(my_past_by_agent[agent_index]),
                )
                if raw_contribution is None:
                    continue

                contribution = Contribution.model_validate(raw_contribution)

                entry_cost = 0.0
                if not receipts_by_agent[agent_index]:
                    entry_cost = _participation_entry_cost(
                        wealth=runtime.wealth,
                        y=y,
                        phi=config.environment.phi,
                    )

                max_allowed_loss = config.stake_cap_fraction * runtime.wealth
                potential_stake = stake_by_agent[agent_index] + contribution.amount
                potential_loss = (
                    potential_stake
                    + _stake_fee_cost(potential_stake, config.environment.fee_rate)
                    + entry_cost_by_agent[agent_index]
                )
                if not receipts_by_agent[agent_index]:
                    potential_loss += entry_cost

                if potential_loss > max_allowed_loss:
                    continue

                contribution = contribution.model_copy(
                    update={"data": _validate_contribution_data(data_schema, contribution.data)}
                )

                state_at_entry = copy.deepcopy(state)
                state, raw_receipt = mechanism.on_contribution(state, contribution)
                if raw_receipt is None:
                    continue

                receipt = _normalize_receipt(
                    raw_receipt=raw_receipt,
                    contribution=contribution,
                    state_at_entry=state_at_entry,
                    default_id=f"{proposal_idx}:{receipt_counter}",
                )
                receipt_counter += 1

                receipts_by_agent[agent_index].append(receipt)
                my_past_by_agent[agent_index].append(contribution)
                stake_by_agent[agent_index] += contribution.amount
                if entry_cost_by_agent[agent_index] == 0.0:
                    entry_cost_by_agent[agent_index] = entry_cost

            state, done = mechanism.on_round_end(state)
            if done:
                forced_termination = False
                break

        decision_pre_oracle, payout_fn, use_futarchy = mechanism.outcome(state)
        decision_pre_oracle = _normalize_decision(decision_pre_oracle)

        oracle_signal: float | None = None
        final_decision = decision_pre_oracle
        if use_futarchy:
            oracle_signal = float(x + rng.normal(0.0, 1.0 / math.sqrt(config.environment.tau_F)))
            final_decision = "approve" if oracle_signal > 0.0 else "reject"

        settlement = SettlementContext(
            final_decision=final_decision,
            oracle_used=bool(use_futarchy),
            oracle_signal=oracle_signal,
        )

        all_settled_receipts: list[tuple[Receipt, float]] = []
        for agent_index in range(len(runtimes)):
            for receipt in receipts_by_agent[agent_index]:
                payout = _compute_payout(payout_fn, receipt, settlement)
                payout_by_agent[agent_index] += payout
                all_settled_receipts.append((receipt, payout))

        contribution_total = float(np.sum(stake_by_agent))
        payout_total = float(np.sum(payout_by_agent))
        external_funding = float(mechanism.external_funding(state, settlement))
        if not math.isfinite(external_funding) or external_funding < 0.0:
            raise SimulationError("Mechanism external funding must be a non-negative finite float")

        _enforce_sybil_invariant(all_settled_receipts)

        oracle_cost = config.environment.C if use_futarchy else 0.0
        mechanism_net_profit = contribution_total - payout_total - oracle_cost
        proposal_utility = x * y if final_decision == "approve" else 0.0
        oracle_optimal_value = x * y if x > 0.0 else 0.0

        for agent_index, runtime in enumerate(runtimes):
            stake = float(stake_by_agent[agent_index])
            entry_cost = float(entry_cost_by_agent[agent_index])
            fee_cost = _stake_fee_cost(stake, config.environment.fee_rate)
            transfer = float(payout_by_agent[agent_index] - stake - entry_cost - fee_cost)
            terminal_wealth = runtime.wealth + transfer
            if terminal_wealth <= 0.0:
                raise SimulationError(
                    f"Non-positive terminal wealth for {runtime.instance_id} on proposal {proposal_idx}"
                )
            utility = math.log(terminal_wealth)

            runtime.total_utility += utility
            runtime.total_stake += stake
            runtime.total_transfer += transfer
            if stake > 0.0:
                runtime.participation_count += 1

        proposal_rows.append(
            ProposalReport(
                index=proposal_idx,
                x=x,
                y=y,
                decision_pre_oracle=decision_pre_oracle,
                final_decision=final_decision,
                use_futarchy=bool(use_futarchy),
                oracle_signal=oracle_signal,
                contribution_total=contribution_total,
                payout_total=payout_total,
                external_funding=external_funding,
                mechanism_net_profit=mechanism_net_profit,
                proposal_utility=proposal_utility,
                oracle_optimal_value=oracle_optimal_value,
                forced_termination=forced_termination,
            )
        )

    agent_rows = [
        AgentReport(
            agent_instance_id=runtime.instance_id,
            agent_type_id=runtime.type_id,
            wealth=runtime.wealth,
            total_utility=runtime.total_utility,
            mean_utility=runtime.total_utility / config.num_proposals,
            total_stake=runtime.total_stake,
            total_transfer=runtime.total_transfer,
            participation_count=runtime.participation_count,
        )
        for runtime in runtimes
    ]

    report = SimulationReport(
        metadata={
            "scenario_hash": scenario_hash(config),
            "seed": config.seed,
            "duration_seconds": time.perf_counter() - started,
            "discovered_agents": sorted(registry.agents.keys()),
            "discovered_mechanisms": sorted(registry.mechanisms.keys()),
        },
        aggregates=build_aggregates(proposal_rows, agent_rows),
        per_agent=agent_rows,
        per_proposal=proposal_rows,
    )
    return report


def _instantiate_agent_population(
    config: ScenarioConfig,
    registry: SubmissionRegistry,
    rng: np.random.Generator,
) -> list[_AgentRuntime]:
    runtimes: list[_AgentRuntime] = []
    counters: dict[str, int] = defaultdict(int)

    for agent_config in config.agents:
        for _ in range(agent_config.count):
            strategy = registry.create_agent(agent_config.id, agent_config.params)
            wealth = float(np.exp(rng.normal(config.environment.mu_W, config.environment.sigma_W)))
            precision = (config.environment.phi / config.environment.alpha) * wealth
            if precision <= 0:
                raise SimulationError(f"Agent precision must be positive for {agent_config.id}")

            next_index = counters[agent_config.id]
            counters[agent_config.id] += 1
            runtimes.append(
                _AgentRuntime(
                    instance_id=f"{agent_config.id}#{next_index}",
                    type_id=agent_config.id,
                    strategy=strategy,
                    wealth=wealth,
                    precision=precision,
                )
            )

    return runtimes


def _validate_contribution_data(schema: type[BaseModel] | None, data: Any) -> Any:
    if schema is None:
        return data

    if not inspect.isclass(schema) or not issubclass(schema, BaseModel):
        raise InvalidSubmissionError("Mechanism.valid_data() must return a pydantic BaseModel class or None")

    try:
        validated = schema.model_validate(data)
    except ValidationError as exc:
        raise SimulationError(f"Contribution data failed schema validation: {exc}") from exc
    return validated.model_dump(mode="python")


def _participation_entry_cost(wealth: float, y: float, phi: float) -> float:
    return phi * wealth * math.sqrt(y)


def _stake_fee_cost(stake: float, fee_rate: float) -> float:
    return fee_rate * stake


def _normalize_receipt(
    raw_receipt: Receipt | dict[str, Any],
    contribution: Contribution,
    state_at_entry: Any,
    default_id: str,
) -> Receipt:
    if isinstance(raw_receipt, Receipt):
        receipt = raw_receipt
    elif isinstance(raw_receipt, dict):
        payload = dict(raw_receipt)
        payload.setdefault("id", default_id)
        payload.setdefault("amount", contribution.amount)
        payload.setdefault("data", contribution.data)
        payload.setdefault("state_at_entry", state_at_entry)
        receipt = Receipt.model_validate(payload)
    else:
        raise SimulationError("Mechanism on_contribution must return Receipt | dict | None")

    if abs(receipt.amount - contribution.amount) > _EPS:
        raise SimulationError("Receipt amount must equal contribution amount")
    if receipt.data != contribution.data:
        raise SimulationError("Receipt data must equal contribution data")

    return receipt.model_copy(
        update={
            "amount": contribution.amount,
            "data": contribution.data,
            "state_at_entry": state_at_entry,
        }
    )


def _normalize_decision(value: Any) -> str:
    if value not in {"approve", "reject"}:
        raise SimulationError(f"Invalid mechanism decision: {value}")
    return value


def _compute_payout(payout_fn: Any, receipt: Receipt, settlement: SettlementContext) -> float:
    if _accepts_settlement_arg(payout_fn):
        payout = payout_fn(receipt, settlement)
    else:
        payout = payout_fn(receipt)

    payout_value = float(payout)
    if not math.isfinite(payout_value):
        raise SimulationError("Non-finite payout returned by mechanism")
    if payout_value < -_EPS:
        raise SimulationError("Negative payout returned by mechanism")

    return max(0.0, payout_value)


def _accepts_settlement_arg(fn: Any) -> bool:
    signature = inspect.signature(fn)
    has_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values())
    if has_var_args:
        return True

    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _enforce_sybil_invariant(settled_receipts: list[tuple[Receipt, float]]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for receipt, payout in settled_receipts:
        key_payload = [receipt.amount, receipt.data, receipt.state_at_entry]
        key = json.dumps(key_payload, sort_keys=True, default=_json_default, separators=(",", ":"))
        groups[key].append(payout)

    for payouts in groups.values():
        if not payouts:
            continue
        if max(payouts) - min(payouts) > _EPS:
            raise SybilViolationError("Equivalent receipts settled to different payouts")


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def _default_submission_dir() -> Path:
    cwd_candidate = Path.cwd() / "mechanism-design/proposal-evaluation"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    package_root = Path(__file__).resolve().parents[2]
    package_candidate = package_root / "mechanism-design/proposal-evaluation"
    if package_candidate.exists():
        return package_candidate

    raise InvalidSubmissionError(
        "Could not find default submissions directory 'mechanism-design/proposal-evaluation'"
    )
