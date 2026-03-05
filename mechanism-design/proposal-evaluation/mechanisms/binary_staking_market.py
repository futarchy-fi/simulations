"""Example binary staking mechanism."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel

from proposal_poker.interfaces import MechanismBase
from proposal_poker.types import Contribution, Receipt, SettlementContext


class _BinaryStakeData(BaseModel):
    side: Literal["approve", "reject"]


class BinaryStakingMarket(MechanismBase):
    """Simple pari-mutuel staking market over approve/reject."""

    mechanism_id = "binary_staking_market"

    def __init__(
        self,
        max_rounds: int = 1,
        oracle_margin_threshold: float | None = 0.10,
        **params: object,
    ) -> None:
        super().__init__(**params)
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if oracle_margin_threshold is not None and oracle_margin_threshold < 0:
            raise ValueError("oracle_margin_threshold must be >= 0")

        self.max_rounds = int(max_rounds)
        self.oracle_margin_threshold = oracle_margin_threshold

    def init(self) -> dict[str, object]:
        return {
            "round": 0,
            "seq": 0,
            "stakes": {
                "approve": 0.0,
                "reject": 0.0,
            },
        }

    def publish(self, state: dict[str, object]) -> dict[str, object]:
        stakes = state["stakes"]
        return {
            "round": state["round"],
            "approve_stake": stakes["approve"],
            "reject_stake": stakes["reject"],
            "total_stake": stakes["approve"] + stakes["reject"],
        }

    def on_contribution(
        self,
        state: dict[str, object],
        contribution: Contribution,
    ) -> tuple[dict[str, object], Receipt | None]:
        side = contribution.data.get("side")
        if side not in {"approve", "reject"}:
            return state, None

        next_state = deepcopy(state)
        next_state["stakes"][side] += contribution.amount

        receipt = Receipt(
            id=f"r-{state['seq']}",
            amount=contribution.amount,
            data=contribution.data,
            state_at_entry=state,
        )

        next_state["seq"] += 1
        return next_state, receipt

    def on_round_end(self, state: dict[str, object]) -> tuple[dict[str, object], bool]:
        next_state = deepcopy(state)
        next_state["round"] += 1
        done = next_state["round"] >= self.max_rounds
        return next_state, done

    def outcome(self, state: dict[str, object]):
        approve_stake = float(state["stakes"]["approve"])
        reject_stake = float(state["stakes"]["reject"])
        total_stake = approve_stake + reject_stake

        decision = "approve" if approve_stake > reject_stake else "reject"
        if total_stake > 0:
            margin = abs(approve_stake - reject_stake) / total_stake
        else:
            margin = 0.0

        use_futarchy = bool(
            total_stake > 0
            and self.oracle_margin_threshold is not None
            and margin < self.oracle_margin_threshold
        )

        def payout_fn(receipt: Receipt, settlement: SettlementContext | None = None) -> float:
            winner = decision if settlement is None else settlement.final_decision
            winning_pool = approve_stake if winner == "approve" else reject_stake
            if winning_pool <= 0.0:
                return 0.0

            receipt_side = receipt.data["side"]
            if receipt_side != winner:
                return 0.0

            return float(receipt.amount * total_stake / winning_pool)

        return decision, payout_fn, use_futarchy

    def valid_data(self):
        return _BinaryStakeData
