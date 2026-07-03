"""Interfaces for agent and mechanism implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel

from .types import Contribution, Receipt, SettlementContext

Decision = Literal["approve", "reject"]
PayoutFn = Callable[..., float]


class AgentBase(ABC):
    """Abstract base class for all agent strategies."""

    agent_id: ClassVar[str]

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def act(
        self,
        wealth: float,
        signal: float,
        y: float,
        public_history: list[Any],
        my_past: list[Contribution],
    ) -> Contribution | None:
        """Return a contribution for the current turn or None to skip."""


class MechanismBase(ABC):
    """Abstract base class for mechanism implementations."""

    mechanism_id: ClassVar[str]

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def init(self) -> Any:
        """Return initial mechanism state for a proposal."""

    @abstractmethod
    def publish(self, state: Any) -> Any:
        """Return public message emitted by the mechanism."""

    @abstractmethod
    def on_contribution(self, state: Any, contribution: Contribution) -> tuple[Any, Receipt | None]:
        """Apply one contribution and optionally return a receipt."""

    @abstractmethod
    def on_round_end(self, state: Any) -> tuple[Any, bool]:
        """Return updated state and whether the proposal game is done."""

    @abstractmethod
    def outcome(self, state: Any) -> tuple[Decision, PayoutFn, bool]:
        """Return provisional decision, payout function, and oracle usage flag."""

    def external_funding(self, state: Any, settlement: SettlementContext) -> float:
        """Return non-agent funds available for settlement, if any."""
        del state, settlement
        return 0.0

    @abstractmethod
    def valid_data(self) -> type[BaseModel] | None:
        """Return a pydantic model class to validate contribution payload data."""
