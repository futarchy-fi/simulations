"""Domain errors for Proposal Poker simulation."""


class ProposalPokerError(Exception):
    """Base class for Proposal Poker errors."""


class DiscoveryError(ProposalPokerError):
    """Raised when discovery/loading of submissions fails."""


class DuplicateSubmissionError(DiscoveryError):
    """Raised when duplicate agent/mechanism identifiers are found."""


class InvalidSubmissionError(DiscoveryError):
    """Raised when a submission does not satisfy required interfaces."""


class SimulationError(ProposalPokerError):
    """Raised when simulation invariants are violated."""


class BudgetBalanceError(SimulationError):
    """Raised when a mechanism pays out more than contributed."""


class SybilViolationError(SimulationError):
    """Raised when equivalent receipts settle to different payouts."""
