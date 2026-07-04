"""The four information & payoff structures from Galanis (2026).

State space: omega in {a, b, c, d, e, f, g, h}, where each state is a
triple (d_a, d_b, d_c) in {0, 1}^3. Common uniform prior assigns each
of the 8 states probability 1/8.

The Yes security X pays 1 if the answer to the prediction-market question
is Yes, 0 otherwise. The four structures differ in:

  (a) which signals each trader observes (the partition), and
  (b) the function X(omega) mapping states to the security's payoff.

Structures `t3s111y2`, `t3s110`, `t3s111` share the same partition:
trader i privately sees d_i (so each trader observes 1 bit). The cell of
trader i's partition is determined by his signal's realisation.

Structure `t3s111o2ye2` swaps to "each trader sees the OTHER two signals"
(2 bits each), making interactive reasoning materially harder.

Payoff rules:
  t3s111y2     X = 1 iff at least 2 of {d_a, d_b, d_c} = 1
  t3s110       X = 1 iff all three signals = 1
  t3s111       X = 1 iff all three signals = 1
  t3s111o2ye2  X = 1 iff exactly 2 of {d_a, d_b, d_c} = 1

(`t3s110` and `t3s111` differ only in the canonical "true" state shown in
the paper's figures; the game itself is identical. We keep both names for
fidelity but mark `t3s110` as an alias.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

# 8 states encoded as (d_a, d_b, d_c). The paper labels them a..h with
# state a = (1, 1, 1) and state h = (0, 0, 0). We follow that ordering.
STATES: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1),  # a
    (1, 1, 0),  # b
    (1, 0, 1),  # c
    (1, 0, 0),  # d
    (0, 1, 1),  # e
    (0, 1, 0),  # f
    (0, 0, 1),  # g
    (0, 0, 0),  # h
)

STATE_LABELS: Tuple[str, ...] = tuple("abcdefgh")


def _at_least_two_yes(state: Tuple[int, int, int]) -> int:
    return int(sum(state) >= 2)


def _all_yes(state: Tuple[int, int, int]) -> int:
    return int(all(state))


def _exactly_two_yes(state: Tuple[int, int, int]) -> int:
    return int(sum(state) == 2)


# Partition functions: given the full state, return the partition cell
# (an int) that trader i observes. Each partition is the level set of a
# small projection of `state`.
#
# `own_signal_partition[i]` returns d_i (trader i sees only their own bit).
# `others_signals_partition[i]` returns the 2-bit observation of the OTHER
# two signals.


def own_signal_partition(trader: int, state: Tuple[int, int, int]) -> int:
    return state[trader]


def others_signals_partition(
    trader: int, state: Tuple[int, int, int]
) -> int:
    other_a, other_b = (j for j in range(3) if j != trader)
    return (state[other_a] << 1) | state[other_b]


@dataclass(frozen=True)
class Structure:
    name: str
    payoff_fn: Callable[[Tuple[int, int, int]], int]
    partition_fn: Callable[[int, Tuple[int, int, int]], int]
    # number of distinct partition cells per trader (used for bookkeeping)
    cells_per_trader: int

    def x_of(self, state: Tuple[int, int, int]) -> int:
        return self.payoff_fn(state)

    def cell_of(self, trader: int, state: Tuple[int, int, int]) -> int:
        return self.partition_fn(trader, state)


STRUCTURES: dict[str, Structure] = {
    "t3s111y2": Structure(
        name="t3s111y2",
        payoff_fn=_at_least_two_yes,
        partition_fn=own_signal_partition,
        cells_per_trader=2,
    ),
    "t3s110": Structure(
        name="t3s110",
        payoff_fn=_all_yes,
        partition_fn=own_signal_partition,
        cells_per_trader=2,
    ),
    "t3s111": Structure(
        name="t3s111",
        payoff_fn=_all_yes,
        partition_fn=own_signal_partition,
        cells_per_trader=2,
    ),
    "t3s111o2ye2": Structure(
        name="t3s111o2ye2",
        payoff_fn=_exactly_two_yes,
        partition_fn=others_signals_partition,
        cells_per_trader=4,
    ),
}


__all__ = ["STATES", "STATE_LABELS", "Structure", "STRUCTURES"]
