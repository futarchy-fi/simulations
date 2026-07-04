"""Sanity tests for information & payoff structures."""

import pytest

from galanis_market.structures import STATES, STRUCTURES


def test_eight_states() -> None:
    assert len(STATES) == 8
    assert len(set(STATES)) == 8


@pytest.mark.parametrize("name", ["t3s111y2", "t3s110", "t3s111", "t3s111o2ye2"])
def test_structure_registered(name: str) -> None:
    assert name in STRUCTURES
    s = STRUCTURES[name]
    assert s.name == name


def test_t3s111y2_payoff() -> None:
    s = STRUCTURES["t3s111y2"]
    # At least 2 yes -> X = 1
    expected = {(1, 1, 1): 1, (1, 1, 0): 1, (1, 0, 1): 1, (0, 1, 1): 1,
                (1, 0, 0): 0, (0, 1, 0): 0, (0, 0, 1): 0, (0, 0, 0): 0}
    for state, x in expected.items():
        assert s.x_of(state) == x


def test_t3s110_payoff_is_all_yes() -> None:
    s = STRUCTURES["t3s110"]
    for state in STATES:
        assert s.x_of(state) == int(all(state))


def test_t3s111o2ye2_payoff_exactly_two() -> None:
    s = STRUCTURES["t3s111o2ye2"]
    for state in STATES:
        assert s.x_of(state) == int(sum(state) == 2)


@pytest.mark.parametrize("name", ["t3s111y2", "t3s110", "t3s111"])
def test_own_signal_partition(name: str) -> None:
    s = STRUCTURES[name]
    # Trader i's observation must equal d_i.
    for state in STATES:
        for i in range(3):
            assert s.cell_of(i, state) == state[i]


def test_others_signals_partition_distinguishes_4_cells() -> None:
    s = STRUCTURES["t3s111o2ye2"]
    # For each trader, the partition over the 8 states must have 4 cells
    # of size 2 each (one cell per realization of the other two signals).
    for trader in range(3):
        cells_seen: dict[int, list] = {}
        for state in STATES:
            c = s.cell_of(trader, state)
            cells_seen.setdefault(c, []).append(state)
        assert len(cells_seen) == 4
        for members in cells_seen.values():
            assert len(members) == 2


def test_information_aggregation_recovers_state() -> None:
    # In every structure, the conjunction of the three traders' partition
    # cells must uniquely identify the state (otherwise X is not separable).
    for name, s in STRUCTURES.items():
        signature_to_state: dict[tuple, tuple] = {}
        for state in STATES:
            sig = tuple(s.cell_of(i, state) for i in range(3))
            if sig in signature_to_state:
                assert signature_to_state[sig] == state, (
                    f"{name}: signature {sig} maps to multiple states"
                )
            signature_to_state[sig] = state
        assert len(signature_to_state) == 8, (
            f"{name}: pooled signatures only cover {len(signature_to_state)} states"
        )
