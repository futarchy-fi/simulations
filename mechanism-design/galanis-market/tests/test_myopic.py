"""Sanity tests for the myopic Bayesian benchmark."""

import pytest

from galanis_market.myopic import myopic_final_prices, myopic_trajectory
from galanis_market.structures import STATES, STATE_LABELS, STRUCTURES


@pytest.mark.parametrize("name", ["t3s111y2", "t3s110", "t3s111"])
def test_myopic_aggregates_perfectly_in_easy_structures(name: str) -> None:
    """In each "easy" structure, three rounds of myopic play should
    pin the final price to X(omega) exactly, because the partition is
    one-bit per trader and Ostrovsky's separable-security result kicks
    in immediately.
    """
    s = STRUCTURES[name]
    finals = myopic_final_prices(s, num_rounds=3)
    for omega_idx, label in enumerate(STATE_LABELS):
        truth = s.x_of(STATES[omega_idx])
        assert finals[label] == pytest.approx(float(truth), abs=1e-6), (
            f"{name}, omega={label}: expected {truth}, got {finals[label]}"
        )


def test_myopic_trajectory_length() -> None:
    s = STRUCTURES["t3s111y2"]
    traj = myopic_trajectory(s, omega_idx=0, num_rounds=3)
    assert len(traj) == 4  # initial + 3 rounds
    assert traj[0] == 0.5


def test_t3s111o2ye2_first_round_posterior() -> None:
    # In the Very-Hard structure trader 0 sees (d_b, d_c). At omega = a
    # = (1,1,1), trader 0 observes cell (1,1). Under uniform prior, what
    # is E[X | (d_b, d_c) = (1, 1), payoff exactly 2 yes]?
    #
    # The two states consistent with trader 0's observation are
    # a = (1,1,1) [exactly 3 yes, X=0] and e = (0,1,1) [exactly 2 yes, X=1].
    # Uniform prior over these two -> E[X] = 0.5.
    s = STRUCTURES["t3s111o2ye2"]
    traj = myopic_trajectory(s, omega_idx=0, num_rounds=1)
    assert traj[1] == pytest.approx(0.5)
