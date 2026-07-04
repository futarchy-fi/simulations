"""Closed-form myopic Bayesian benchmark for the Galanis market.

Under the "everyone is myopic and that is common knowledge" assumption,
each trader best-responds to the current public price + their own
partition cell by moving the price to their posterior on X.

This module computes that benchmark trajectory recursively. It serves
as a sanity check against CFR+ output: for the three easy structures
(t3s111y2, t3s110, t3s111) the myopic equilibrium aggregates X
perfectly within three rounds, while for t3s111o2ye2 it can differ.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from galanis_market.structures import STATES, STATE_LABELS, Structure


def _posterior_x_given_history(
    structure: Structure,
    trader: int,
    cell: int,
    posterior_belief: Dict[int, float],
) -> float:
    """E[X | trader's cell, public belief over omega]."""
    # Restrict to states consistent with trader's observation.
    weights = {
        omega: prob
        for omega, prob in posterior_belief.items()
        if structure.cell_of(trader, STATES[omega]) == cell
    }
    z = sum(weights.values())
    if z <= 0.0:
        return 0.5  # off-equilibrium fallback
    expected = sum(
        structure.x_of(STATES[omega]) * prob / z
        for omega, prob in weights.items()
    )
    return float(expected)


def myopic_trajectory(
    structure: Structure,
    omega_idx: int,
    num_rounds: int = 3,
    initial_belief: Optional[Dict[int, float]] = None,
) -> List[float]:
    """Compute the public-price trajectory under fully-Bayesian myopic play.

    Assumes traders rotate in fixed order (0, 1, 2, 0, ...) and each
    trader, on each turn, observes (a) the current public belief over
    omega derivable from price history and structure, and (b) their own
    partition cell, then moves the price to their conditional expectation
    of X.

    The public belief update upon seeing trader i set price p_t is:

        Pr(omega | p_t) ∝ Pr(omega) * 1{cell_i(omega) is consistent}

    where the consistency check is: the trader's reported posterior p_t,
    given cell c, must equal posterior_x_given_history(structure, i, c, prior),
    so the public can back out which cell c the trader was in.

    Returns the price sequence [p_initial, p_1, p_2, ..., p_num_rounds].
    """
    if initial_belief is None:
        initial_belief = {i: 1.0 / len(STATES) for i in range(len(STATES))}
    belief = dict(initial_belief)

    prices: List[float] = [0.5]
    cells_for_traders = [structure.cell_of(t, STATES[omega_idx]) for t in range(3)]

    # Pre-compute for each trader, a map from cell -> posterior under the
    # CURRENT belief. The public uses this map (parametric in belief) to
    # invert observed price into cell.
    for r in range(num_rounds):
        trader = r % 3
        cell = cells_for_traders[trader]
        # Trader's myopic price = their posterior on X.
        p_t = _posterior_x_given_history(structure, trader, cell, belief)
        prices.append(p_t)

        # Public update: which cells of trader could have produced p_t?
        # In an equilibrium with strict separability the mapping cell -> price
        # is invertible; otherwise multiple cells could match the same p_t.
        cell_to_price: Dict[int, float] = {}
        for c in range(structure.cells_per_trader):
            cell_to_price[c] = _posterior_x_given_history(
                structure, trader, c, belief
            )
        # Identify consistent cells (within numerical tolerance).
        consistent_cells = [
            c for c, p in cell_to_price.items() if abs(p - p_t) < 1e-9
        ]
        if not consistent_cells:
            consistent_cells = [cell]  # fall back to truth
        # Update belief: keep omegas where trader's cell is in consistent set.
        new_belief: Dict[int, float] = {}
        for omega, prob in belief.items():
            c = structure.cell_of(trader, STATES[omega])
            if c in consistent_cells:
                new_belief[omega] = prob
        # Renormalise.
        z = sum(new_belief.values())
        if z <= 0:
            new_belief = belief
        else:
            new_belief = {k: v / z for k, v in new_belief.items()}
        belief = new_belief

    return prices


def myopic_final_prices(
    structure: Structure, num_rounds: int = 3
) -> Dict[str, float]:
    """Return final myopic-equilibrium price for each of the 8 omegas."""
    out: Dict[str, float] = {}
    for omega_idx, label in enumerate(STATE_LABELS):
        traj = myopic_trajectory(structure, omega_idx, num_rounds=num_rounds)
        out[label] = traj[-1]
    return out


__all__ = ["myopic_trajectory", "myopic_final_prices"]
