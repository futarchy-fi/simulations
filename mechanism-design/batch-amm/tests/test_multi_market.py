import numpy as np
import pytest

from batch_amm.multi_market import allocate_budget, prepare_panel, run_strategy


def test_panel_requires_even_honest_split():
    with pytest.raises(ValueError):
        prepare_panel(k=3, n_total=32, m=10)


def test_draws_are_nested_across_k():
    big = prepare_panel(k=1, n_total=32, m=40, seed=9)
    split = prepare_panel(k=2, n_total=32, m=40, seed=9)
    assert np.array_equal(big["v"][:, 0], split["v"][:, 0])
    assert np.array_equal(big["manip_signal"][:, 0], split["manip_signal"][:, 0])
    assert np.array_equal(
        big["honest_signals"][:, 0, :16], split["honest_signals"][:, 0, :]
    )


@pytest.mark.parametrize("strategy", ["concentrate", "uniform", "greedy"])
def test_allocations_exhaust_budget(strategy):
    panel = prepare_panel(k=4, n_total=16, m=80, seed=3)
    allocation = allocate_budget(panel, 0.15, strategy, grid_steps=10)
    np.testing.assert_allclose(allocation.sum(axis=1), 0.15, atol=1e-15)
    assert np.all(allocation >= 0.0)


def test_fixed_allocation_rules():
    panel = prepare_panel(k=4, n_total=16, m=8, seed=4)
    concentrated = allocate_budget(panel, 0.2, "concentrate")
    uniform = allocate_budget(panel, 0.2, "uniform")
    assert np.all(concentrated[:, 0] == 0.2)
    assert np.all(concentrated[:, 1:] == 0.0)
    assert np.all(uniform == 0.05)


def test_k1_strategies_are_identical():
    panel = prepare_panel(k=1, n_total=16, m=200, seed=5)
    records = [run_strategy(panel, 0.15, s, grid_steps=10)[0] for s in (
        "concentrate", "uniform", "greedy"
    )]
    means = [r["metrics"]["iw_mean_abs_distortion"]["mean"] for r in records]
    assert means[0] == means[1] == means[2]


def test_zero_budget_reproduces_honest_counterfactual():
    panel = prepare_panel(k=4, n_total=16, m=200, seed=6)
    record, raw = run_strategy(panel, 0.0, "greedy")
    assert np.array_equal(raw["iw_mean_abs_distortion"], np.zeros(panel["m"]))
    assert np.array_equal(raw["manip_cost"], np.zeros(panel["m"]))
    assert record["checks"]["conservation_maxabs"] < 1e-12


def test_metrics_are_finite_and_conserve():
    panel = prepare_panel(k=8, n_total=32, m=500, seed=7)
    record, raw = run_strategy(panel, 0.5, "greedy", grid_steps=10)
    assert all(np.isfinite(values).all() for values in raw.values())
    assert record["checks"]["budget_max_abs_error"] < 1e-12
    assert record["checks"]["conservation_maxabs"] < 1e-12
    assert record["metrics"]["iw_mean_abs_distortion"]["mean"] > 0.0
