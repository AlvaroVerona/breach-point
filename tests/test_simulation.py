"""Phase 8 tests: simulation shapes, reproducibility, percentile ordering,
and scenario-direction sanity (optimistic > base > pessimistic, a
data-driven check, not just structural) on the real generated data."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.monte_carlo import (
    SCENARIO_ADJUSTMENTS, load_simulation_context, percentile_summary, run_simulation,
)


@pytest.fixture(scope="module")
def context():
    return load_simulation_context()


def test_run_simulation_output_shapes(context):
    result = run_simulation(context, n_simulations=500)
    horizon = 12
    assert result["consolidated_cash"].shape == (500, horizon)
    for entity_result in result["entity_results"].values():
        assert entity_result["cash"].shape == (500, horizon)
    assert len(result["periods"]) == horizon


def test_run_simulation_uses_requested_n_simulations(context):
    result = run_simulation(context, n_simulations=777)
    assert result["n_simulations"] == 777
    assert result["consolidated_cash"].shape[0] == 777


def test_run_simulation_reproducible(context):
    result_a = run_simulation(context, n_simulations=500, seed=42)
    result_b = run_simulation(context, n_simulations=500, seed=42)
    np.testing.assert_array_equal(result_a["consolidated_cash"], result_b["consolidated_cash"])


def test_different_seeds_give_different_paths(context):
    result_a = run_simulation(context, n_simulations=500, seed=1)
    result_b = run_simulation(context, n_simulations=500, seed=2)
    assert not np.array_equal(result_a["consolidated_cash"], result_b["consolidated_cash"])


def test_consolidated_equals_sum_of_entities(context):
    result = run_simulation(context, n_simulations=300)
    summed = sum(r["cash"] for r in result["entity_results"].values())
    np.testing.assert_allclose(result["consolidated_cash"], summed)


def test_percentile_summary_is_monotonic(context):
    result = run_simulation(context, n_simulations=1000)
    summary = result["summary"]
    for _, row in summary.iterrows():
        values = [row["p5"], row["p10"], row["p25"], row["median"], row["p75"], row["p90"], row["p95"]]
        assert values == sorted(values)


def test_unknown_scenario_raises(context):
    with pytest.raises(ValueError):
        run_simulation(context, scenario="hyperinflation")


def test_scenario_direction_is_correct(context):
    """A real, data-driven check: the optimistic scenario should produce
    higher month-12 consolidated cash than base, and pessimistic lower --
    not just structurally different, but ordered the right way."""
    base = run_simulation(context, scenario="base", n_simulations=2000, seed=42)
    optimistic = run_simulation(context, scenario="optimistic", n_simulations=2000, seed=42)
    pessimistic = run_simulation(context, scenario="pessimistic", n_simulations=2000, seed=42)

    base_final = base["consolidated_cash"][:, -1].mean()
    optimistic_final = optimistic["consolidated_cash"][:, -1].mean()
    pessimistic_final = pessimistic["consolidated_cash"][:, -1].mean()

    assert optimistic_final > base_final > pessimistic_final


def test_percentile_summary_helper_matches_numpy():
    cash_paths = np.array([[100.0, 200.0], [200.0, 300.0], [300.0, 400.0], [400.0, 500.0]])
    summary = percentile_summary(cash_paths, periods=["2026-01", "2026-02"])
    assert summary.iloc[0]["median"] == pytest.approx(250.0)
    assert summary.iloc[0]["p50"] == pytest.approx(np.percentile(cash_paths[:, 0], 50))
