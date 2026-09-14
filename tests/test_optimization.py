"""Phase 9 tests: the LP on handcrafted baselines (feasible/infeasible,
cash constraint satisfied, no negative decision variables), plus
integration checks on the real optimization (the real liquidity policy
headroom is genuinely positive, the demo optimization reduces re-simulated
breach probability, and the plan's cash column is internally consistent)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import load_config
from src.optimization.cash_management import (
    compare_before_after, consolidated_baseline, run_optimization, solve_optimization,
)
from src.simulation.monte_carlo import load_simulation_context, run_simulation


def _easy_baseline(horizon: int = 6, opening_cash: float = 1_000_000.0) -> dict:
    return {
        "net_cf": np.full(horizon, 50_000.0),  # comfortably cash-generative
        "cash_baseline": opening_cash + np.cumsum(np.full(horizon, 50_000.0)),
        "opening_cash": opening_cash,
        "revenue": np.full(horizon, 500_000.0),
        "ap_related_spend": np.full(horizon, 300_000.0),
        "planned_capex": np.full(horizon, 20_000.0),
    }


def _shortfall_baseline(horizon: int = 6, opening_cash: float = 500_000.0) -> dict:
    return {
        "net_cf": np.full(horizon, -100_000.0),  # burning cash every month
        "cash_baseline": opening_cash + np.cumsum(np.full(horizon, -100_000.0)),
        "opening_cash": opening_cash,
        "revenue": np.full(horizon, 500_000.0),
        "ap_related_spend": np.full(horizon, 300_000.0),
        "planned_capex": np.full(horizon, 20_000.0),
    }


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_feasible_solution_when_already_comfortable(config):
    baseline = _easy_baseline()
    result = solve_optimization(baseline, config, minimum_cash=100_000.0)
    assert result["feasible"]
    assert result["status"] == "OPTIMAL"
    assert result["total_cost"] == pytest.approx(0.0, abs=1.0)  # no levers needed


def test_cash_constraint_satisfied_under_shortfall(config):
    baseline = _shortfall_baseline()
    result = solve_optimization(baseline, config, minimum_cash=200_000.0)
    assert result["feasible"]
    assert (result["plan"]["cash_optimized"] >= 200_000.0 - 1.0).all()


def test_no_negative_decision_variables(config):
    baseline = _shortfall_baseline()
    result = solve_optimization(baseline, config, minimum_cash=200_000.0)
    plan = result["plan"]
    for col in ["borrow_balance", "defer_balance", "accel", "capex_cut"]:
        assert (plan[col] >= -1e-6).all(), f"{col} went negative"


def test_infeasible_when_impossible(config):
    baseline = _shortfall_baseline(opening_cash=100_000.0)
    # A minimum cash far beyond anything max_borrowing + the other capped
    # levers could ever bridge.
    result = solve_optimization(baseline, config, minimum_cash=50_000_000.0)
    assert not result["feasible"]
    assert result["status"] == "INFEASIBLE"


def test_plan_cash_optimized_equals_baseline_plus_adjustment(config):
    baseline = _shortfall_baseline()
    result = solve_optimization(baseline, config, minimum_cash=200_000.0)
    plan = result["plan"]
    diff = (plan["cash_optimized"] - (plan["cash_baseline"] + plan["net_adjustment"])).abs()
    assert diff.max() < 1e-6


@pytest.fixture(scope="module")
def context():
    return load_simulation_context()


def test_consolidated_baseline_sums_entities(context):
    horizon = 12
    consolidated = consolidated_baseline(context, "base", horizon)
    assert len(consolidated["net_cf"]) == horizon
    assert consolidated["opening_cash"] > 0


def test_real_policy_headroom_is_positive(context):
    """Matches the Phase 8/9 finding documented in CLAUDE.md and the
    README: the real EUR 2.75M liquidity policy is never breached, even
    under the stress scenario."""
    result = run_optimization(scenario="stress")
    assert result["real_policy_headroom"] > 0


def test_demo_optimization_reduces_breach_probability(context):
    result = run_optimization(scenario="stress")
    assert result["solution"]["feasible"]
    comparison = result["comparison"]
    assert comparison["after"]["probability_of_liquidity_breach"] < comparison["before"]["probability_of_liquidity_breach"]
    assert comparison["after"]["average_cash"] > comparison["before"]["average_cash"]


def test_compare_before_after_uses_shared_simulation(context, config):
    """compare_before_after should not re-run the simulation -- 'before'
    must be exactly the passed-in mc_result's own cash paths."""
    mc_result = run_simulation(context, scenario="stress", n_simulations=500)
    from src.optimization.cash_management import consolidated_baseline as cb, solve_optimization as so
    baseline = cb(context, "stress", 12)
    solution = so(baseline, config, minimum_cash=config["optimization"]["demo_minimum_cash"])
    comparison = compare_before_after(mc_result, config, solution["plan"], config["optimization"]["demo_minimum_cash"])
    expected_avg = float(mc_result["consolidated_cash"].mean())
    assert comparison["before"]["average_cash"] == pytest.approx(expected_avg)
