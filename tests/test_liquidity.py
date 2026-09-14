"""Phase 8 tests: liquidity metrics arithmetic on a handcrafted cash-path
fixture (precise), the risk-score threshold logic, and sanity checks on
the real simulation (pessimistic scenario is never a lower breach
probability than optimistic)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import load_config
from src.risk.liquidity import compute_liquidity_metrics, risk_score
from src.simulation.monte_carlo import load_simulation_context, run_simulation


def test_liquidity_metrics_on_handcrafted_paths():
    # 4 simulations x 3 months. Sim 0 never breaches; sim 1 breaches once;
    # sims 2-3 breach every month.
    cash_paths = np.array([
        [500.0, 500.0, 500.0],
        [500.0, 50.0, 500.0],
        [10.0, 10.0, 10.0],
        [5.0, 5.0, 5.0],
    ])
    summary, monthly = compute_liquidity_metrics(cash_paths, periods=["m1", "m2", "m3"], minimum_cash=100.0)

    assert summary["probability_of_liquidity_breach"] == pytest.approx(3 / 4)  # sims 1,2,3 breach at least once
    assert summary["worst_simulated_cash"] == pytest.approx(5.0)
    assert summary["best_simulated_cash"] == pytest.approx(500.0)
    assert summary["avg_months_below_threshold"] == pytest.approx((0 + 1 + 3 + 3) / 4)
    assert summary["max_months_below_threshold"] == 3
    assert list(monthly["probability_of_breach"]) == pytest.approx([0.5, 0.75, 0.5])


def test_risk_score_thresholds():
    thresholds = {"low_max_breach_prob": 0.05, "medium_max_breach_prob": 0.20, "high_max_breach_prob": 0.50}
    assert risk_score(0.0, thresholds) == "LOW"
    assert risk_score(0.05, thresholds) == "LOW"
    assert risk_score(0.10, thresholds) == "MEDIUM"
    assert risk_score(0.35, thresholds) == "HIGH"
    assert risk_score(0.90, thresholds) == "CRITICAL"


@pytest.fixture(scope="module")
def context():
    return load_simulation_context()


def test_pessimistic_breach_probability_not_below_optimistic(context):
    config = load_config()
    minimum_cash = config["liquidity"]["minimum_cash"]

    optimistic = run_simulation(context, scenario="optimistic", n_simulations=2000, seed=42)
    pessimistic = run_simulation(context, scenario="pessimistic", n_simulations=2000, seed=42)

    opt_summary, _ = compute_liquidity_metrics(optimistic["consolidated_cash"], optimistic["periods"], minimum_cash)
    pess_summary, _ = compute_liquidity_metrics(pessimistic["consolidated_cash"], pessimistic["periods"], minimum_cash)

    assert pess_summary["probability_of_liquidity_breach"] >= opt_summary["probability_of_liquidity_breach"]
    assert pess_summary["expected_minimum_cash"] <= opt_summary["expected_minimum_cash"]


def test_liquidity_summary_has_required_fields(context):
    config = load_config()
    minimum_cash = config["liquidity"]["minimum_cash"]
    result = run_simulation(context, scenario="base", n_simulations=1000)
    summary, monthly = compute_liquidity_metrics(result["consolidated_cash"], result["periods"], minimum_cash)

    required = {
        "probability_of_liquidity_breach", "expected_minimum_cash", "worst_simulated_cash",
        "avg_months_below_threshold", "max_months_below_threshold",
    }
    assert required <= set(summary.keys())
    assert len(monthly) == len(result["periods"])
