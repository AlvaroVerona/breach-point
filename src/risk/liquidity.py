"""Liquidity risk metrics from the Monte Carlo cash simulation (§35-36):
probability of breaching the minimum cash requirement, expected minimum
cash, worst simulated outcome, months spent below threshold, and a
transparent LOW/MEDIUM/HIGH/CRITICAL score with documented thresholds
(config/settings.yaml: liquidity.risk_thresholds).

Run: python -m src.risk.liquidity
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT, load_config
from src.common.logging_config import get_logger
from src.simulation.monte_carlo import SCENARIO_ADJUSTMENTS, load_simulation_context, run_simulation

log = get_logger("liquidity")

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "liquidity_risk.json"
MONTHLY_OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "liquidity_risk_by_month.csv"


def compute_liquidity_metrics(
    cash_paths: np.ndarray, periods: list, minimum_cash: float,
) -> tuple[dict, pd.DataFrame]:
    breach_mask = cash_paths < minimum_cash
    breach_prob_by_month = breach_mask.mean(axis=0)
    monthly = pd.DataFrame({
        "period": periods, "probability_of_breach": breach_prob_by_month,
        "median_cash": np.median(cash_paths, axis=0), "p5_cash": np.percentile(cash_paths, 5, axis=0),
    })

    ever_breach = breach_mask.any(axis=1)
    months_below = breach_mask.sum(axis=1)
    min_cash_per_sim = cash_paths.min(axis=1)

    summary = {
        "minimum_cash_requirement": minimum_cash,
        "probability_of_liquidity_breach": float(ever_breach.mean()),
        "expected_minimum_cash": float(min_cash_per_sim.mean()),
        "worst_simulated_cash": float(cash_paths.min()),
        "best_simulated_cash": float(cash_paths.max()),
        "avg_months_below_threshold": float(months_below.mean()),
        "max_months_below_threshold": int(months_below.max()),
        "pct_simulations_ever_breaching": float(ever_breach.mean() * 100),
    }
    return summary, monthly


def risk_score(probability_of_breach: float, thresholds: dict) -> str:
    if probability_of_breach <= thresholds["low_max_breach_prob"]:
        return "LOW"
    if probability_of_breach <= thresholds["medium_max_breach_prob"]:
        return "MEDIUM"
    if probability_of_breach <= thresholds["high_max_breach_prob"]:
        return "HIGH"
    return "CRITICAL"


def main() -> None:
    config = load_config()
    minimum_cash = config["liquidity"]["minimum_cash"]
    thresholds = config["liquidity"]["risk_thresholds"]

    context = load_simulation_context()
    all_summaries = {}
    base_monthly = None

    for scenario in SCENARIO_ADJUSTMENTS:
        result = run_simulation(context, scenario=scenario)
        summary, monthly = compute_liquidity_metrics(result["consolidated_cash"], result["periods"], minimum_cash)
        summary["risk_score"] = risk_score(summary["probability_of_liquidity_breach"], thresholds)
        all_summaries[scenario] = summary
        if scenario == "base":
            base_monthly = monthly
        log.info(
            "[%s] Liquidity Risk: %s (P(breach)=%.1f%%, expected min cash=%.0f, worst case=%.0f, "
            "avg months below threshold=%.1f/%d)",
            scenario, summary["risk_score"], summary["probability_of_liquidity_breach"] * 100,
            summary["expected_minimum_cash"], summary["worst_simulated_cash"],
            summary["avg_months_below_threshold"], len(result["periods"]),
        )

    all_summaries["risk_thresholds"] = thresholds
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    base_monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)

    log.info("Wrote liquidity risk summary (base/optimistic/pessimistic) to %s", OUTPUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
