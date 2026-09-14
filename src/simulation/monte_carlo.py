"""Monte Carlo simulation of future consolidated cash balances (§33-34).

Every stochastic input is calibrated from actual historical/forecast data,
not invented:

    Revenue, Operating Expense  <- Phase 7 forecast point estimate + sigma
                                   (sigma derived from that series' own
                                   held-out validation residual std)
    COGS                        <- simulated Revenue x historical gross
                                   margin distribution (mean/std from the
                                   Income Statement)
    Customer payment timing     <- simulated DSO ~ historical DSO
                                   distribution (Phase 6)
    Supplier payment timing     <- simulated DPO ~ historical DPO
                                   distribution (Phase 6)
    Working capital             <- AR/AP implied by simulated
                                   Revenue/COGS+OpEx and simulated DSO/DPO;
                                   collections/payments are the period-over-
                                   period change, so timing risk (not just
                                   revenue risk) directly drives cash
    Other cash flows            <- historical Investing CF (CAPEX)
                                   distribution, net of a near-deterministic
                                   interest outflow (debt balance is roughly
                                   flat historically -- see CLAUDE.md)

Simplifying assumptions, documented rather than hidden: draws are
independent across metrics/entities/months (no explicit correlation
structure -- e.g. no shared macro shock), and Salaries (paid directly, no
AP lag) is folded into the same DPO-timed OpEx bucket as everything else
for simplicity. Both are real limitations, not silently-made-up numbers.

Run: python -m src.simulation.monte_carlo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.accounting.balance_sheet import build_balance_sheet
from src.accounting.cash_flow import build_cash_flow
from src.accounting.income_statement import build_income_statement, _load_processed
from src.accounting.working_capital import build_working_capital
from src.common.config import PROJECT_ROOT, load_config
from src.common.logging_config import get_logger
from src.data.ledger import DEBT_ANNUAL_RATE
from src.forecasting.evaluation import build_all_forecasts

log = get_logger("monte_carlo")

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "monte_carlo_summary.csv"
PATHS_OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "monte_carlo_consolidated_paths.npy"
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]

# Scenario assumptions (§32): applied as a shift to the simulation inputs
# before drawing any randomness, so scenarios differ in their *center*
# while the Monte Carlo still explores uncertainty *around* that center.
# Percentages/day-shifts are the spec's own illustrative examples applied
# consistently to every entity, not tuned per entity to engineer an outcome.
SCENARIO_ADJUSTMENTS = {
    "base": {"revenue_pct": 0.0, "opex_pct": 0.0, "dso_days": 0.0, "dpo_days": 0.0},
    "optimistic": {"revenue_pct": 0.10, "opex_pct": -0.03, "dso_days": -5.0, "dpo_days": 0.0},
    "pessimistic": {"revenue_pct": -0.10, "opex_pct": 0.05, "dso_days": 10.0, "dpo_days": -5.0},
}


def _forecast_arrays(forecasts: pd.DataFrame, entity_id: str, metric: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    sub = forecasts[(forecasts["entity_id"] == entity_id) & (forecasts["metric"] == metric)].sort_values("step")
    assert len(sub) == horizon, f"expected {horizon} steps for {entity_id}/{metric}, got {len(sub)}"
    point = sub["forecast"].to_numpy()
    sigma = ((sub["ci_upper"] - sub["ci_lower"]) / (2 * 1.96)).to_numpy()
    return point, sigma


def _entity_inputs(
    entity_id: str, forecasts: pd.DataFrame, income_statement: pd.DataFrame,
    working_capital: pd.DataFrame, cash_flow: pd.DataFrame, balance_sheet: pd.DataFrame, horizon: int,
    scenario: str = "base",
) -> dict:
    adj = SCENARIO_ADJUSTMENTS[scenario]
    rev_point, rev_sigma = _forecast_arrays(forecasts, entity_id, "revenue", horizon)
    rev_point = rev_point * (1 + adj["revenue_pct"])
    opex_point, opex_sigma = _forecast_arrays(forecasts, entity_id, "operating_expense", horizon)
    opex_point = opex_point * (1 + adj["opex_pct"])

    is_hist = income_statement[income_statement["entity_id"] == entity_id]
    margin = 1 - (is_hist["cogs"] / is_hist["revenue"])
    margin_mean, margin_std = float(margin.mean()), float(margin.std(ddof=1))

    wc_hist = working_capital[working_capital["entity_id"] == entity_id]
    dso_mean = float(wc_hist["dso"].mean()) + adj["dso_days"]
    dso_std = float(wc_hist["dso"].std(ddof=1))
    dpo_mean = max(1.0, float(wc_hist["dpo"].mean()) + adj["dpo_days"])
    dpo_std = float(wc_hist["dpo"].std(ddof=1))

    cf_hist = cash_flow[cash_flow["entity_id"] == entity_id]
    other_cf_mean = float(cf_hist["investing_cf"].mean())
    other_cf_std = float(cf_hist["investing_cf"].std(ddof=1))

    bs_last = balance_sheet[balance_sheet["entity_id"] == entity_id].sort_values("period").iloc[-1]
    monthly_interest = float(bs_last["debt"]) * DEBT_ANNUAL_RATE / 12

    last_period = forecasts[forecasts["entity_id"] == entity_id]["period"].min() - 1
    days_in_month = np.array([(last_period + step).days_in_month for step in range(1, horizon + 1)])

    return {
        "revenue_point": rev_point, "revenue_sigma": rev_sigma,
        "opex_point": opex_point, "opex_sigma": opex_sigma,
        "margin_mean": margin_mean, "margin_std": max(margin_std, 1e-6),
        "dso_mean": dso_mean, "dso_std": max(dso_std, 1e-6),
        "dpo_mean": dpo_mean, "dpo_std": max(dpo_std, 1e-6),
        "other_cf_mean": other_cf_mean, "other_cf_std": max(other_cf_std, 1e-6),
        "monthly_interest": monthly_interest,
        "opening_ar": float(bs_last["accounts_receivable"]), "opening_ap": float(bs_last["accounts_payable"]),
        "opening_cash": float(bs_last["cash"]),
        "days_in_month": days_in_month,
    }


def simulate_entity(rng: np.random.Generator, n_simulations: int, horizon: int, inp: dict) -> dict[str, np.ndarray]:
    shape = (n_simulations, horizon)

    revenue_sim = np.clip(rng.normal(inp["revenue_point"], inp["revenue_sigma"], size=shape), 0, None)
    opex_sim = np.clip(rng.normal(inp["opex_point"], inp["opex_sigma"], size=shape), 0, None)
    margin_sim = np.clip(rng.normal(inp["margin_mean"], inp["margin_std"], size=shape), 0.05, 0.95)
    cogs_sim = revenue_sim * margin_sim

    dso_sim = np.clip(rng.normal(inp["dso_mean"], inp["dso_std"], size=shape), 1, None)
    dpo_sim = np.clip(rng.normal(inp["dpo_mean"], inp["dpo_std"], size=shape), 1, None)
    days = inp["days_in_month"][None, :]

    ar_sim = revenue_sim * dso_sim / days
    ap_sim = (cogs_sim + opex_sim) * dpo_sim / days

    ar_prev = np.hstack([np.full((n_simulations, 1), inp["opening_ar"]), ar_sim[:, :-1]])
    ap_prev = np.hstack([np.full((n_simulations, 1), inp["opening_ap"]), ap_sim[:, :-1]])

    collections = ar_prev + revenue_sim - ar_sim
    payments = ap_prev + (cogs_sim + opex_sim) - ap_sim
    operating_cf_sim = collections - payments

    other_cf_sim = rng.normal(inp["other_cf_mean"], inp["other_cf_std"], size=shape) - inp["monthly_interest"]

    net_cf_sim = operating_cf_sim + other_cf_sim
    cash_sim = inp["opening_cash"] + np.cumsum(net_cf_sim, axis=1)

    return {
        "cash": cash_sim, "operating_cf": operating_cf_sim, "revenue": revenue_sim,
        "opex": opex_sim, "ar": ar_sim, "ap": ap_sim,
    }


def percentile_summary(cash_paths: np.ndarray, periods: list) -> pd.DataFrame:
    rows = []
    for t, period in enumerate(periods):
        col = cash_paths[:, t]
        row = {"period": period, "mean": col.mean(), "median": np.median(col)}
        for p in PERCENTILES:
            row[f"p{p}"] = np.percentile(col, p)
        rows.append(row)
    return pd.DataFrame(rows)


def load_simulation_context() -> dict:
    """Everything the simulation needs, computed once so multiple scenario
    runs (base/optimistic/pessimistic) don't each re-run the full Phase 4/6/7
    pipeline (Phase 7's evaluation in particular fits a GBR model per series)."""
    transactions, coa, fx_rates = _load_processed()
    income_statement = build_income_statement(transactions, coa, fx_rates)
    balance_sheet = build_balance_sheet(transactions, coa, fx_rates, income_statement)
    cash_flow = build_cash_flow(transactions, coa, fx_rates, balance_sheet)
    working_capital = build_working_capital(income_statement, balance_sheet)
    forecasts, _ = build_all_forecasts()
    return {
        "income_statement": income_statement, "balance_sheet": balance_sheet,
        "cash_flow": cash_flow, "working_capital": working_capital, "forecasts": forecasts,
    }


def run_simulation(
    context: dict | None = None, scenario: str = "base",
    n_simulations: int | None = None, seed: int | None = None,
) -> dict:
    if scenario not in SCENARIO_ADJUSTMENTS:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {list(SCENARIO_ADJUSTMENTS)}.")
    config = load_config()
    n_simulations = n_simulations or config["simulation"]["n_simulations"]
    seed = seed if seed is not None else config["data"]["seed"]
    horizon = config["forecast"]["horizon_months"]
    rng = np.random.default_rng(seed)

    context = context or load_simulation_context()
    income_statement, balance_sheet = context["income_statement"], context["balance_sheet"]
    cash_flow, working_capital, forecasts = context["cash_flow"], context["working_capital"], context["forecasts"]

    entities = sorted(income_statement["entity_id"].unique())
    entity_results = {}
    consolidated_cash = np.zeros((n_simulations, horizon))
    periods = None

    for entity_id in entities:
        inp = _entity_inputs(
            entity_id, forecasts, income_statement, working_capital, cash_flow, balance_sheet,
            horizon, scenario,
        )
        result = simulate_entity(rng, n_simulations, horizon, inp)
        entity_results[entity_id] = result
        consolidated_cash += result["cash"]
        if periods is None:
            last_period = forecasts[forecasts["entity_id"] == entity_id]["period"].min() - 1
            periods = [last_period + step for step in range(1, horizon + 1)]

    consolidated_summary = percentile_summary(consolidated_cash, periods)
    consolidated_summary["entity_id"] = "CONSOLIDATED"

    per_entity_summaries = []
    for entity_id, result in entity_results.items():
        s = percentile_summary(result["cash"], periods)
        s["entity_id"] = entity_id
        per_entity_summaries.append(s)

    summary = pd.concat([consolidated_summary] + per_entity_summaries, ignore_index=True)
    cols = ["entity_id", "period", "mean", "median"] + [f"p{p}" for p in PERCENTILES]
    summary = summary[cols]

    return {
        "summary": summary, "consolidated_cash": consolidated_cash, "scenario": scenario,
        "entity_results": entity_results, "periods": periods, "n_simulations": n_simulations,
    }


def main() -> None:
    context = load_simulation_context()
    base_result = None
    for scenario in SCENARIO_ADJUSTMENTS:
        result = run_simulation(context, scenario=scenario)
        if scenario == "base":
            base_result = result
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            result["summary"].to_csv(OUTPUT_PATH, index=False)
            np.save(PATHS_OUTPUT_PATH, result["consolidated_cash"])

        consolidated = result["summary"][result["summary"]["entity_id"] == "CONSOLIDATED"]
        log.info(
            "[%s] Consolidated cash: month 1 median=%.0f p5=%.0f p95=%.0f | "
            "month 12 median=%.0f p5=%.0f p95=%.0f",
            scenario, consolidated.iloc[0]["median"], consolidated.iloc[0]["p5"], consolidated.iloc[0]["p95"],
            consolidated.iloc[-1]["median"], consolidated.iloc[-1]["p5"], consolidated.iloc[-1]["p95"],
        )

    log.info(
        "Ran %d simulations x %d months x %d entities (base scenario written to %s)",
        base_result["n_simulations"], len(base_result["periods"]), len(base_result["entity_results"]),
        OUTPUT_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
