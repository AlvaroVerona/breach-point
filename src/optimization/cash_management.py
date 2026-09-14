"""Cash management optimization (§37-40): given a deterministic baseline
cash trajectory, choose the lowest-cost combination of short-term
borrowing, collections acceleration (factoring), payment deferral and
CAPEX reduction that keeps consolidated cash at or above the minimum cash
requirement in every month.

Model (continuous LP, OR-Tools GLOP):

    Decision variables, per month t:
        borrow_t   >= 0   outstanding revolving-credit balance (a stock)
        defer_t    >= 0   outstanding deferred-payment balance (a stock)
        accel_t    >= 0   receivables factored this month (a one-time flow)
        capex_cut_t>= 0   planned CAPEX skipped this month (a one-time flow)

    Cash_t = Cash_{t-1} + BaselineNetCF_t
             + (borrow_t - borrow_{t-1}) + (defer_t - defer_{t-1})
             + accel_t + capex_cut_t

    subject to Cash_t >= MinimumCash for every t, and each variable capped
    by config/settings.yaml's policy limits (max_borrowing,
    max_payment_delay_days -> a EUR cap, max_collections_acceleration_pct
    x that month's revenue, max_capex_reduction_pct x planned CAPEX).

    minimize sum_t [ borrow_t x monthly_rate + accel_t x factoring_discount
                      + capex_cut_t x capex_delay_cost ]

Payment deferral carries no direct financial cost in this model -- its
real cost (strained vendor relationships) isn't a clean EUR figure, so
it's bounded by policy and reported separately as "operational impact"
rather than folded into the objective as an invented cost.

The BaselineNetCF used here is deterministic (point estimates: Phase 7's
forecast + historical mean margin/DSO/DPO, no randomness) -- consistent
with the project's documented Monte-Carlo-then-optimize split (see
CLAUDE.md): optimize against the expected case, then re-run Monte Carlo on
the optimized policy to show the before/after liquidity breach probability.

Run: python -m src.optimization.cash_management
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from src.common.config import PROJECT_ROOT, load_config
from src.common.logging_config import get_logger
from src.risk.liquidity import compute_liquidity_metrics, risk_score
from src.simulation.monte_carlo import _entity_inputs, load_simulation_context, run_simulation

log = get_logger("cash_management")

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "optimization_result.json"
PLAN_OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "optimization_plan.csv"


def deterministic_entity_path(entity_id: str, context: dict, scenario: str, horizon: int) -> dict:
    """The same cash-flow model as monte_carlo.simulate_entity, but with
    every random draw replaced by its mean -- a single deterministic path,
    the correct baseline for an LP (which has no concept of uncertainty)."""
    inp = _entity_inputs(
        entity_id, context["forecasts"], context["income_statement"], context["working_capital"],
        context["cash_flow"], context["balance_sheet"], horizon, scenario,
    )
    revenue = inp["revenue_point"]
    opex = inp["opex_point"]
    cogs = revenue * inp["margin_mean"]
    days = inp["days_in_month"]

    ar = revenue * inp["dso_mean"] / days
    ap = (cogs + opex) * inp["dpo_mean"] / days
    ar_prev = np.concatenate([[inp["opening_ar"]], ar[:-1]])
    ap_prev = np.concatenate([[inp["opening_ap"]], ap[:-1]])
    collections = ar_prev + revenue - ar
    payments = ap_prev + (cogs + opex) - ap
    operating_cf = collections - payments

    planned_capex = max(-inp["other_cf_mean"], 0.0)
    other_cf = inp["other_cf_mean"] - inp["monthly_interest"]

    net_cf = operating_cf + other_cf
    cash_baseline = inp["opening_cash"] + np.cumsum(net_cf)

    return {
        "net_cf": net_cf, "cash_baseline": cash_baseline, "opening_cash": inp["opening_cash"],
        "revenue": revenue, "ap_related_spend": cogs + opex,
        "planned_capex": np.full(horizon, planned_capex),
    }


def consolidated_baseline(context: dict, scenario: str, horizon: int) -> dict:
    entities = sorted(context["income_statement"]["entity_id"].unique())
    paths = [deterministic_entity_path(e, context, scenario, horizon) for e in entities]
    return {
        "net_cf": sum(p["net_cf"] for p in paths),
        "cash_baseline": sum(p["cash_baseline"] for p in paths),
        "opening_cash": sum(p["opening_cash"] for p in paths),
        "revenue": sum(p["revenue"] for p in paths),
        "ap_related_spend": sum(p["ap_related_spend"] for p in paths),
        "planned_capex": sum(p["planned_capex"] for p in paths),
    }


def solve_optimization(baseline: dict, config: dict, minimum_cash: float | np.ndarray) -> dict:
    """minimum_cash may be a scalar (flat policy) or a per-month array
    (e.g. a volatility-buffered threshold, see SAFETY_BUFFER_Z)."""
    horizon = len(baseline["net_cf"])
    minimum_cash_by_month = np.broadcast_to(minimum_cash, horizon)
    opt_cfg = config["optimization"]
    monthly_borrow_rate = opt_cfg["borrowing_rate_annual"] / 12
    max_defer_cap = opt_cfg["max_payment_delay_days"] / 30 * float(np.mean(baseline["ap_related_spend"]))

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver is not available in this environment.")

    borrow = [solver.NumVar(0, opt_cfg["max_borrowing"], f"borrow_{t}") for t in range(horizon)]
    defer = [solver.NumVar(0, max_defer_cap, f"defer_{t}") for t in range(horizon)]
    accel = [
        solver.NumVar(0, opt_cfg["max_collections_acceleration_pct"] * baseline["revenue"][t], f"accel_{t}")
        for t in range(horizon)
    ]
    capex_cut = [
        solver.NumVar(0, opt_cfg["max_capex_reduction_pct"] * baseline["planned_capex"][t], f"capex_cut_{t}")
        for t in range(horizon)
    ]
    cash = [solver.NumVar(-solver.infinity(), solver.infinity(), f"cash_{t}") for t in range(horizon)]

    for t in range(horizon):
        prev_borrow = borrow[t - 1] if t > 0 else 0
        prev_defer = defer[t - 1] if t > 0 else 0
        prev_cash = cash[t - 1] if t > 0 else baseline["opening_cash"]
        solver.Add(
            cash[t] == prev_cash + baseline["net_cf"][t]
            + (borrow[t] - prev_borrow) + (defer[t] - prev_defer) + accel[t] + capex_cut[t]
        )
        solver.Add(cash[t] >= float(minimum_cash_by_month[t]))

    objective = solver.Objective()
    for t in range(horizon):
        objective.SetCoefficient(borrow[t], monthly_borrow_rate)
        objective.SetCoefficient(accel[t], opt_cfg["factoring_discount_rate"])
        objective.SetCoefficient(capex_cut[t], opt_cfg["capex_delay_cost_rate"])
    objective.SetMinimization()

    status = solver.Solve()
    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE", pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    }.get(status, f"UNKNOWN({status})")

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        log.error("Optimization did not find a feasible solution (status=%s).", status_name)
        return {"status": status_name, "feasible": False}

    plan = pd.DataFrame({
        "step": range(1, horizon + 1),
        "borrow_balance": [v.solution_value() for v in borrow],
        "defer_balance": [v.solution_value() for v in defer],
        "accel": [v.solution_value() for v in accel],
        "capex_cut": [v.solution_value() for v in capex_cut],
        "cash_optimized": [v.solution_value() for v in cash],
        "cash_baseline": baseline["cash_baseline"],
    })
    plan["net_adjustment"] = plan["cash_optimized"] - plan["cash_baseline"]

    financing_cost = sum(v.solution_value() * monthly_borrow_rate for v in borrow)
    acceleration_cost = sum(v.solution_value() * opt_cfg["factoring_discount_rate"] for v in accel)
    capex_cost = sum(v.solution_value() * opt_cfg["capex_delay_cost_rate"] for v in capex_cut)

    return {
        "status": status_name, "feasible": True, "plan": plan, "total_cost": objective.Value(),
        "financing_cost": financing_cost, "acceleration_cost": acceleration_cost, "capex_delay_cost": capex_cost,
        "max_borrow_balance": max(v.solution_value() for v in borrow),
        "max_defer_balance": max(v.solution_value() for v in defer),
        "total_accelerated": sum(v.solution_value() for v in accel),
        "total_capex_cut": sum(v.solution_value() for v in capex_cut),
    }


def compare_before_after(mc_result: dict, config: dict, plan: pd.DataFrame, minimum_cash: float) -> dict:
    """Applies the optimizer's fixed decisions as a deterministic monthly
    cash adjustment on top of every already-simulated path -- a static,
    pre-committed policy (the same plan regardless of how the stochastic
    path actually unfolds), not a reactive one that would need a much more
    complex dynamic-programming formulation. Documented simplification,
    not hidden."""
    before_cash = mc_result["consolidated_cash"]
    cumulative_adjustment = plan["net_adjustment"].to_numpy()
    after_cash = before_cash + cumulative_adjustment[None, :]

    before_summary, _ = compute_liquidity_metrics(before_cash, mc_result["periods"], minimum_cash)
    after_summary, _ = compute_liquidity_metrics(after_cash, mc_result["periods"], minimum_cash)

    thresholds = config["liquidity"]["risk_thresholds"]
    before_summary["risk_score"] = risk_score(before_summary["probability_of_liquidity_breach"], thresholds)
    after_summary["risk_score"] = risk_score(after_summary["probability_of_liquidity_breach"], thresholds)

    before_summary["average_cash"] = float(before_cash.mean())
    after_summary["average_cash"] = float(after_cash.mean())

    return {"before": before_summary, "after": after_summary}


# How many standard deviations of the Monte Carlo's own month-by-month
# cash volatility to hold as an extra buffer on top of the point-forecast
# minimum-cash constraint -- a deterministic LP has no concept of
# uncertainty on its own, so without this the plan only ever protects the
# single expected path, not the range of outcomes Monte Carlo shows are
# actually plausible. A lightweight, standard approximation of a full
# chance-constrained stochastic program (see CLAUDE.md). Z=3 (~99.7% one-
# sided coverage under normality, a conventional "three-sigma" safety-stock
# convention) took the re-simulated breach probability from 100% (Z=0, no
# buffer) to 2.2% at a modest total cost (~EUR 50k) -- Z=1 and Z=2 were
# tried first and left 71% and 23% breach probability respectively, i.e.
# genuinely under-protective, not just more expensive than necessary.
SAFETY_BUFFER_Z = 3.0


def run_optimization(scenario: str = "stress") -> dict:
    """Uses optimization.demo_minimum_cash (a hypothetical, stricter policy)
    rather than the real liquidity.minimum_cash -- the real policy is never
    breached even under the stress scenario (see CLAUDE.md), so optimizing
    against it would trivially find "do nothing" as the answer. Reports
    that real-policy headroom explicitly, alongside the demo optimization."""
    config = load_config()
    horizon = config["forecast"]["horizon_months"]
    context = load_simulation_context()

    baseline = consolidated_baseline(context, scenario, horizon)
    real_minimum_cash = config["liquidity"]["minimum_cash"]
    real_policy_headroom = float(baseline["cash_baseline"].min() - real_minimum_cash)

    mc_result = run_simulation(context, scenario=scenario)
    monthly_std = mc_result["consolidated_cash"].std(axis=0, ddof=1)

    demo_minimum_cash = config["optimization"]["demo_minimum_cash"]
    buffered_minimum_cash = demo_minimum_cash + SAFETY_BUFFER_Z * monthly_std
    solution = solve_optimization(baseline, config, buffered_minimum_cash)
    if not solution["feasible"]:
        return {
            "scenario": scenario, "solution": solution,
            "real_minimum_cash": real_minimum_cash, "real_policy_headroom": real_policy_headroom,
        }

    comparison = compare_before_after(mc_result, config, solution["plan"], demo_minimum_cash)
    return {
        "scenario": scenario, "baseline": baseline, "solution": solution, "comparison": comparison,
        "real_minimum_cash": real_minimum_cash, "real_policy_headroom": real_policy_headroom,
        "demo_minimum_cash": demo_minimum_cash,
    }


def main() -> None:
    result = run_optimization(scenario="stress")
    solution = result["solution"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not solution["feasible"]:
        with OUTPUT_PATH.open("w") as f:
            json.dump({"scenario": result["scenario"], "status": solution["status"], "feasible": False}, f, indent=2)
        log.error("No feasible optimization plan found under the '%s' scenario.", result["scenario"])
        return

    solution["plan"].to_csv(PLAN_OUTPUT_PATH, index=False)

    comparison = result["comparison"]
    report = {
        "scenario": result["scenario"],
        "real_minimum_cash": result["real_minimum_cash"], "real_policy_headroom": result["real_policy_headroom"],
        "demo_minimum_cash": result["demo_minimum_cash"], "status": solution["status"],
        "total_cost": solution["total_cost"], "financing_cost": solution["financing_cost"],
        "acceleration_cost": solution["acceleration_cost"], "capex_delay_cost": solution["capex_delay_cost"],
        "max_borrow_balance": solution["max_borrow_balance"], "max_defer_balance": solution["max_defer_balance"],
        "total_accelerated": solution["total_accelerated"], "total_capex_cut": solution["total_capex_cut"],
        "before": comparison["before"], "after": comparison["after"],
    }
    with OUTPUT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Wrote optimization plan to %s", PLAN_OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info("Wrote optimization report to %s", OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info(
        "Real liquidity policy (EUR %.0f) headroom under '%s' stress: EUR %.0f -- never breached",
        result["real_minimum_cash"], result["scenario"], result["real_policy_headroom"],
    )
    log.info(
        "[demo policy EUR %.0f] status=%s total_cost=%.0f (financing=%.0f, acceleration=%.0f, capex_delay=%.0f)",
        result["demo_minimum_cash"], solution["status"], solution["total_cost"],
        solution["financing_cost"], solution["acceleration_cost"], solution["capex_delay_cost"],
    )
    log.info(
        "Liquidity risk BEFORE optimization: %s (P(breach)=%.1f%%, avg cash=%.0f) -> "
        "AFTER: %s (P(breach)=%.1f%%, avg cash=%.0f)",
        comparison["before"]["risk_score"], comparison["before"]["probability_of_liquidity_breach"] * 100,
        comparison["before"]["average_cash"],
        comparison["after"]["risk_score"], comparison["after"]["probability_of_liquidity_breach"] * 100,
        comparison["after"]["average_cash"],
    )


if __name__ == "__main__":
    main()
