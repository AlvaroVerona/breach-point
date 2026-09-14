"""Optimization page: current (baseline) vs optimized cash-management
strategy, and a management recommendation generated from the actual
solver output (not a template)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_optimization_plan, load_optimization_result, outputs_available
from app.components.style import apply_layout, page_header, status_badge

st.set_page_config(page_title="Optimization | Breach Point", layout="wide")
page_header("Cash Management Optimization", "OR-Tools LP: lowest-cost mix of borrowing, factoring, payment deferral and CAPEX reduction.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make optimize` first.")
    st.stop()

result = load_optimization_result()
plan = load_optimization_plan()

if not result or result.get("status") not in ("OPTIMAL", "FEASIBLE"):
    st.error("No feasible optimization plan found. See reports/outputs/optimization_result.json.")
    st.stop()

st.info(
    f"Demonstration uses a hypothetical stricter policy (€{result['demo_minimum_cash']:,.0f}) under the "
    f"'{result['scenario']}' stress scenario, since the real liquidity policy "
    f"(€{result['real_minimum_cash']:,.0f}) is never breached -- €{result['real_policy_headroom']:,.0f} "
    "of headroom remains even under stress. See CLAUDE.md."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total cost", f"€{result['total_cost']:,.0f}")
col2.metric("Financing cost", f"€{result['financing_cost']:,.0f}")
col3.metric("Acceleration (factoring) cost", f"€{result['acceleration_cost']:,.0f}")
col4.metric("CAPEX delay cost", f"€{result['capex_delay_cost']:,.0f}")

st.divider()
st.markdown("#### Current strategy vs. optimized strategy (re-simulated with Monte Carlo)")
before, after = result["before"], result["after"]
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Current strategy (no intervention)**")
    st.markdown(status_badge(before["risk_score"], before["risk_score"]), unsafe_allow_html=True)
    st.metric("P(breach)", f"{before['probability_of_liquidity_breach'] * 100:.1f}%")
    st.metric("Average cash", f"€{before['average_cash']:,.0f}")
with c2:
    st.markdown("**Optimized strategy**")
    st.markdown(status_badge(after["risk_score"], after["risk_score"]), unsafe_allow_html=True)
    st.metric("P(breach)", f"{after['probability_of_liquidity_breach'] * 100:.1f}%",
              delta=f"{(after['probability_of_liquidity_breach'] - before['probability_of_liquidity_breach']) * 100:.1f} pp")
    st.metric("Average cash", f"€{after['average_cash']:,.0f}",
              delta=f"€{after['average_cash'] - before['average_cash']:,.0f}")

st.divider()
st.markdown("#### Baseline vs. optimized cash trajectory (deterministic plan)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=plan["step"], y=plan["cash_baseline"], name="Baseline (no action)",
                          mode="lines", line=dict(color="#F58518", width=2, dash="dot")))
fig.add_trace(go.Scatter(x=plan["step"], y=plan["cash_optimized"], name="Optimized",
                          mode="lines", line=dict(color="#54A24B", width=2)))
apply_layout(fig, height=380, yaxis_title="EUR", xaxis_title="Month")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Decision levers used, by month")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=plan["step"], y=plan["borrow_balance"], name="Borrowing balance", marker_color="#4C78A8"))
fig2.add_trace(go.Bar(x=plan["step"], y=plan["defer_balance"], name="Deferred payments balance", marker_color="#B279A2"))
fig2.add_trace(go.Bar(x=plan["step"], y=plan["accel"], name="Accelerated collections", marker_color="#54A24B"))
fig2.add_trace(go.Bar(x=plan["step"], y=plan["capex_cut"], name="CAPEX reduction", marker_color="#F58518"))
apply_layout(fig2, height=340, yaxis_title="EUR", barmode="stack", xaxis_title="Month")
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.markdown("#### Recommendation")


def build_recommendation(result: dict) -> list[str]:
    lines = [f"Liquidity Risk (demo policy): {result['before']['risk_score']} → {result['after']['risk_score']}."]
    levers = []
    if result["total_accelerated"] > 1:
        levers.append(f"accelerate collections (factoring) up to €{result['total_accelerated']:,.0f} total")
    if result["max_borrow_balance"] > 1:
        levers.append(f"draw the credit facility up to €{result['max_borrow_balance']:,.0f} at its peak")
    if result["max_defer_balance"] > 1:
        levers.append(f"defer up to €{result['max_defer_balance']:,.0f} of payments at the peak")
    if result["total_capex_cut"] > 1:
        levers.append(f"reduce CAPEX by €{result['total_capex_cut']:,.0f} total")
    if levers:
        lines.append("Recommended actions, ranked by the solver's own cost-minimizing order: " + "; ".join(levers) + ".")
    else:
        lines.append("No intervention required under this scenario.")
    return lines


for line in build_recommendation(result):
    st.markdown(f"- {line}")
