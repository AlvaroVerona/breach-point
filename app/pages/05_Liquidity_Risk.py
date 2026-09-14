"""Liquidity Risk page: Monte Carlo fan chart, probability of breach by
month, and the Base/Optimistic/Pessimistic scenario comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    get_config, load_liquidity_risk, load_liquidity_risk_by_month, load_monte_carlo_summary, outputs_available,
)
from app.components.style import apply_layout, page_header, status_badge

st.set_page_config(page_title="Liquidity Risk | Breach Point", layout="wide")
page_header("Liquidity Risk", "10,000-simulation Monte Carlo of consolidated cash, base scenario.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make simulate risk` first.")
    st.stop()

liquidity = load_liquidity_risk()
monthly = load_liquidity_risk_by_month()
mc_summary = load_monte_carlo_summary()
config = get_config()
minimum_cash = config["liquidity"]["minimum_cash"]

base = liquidity.get("base", {})
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Liquidity Risk Score**")
    st.markdown(status_badge(base.get("risk_score", "N/A"), base.get("risk_score", "N/A")), unsafe_allow_html=True)
col2.metric("P(breach within 12mo)", f"{base.get('probability_of_liquidity_breach', 0) * 100:.1f}%")
col3.metric("Expected minimum cash", f"€{base.get('expected_minimum_cash', 0):,.0f}")
col4.metric("Worst simulated cash", f"€{base.get('worst_simulated_cash', 0):,.0f}")

st.caption(f"Minimum cash requirement: €{minimum_cash:,.0f} (~2 months of consolidated operating outflow -- see CLAUDE.md)")

st.divider()
st.markdown("#### Consolidated cash -- fan chart (base scenario)")
consolidated = mc_summary[mc_summary["entity_id"] == "CONSOLIDATED"].sort_values("period")

fig = go.Figure()
band_pairs = [("p5", "p95", "rgba(76,120,168,0.10)"), ("p10", "p90", "rgba(76,120,168,0.18)"), ("p25", "p75", "rgba(76,120,168,0.28)")]
for low, high, color in band_pairs:
    fig.add_trace(go.Scatter(
        x=list(consolidated["period"]) + list(consolidated["period"][::-1]),
        y=list(consolidated[high]) + list(consolidated[low][::-1]),
        fill="toself", fillcolor=color, line=dict(width=0), name=f"{low}-{high}", hoverinfo="skip",
    ))
fig.add_trace(go.Scatter(x=consolidated["period"], y=consolidated["median"], name="Median",
                          mode="lines", line=dict(color="#4C78A8", width=2)))
fig.add_hline(y=minimum_cash, line_dash="dash", line_color="#E45756", annotation_text="Minimum cash requirement")
apply_layout(fig, height=440, yaxis_title="EUR")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Probability of breach by month")
fig2 = go.Figure(go.Bar(x=monthly["period"], y=monthly["probability_of_breach"] * 100, marker_color="#E45756"))
apply_layout(fig2, height=280, yaxis_title="P(breach) %", showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.markdown("#### Scenario comparison")
scenario_rows = []
for scenario in ["optimistic", "base", "pessimistic"]:
    s = liquidity.get(scenario, {})
    if s:
        scenario_rows.append({
            "scenario": scenario, "risk_score": s.get("risk_score"),
            "P(breach)": f"{s.get('probability_of_liquidity_breach', 0) * 100:.1f}%",
            "expected_min_cash": f"€{s.get('expected_minimum_cash', 0):,.0f}",
            "worst_case": f"€{s.get('worst_simulated_cash', 0):,.0f}",
        })
st.dataframe(scenario_rows, use_container_width=True, hide_index=True)
