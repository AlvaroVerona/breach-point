"""Breach Point -- Executive Overview.

Run: streamlit run app/app.py
Requires `make all` (or at least generate-data/validate/build-statements/
provisions/working-capital/train/simulate/risk/optimize) to have populated
reports/outputs/ first -- this dashboard reads those artifacts rather than
recomputing the pipeline live.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    get_config, load_balance_sheet, load_income_statement, load_liquidity_risk,
    load_quality_report, load_working_capital, outputs_available,
)
from app.components.style import apply_layout, entity_color, status_badge

st.set_page_config(page_title="Breach Point", page_icon="\U0001F4C9", layout="wide")

st.title("Breach Point")
st.caption("Financial Data Quality, Forecasting & Liquidity Optimization Platform")

if not outputs_available():
    st.warning(
        "No pipeline outputs found yet. Run `make all` (or at minimum "
        "`make generate-data validate build-statements`) from the project root, then reload."
    )
    st.stop()

income_statement = load_income_statement()
balance_sheet = load_balance_sheet()
working_capital = load_working_capital()
quality = load_quality_report()
liquidity = load_liquidity_risk()
config = get_config()

latest_period = income_statement["period"].max()
latest_is = income_statement[income_statement["period"] == latest_period]
latest_bs = balance_sheet[balance_sheet["period"] == latest_period]
latest_wc = working_capital[working_capital["period"] == latest_period]

revenue = latest_is["revenue"].sum()
ebitda = latest_is["ebitda"].sum()
net_income = latest_is["net_income"].sum()
cash = latest_bs["cash"].sum()
net_working_capital = (latest_bs["accounts_receivable"] + latest_bs["inventory"] - latest_bs["accounts_payable"]).sum()
quality_score = quality.get("overall_score", float("nan"))
base_liquidity = liquidity.get("base", {})
risk_score = base_liquidity.get("risk_score", "N/A")
breach_prob = base_liquidity.get("probability_of_liquidity_breach", 0.0) * 100

st.caption(f"All KPIs below are consolidated (3 entities) for the latest month in the dataset: **{latest_period}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue (month)", f"€{revenue:,.0f}")
col2.metric("EBITDA (month)", f"€{ebitda:,.0f}")
col3.metric("Net Income (month)", f"€{net_income:,.0f}")
col4.metric("Cash", f"€{cash:,.0f}")

col5, col6, col7 = st.columns(3)
col5.metric("Net Working Capital", f"€{net_working_capital:,.0f}")
with col6:
    st.markdown("**Financial Data Quality Score**")
    st.markdown(f"### {quality_score:.1f} / 100")
with col7:
    st.markdown("**Liquidity Risk (base scenario)**")
    st.markdown(status_badge(risk_score, risk_score) + f" &nbsp; P(breach)={breach_prob:.1f}%", unsafe_allow_html=True)

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("#### Consolidated Revenue & Net Income")
    monthly = income_statement.groupby("period")[["revenue", "net_income"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["period"], y=monthly["revenue"], name="Revenue",
                              mode="lines", line=dict(color=entity_color("ENT_EU"), width=2)))
    fig.add_trace(go.Scatter(x=monthly["period"], y=monthly["net_income"], name="Net Income",
                              mode="lines", line=dict(color=entity_color("ENT_US"), width=2)))
    apply_layout(fig, yaxis_title="EUR", height=340)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Cash by Entity")
    fig = go.Figure()
    for entity_id, group in balance_sheet.groupby("entity_id"):
        group = group.sort_values("period")
        fig.add_trace(go.Scatter(x=group["period"], y=group["cash"], name=entity_id,
                                  mode="lines", line=dict(color=entity_color(entity_id), width=2)))
    apply_layout(fig, yaxis_title="EUR", height=340)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Use the sidebar to open Data Quality, Financial Statements, Working Capital, "
    "Forecasting, Liquidity Risk and Optimization."
)
