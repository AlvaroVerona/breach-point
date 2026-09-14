"""Forecasting page: historical + 12-month forecast + confidence interval,
selectable entity/metric, and the model comparison table (§29-31)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import (
    load_balance_sheet, load_cash_flow, load_forecast_evaluation, load_forecasts,
    load_income_statement, outputs_available,
)
from app.components.style import apply_layout, entity_color, page_header

st.set_page_config(page_title="Forecasting | Breach Point", layout="wide")
page_header("Forecasting", "12-month-ahead forecast per entity, best model selected on a blind 6-month holdout (see CLAUDE.md).")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make train` first.")
    st.stop()

forecasts = load_forecasts()
evaluation = load_forecast_evaluation()
income_statement = load_income_statement()
balance_sheet = load_balance_sheet()
cash_flow = load_cash_flow()

METRIC_SOURCES = {
    "revenue": (income_statement, "revenue"),
    "operating_expense": (income_statement, "operating_expense"),
    "accounts_receivable": (balance_sheet, "accounts_receivable"),
    "accounts_payable": (balance_sheet, "accounts_payable"),
    "operating_cf": (cash_flow, "operating_cf"),
    "ending_cash": (balance_sheet, "cash"),
}

col1, col2 = st.columns(2)
entity_choice = col1.selectbox("Entity", sorted(forecasts["entity_id"].unique()))
metric_choice = col2.selectbox("Metric", list(METRIC_SOURCES.keys()))

hist_df, hist_col = METRIC_SOURCES[metric_choice]
historical = hist_df[hist_df["entity_id"] == entity_choice].sort_values("period")
fc = forecasts[(forecasts["entity_id"] == entity_choice) & (forecasts["metric"] == metric_choice)].sort_values("step")

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical["period"], y=historical[hist_col], name="Historical",
                          mode="lines", line=dict(color=entity_color(entity_choice), width=2)))
if not fc.empty:
    fig.add_trace(go.Scatter(x=fc["period"], y=fc["forecast"], name=f"Forecast ({fc['model'].iloc[0]})",
                              mode="lines", line=dict(color="#B279A2", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["period"], fc["period"][::-1]]),
        y=pd.concat([fc["ci_upper"], fc["ci_lower"][::-1]]),
        fill="toself", fillcolor="rgba(178,121,162,0.15)", line=dict(width=0),
        name="95% CI", hoverinfo="skip",
    ))
apply_layout(fig, height=440, yaxis_title=metric_choice)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Model comparison (6-month blind holdout)")
eval_view = evaluation[(evaluation["entity_id"] == entity_choice) & (evaluation["metric"] == metric_choice)]
eval_view = eval_view[["model", "mae", "rmse", "mape", "smape", "selected"]].sort_values("rmse")
st.dataframe(eval_view.round(2), use_container_width=True, hide_index=True)

st.caption(
    "MAPE/sMAPE can be unstable for a series that crosses near zero (e.g. Operating Cash "
    "Flow some months) -- a metric limitation, not a forecast failure. Prefer RMSE there."
)

st.divider()
st.markdown("#### Model selection across all series")
selected = evaluation[evaluation["selected"]]
counts = selected["model"].value_counts().reset_index()
counts.columns = ["model", "count"]
fig2 = go.Figure(go.Bar(x=counts["model"], y=counts["count"], marker_color="#4C78A8"))
apply_layout(fig2, height=300, yaxis_title="Series selected", showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
