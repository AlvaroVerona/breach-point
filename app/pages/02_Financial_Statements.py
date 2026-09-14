"""Financial Statements page: Income Statement, Balance Sheet and Cash
Flow with entity selection, monthly/yearly view, and accounting integrity
indicators (Balance Sheet / Cash Flow reconciliation status)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_balance_sheet, load_cash_flow, load_income_statement, outputs_available
from app.components.style import apply_layout, page_header, status_color

st.set_page_config(page_title="Financial Statements | Breach Point", layout="wide")
page_header("Financial Statements", "Monthly Income Statement, Balance Sheet and Cash Flow, in EUR.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make build-statements` first.")
    st.stop()

income_statement = load_income_statement()
balance_sheet = load_balance_sheet()
cash_flow = load_cash_flow()

entities = sorted(income_statement["entity_id"].unique())
col1, col2 = st.columns([2, 1])
entity_choice = col1.multiselect("Entity", entities, default=entities)
view = col2.radio("View", ["Monthly", "Yearly"], horizontal=True)

def _aggregate(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    d = df[df["entity_id"].isin(entity_choice)].copy()
    if view == "Yearly":
        d["group"] = d["period"].str.slice(0, 4)
    else:
        d["group"] = d["period"]
    return d.groupby("group")[value_cols].sum().reset_index()


tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

with tab1:
    is_cols = ["revenue", "cogs", "gross_profit", "operating_expense", "ebitda",
               "depreciation", "ebit", "interest", "pretax_income", "taxes", "net_income"]
    agg = _aggregate(income_statement, is_cols)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["group"], y=agg["revenue"], name="Revenue", marker_color="#4C78A8"))
    fig.add_trace(go.Bar(x=agg["group"], y=agg["cogs"], name="COGS", marker_color="#F58518"))
    fig.add_trace(go.Bar(x=agg["group"], y=agg["operating_expense"], name="Operating Expense", marker_color="#E45756"))
    fig.add_trace(go.Scatter(x=agg["group"], y=agg["net_income"], name="Net Income",
                              mode="lines+markers", line=dict(color="#54A24B", width=3)))
    apply_layout(fig, height=420, yaxis_title="EUR", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg.round(0), use_container_width=True, hide_index=True)

with tab2:
    bs_cols = ["cash", "accounts_receivable", "inventory", "other_current_assets", "total_assets",
               "accounts_payable", "debt", "other_liabilities", "income_tax_payable", "total_liabilities",
               "common_stock", "retained_earnings", "total_equity", "difference"]
    d = balance_sheet[balance_sheet["entity_id"].isin(entity_choice)]
    if view == "Yearly":
        snap = d.sort_values("period").groupby([d["period"].str.slice(0, 4), "entity_id"]).last().reset_index()
        snap = snap.rename(columns={"period": "group"})
    else:
        snap = d.rename(columns={"period": "group"})
    agg = snap.groupby("group")[bs_cols].sum(numeric_only=True).reset_index()

    n_fail = (d["status"] == "FAIL").sum()
    n_implausible = d["economically_implausible"].sum() if "economically_implausible" in d.columns else 0
    badge = "✅ All periods reconcile" if n_fail == 0 else f"⚠️ {n_fail} periods FAIL to reconcile"
    st.markdown(f"**Accounting integrity:** {badge}" + (
        f" &nbsp;|&nbsp; {n_implausible} periods flagged economically implausible (see CLAUDE.md)"
        if n_implausible else ""
    ))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["group"], y=agg["total_assets"], name="Total Assets", marker_color="#4C78A8"))
    fig.add_trace(go.Bar(x=agg["group"], y=agg["total_liabilities"], name="Total Liabilities", marker_color="#F58518"))
    fig.add_trace(go.Bar(x=agg["group"], y=agg["total_equity"], name="Total Equity", marker_color="#54A24B"))
    apply_layout(fig, height=420, yaxis_title="EUR", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg.round(0), use_container_width=True, hide_index=True)

with tab3:
    cf_cols = ["operating_cf", "investing_cf", "financing_cf", "net_change_in_cash", "ending_cash"]
    d = cash_flow[cash_flow["entity_id"].isin(entity_choice)]
    if view == "Yearly":
        d2 = d.copy()
        d2["group"] = d2["period"].str.slice(0, 4)
        agg = d2.groupby("group")[["operating_cf", "investing_cf", "financing_cf", "net_change_in_cash"]].sum().reset_index()
    else:
        agg = d.rename(columns={"period": "group"}).groupby("group")[
            ["operating_cf", "investing_cf", "financing_cf", "net_change_in_cash"]
        ].sum().reset_index()

    n_fail = (d["status"] == "FAIL").sum()
    badge = "✅ All periods reconcile" if n_fail == 0 else f"⚠️ {n_fail} periods FAIL to reconcile"
    st.markdown(f"**Cash Flow ↔ Balance Sheet reconciliation:** {badge}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["group"], y=agg["operating_cf"], name="Operating CF", marker_color="#4C78A8"))
    fig.add_trace(go.Bar(x=agg["group"], y=agg["investing_cf"], name="Investing CF", marker_color="#F58518"))
    fig.add_trace(go.Bar(x=agg["group"], y=agg["financing_cf"], name="Financing CF", marker_color="#B279A2"))
    fig.add_trace(go.Scatter(x=agg["group"], y=agg["net_change_in_cash"], name="Net Change",
                              mode="lines+markers", line=dict(color="#54A24B", width=3)))
    apply_layout(fig, height=420, yaxis_title="EUR", barmode="relative")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg.round(0), use_container_width=True, hide_index=True)
