"""Working Capital page: DSO, DPO, DIO, CCC trends and AR/AP levels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_working_capital, outputs_available
from app.components.style import apply_layout, entity_color, page_header

st.set_page_config(page_title="Working Capital | Breach Point", layout="wide")
page_header("Working Capital", "DSO, DPO, DIO and Cash Conversion Cycle -- pure ratios of the financial statements, 3-month smoothed for trend.")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make working-capital` first.")
    st.stop()

wc = load_working_capital()
entities = sorted(wc["entity_id"].unique())
entity_choice = st.multiselect("Entity", entities, default=entities)
filtered = wc[wc["entity_id"].isin(entity_choice)]

metric_labels = {"dso_3m_avg": "DSO", "dpo_3m_avg": "DPO", "dio_3m_avg": "DIO", "ccc_3m_avg": "CCC"}

top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)
panels = [top_left, top_right, bottom_left, bottom_right]

for panel, (metric, label) in zip(panels, metric_labels.items()):
    with panel:
        st.markdown(f"#### {label} (3-month average)")
        fig = go.Figure()
        for entity_id, group in filtered.groupby("entity_id"):
            group = group.sort_values("period")
            fig.add_trace(go.Scatter(x=group["period"], y=group[metric], name=entity_id,
                                      mode="lines", line=dict(color=entity_color(entity_id), width=2)))
        apply_layout(fig, height=300, yaxis_title="Days")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("#### Accounts Receivable & Accounts Payable")
fig = go.Figure()
for entity_id, group in filtered.groupby("entity_id"):
    group = group.sort_values("period")
    fig.add_trace(go.Scatter(x=group["period"], y=group["accounts_receivable"], name=f"{entity_id} AR",
                              mode="lines", line=dict(color=entity_color(entity_id), width=2)))
    fig.add_trace(go.Scatter(x=group["period"], y=group["accounts_payable"], name=f"{entity_id} AP",
                              mode="lines", line=dict(color=entity_color(entity_id), width=2, dash="dot")))
apply_layout(fig, height=380, yaxis_title="EUR")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "DPO uses AP / (COGS + Operating Expense) rather than the textbook AP / COGS -- this "
    "company's AP funds a broad vendor base (rent, software, marketing, logistics, not just "
    "inventory purchases). See CLAUDE.md for why."
)
