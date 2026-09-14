"""Data Quality page: overall score and dimension breakdown, missing/
duplicate/invalid/reconciliation counts, and a filterable table of
quarantined records (Entity, Source, Account, Severity, Date)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data_loader import load_quality_report, load_quarantine_detail, outputs_available
from app.components.style import apply_layout, page_header, status_color

st.set_page_config(page_title="Data Quality | Breach Point", layout="wide")
page_header("Data Quality", "Every number here comes from the last `make validate` run (reports/outputs/quality_report.json).")

if not outputs_available():
    st.warning("No pipeline outputs found. Run `make generate-data validate` first.")
    st.stop()

report = load_quality_report()

st.markdown(f"### Financial Data Quality Score: {report.get('overall_score', float('nan')):.1f} / 100")

dims = report.get("dimension_scores", {})
cols = st.columns(len(dims) or 1)
for col, (name, score) in zip(cols, dims.items()):
    col.metric(name.capitalize(), f"{score:.1f}")

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("#### Issues by severity")
    severity_counts = report.get("issue_counts_by_severity", {})
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    severities = [s for s in order if s in severity_counts] + [s for s in severity_counts if s not in order]
    fig = go.Figure(go.Bar(
        x=severities, y=[severity_counts[s] for s in severities],
        marker_color=[status_color(s) for s in severities],
    ))
    apply_layout(fig, height=340, yaxis_title="Issue count", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Row counts: raw → validated → quarantined")
    raw_counts = report.get("row_counts_raw", {})
    validated_counts = report.get("row_counts_validated", {})
    quarantined_counts = report.get("row_counts_quarantined", {})
    datasets = list(raw_counts.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Validated", x=datasets, y=[validated_counts.get(d, 0) for d in datasets],
                          marker_color=status_color("PASS")))
    fig.add_trace(go.Bar(name="Quarantined", x=datasets, y=[quarantined_counts.get(d, 0) for d in datasets],
                          marker_color=status_color("CRITICAL")))
    apply_layout(fig, height=340, barmode="stack", yaxis_title="Records")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Top validation rules triggered")
rule_counts = report.get("issue_counts_by_rule", {})
rules_df = pd.DataFrame(sorted(rule_counts.items(), key=lambda kv: -kv[1]), columns=["validation_rule", "count"])
st.dataframe(rules_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown("#### Quarantined records")

dataset_choice = st.selectbox("Dataset", ["transactions", "accounts_receivable", "accounts_payable"])
detail = load_quarantine_detail(dataset_choice, "row_uid")

if detail.empty:
    st.info("No quarantined records for this dataset.")
else:
    filter_cols = st.columns(4)
    severities_available = sorted(detail["severity"].dropna().unique())
    severity_filter = filter_cols[0].multiselect("Severity", severities_available, default=severities_available)

    entity_filter = None
    if "entity_id" in detail.columns:
        entities_available = sorted(detail["entity_id"].dropna().unique())
        entity_filter = filter_cols[1].multiselect("Entity", entities_available, default=entities_available)

    source_col = "source_system" if "source_system" in detail.columns else None
    source_filter = None
    if source_col:
        sources_available = sorted(detail[source_col].dropna().unique())
        source_filter = filter_cols[2].multiselect("Source", sources_available, default=sources_available)

    rule_filter = filter_cols[3].multiselect("Validation rule", sorted(detail["validation_rule"].unique()))

    filtered = detail[detail["severity"].isin(severity_filter)]
    if entity_filter is not None:
        filtered = filtered[filtered["entity_id"].isin(entity_filter)]
    if source_filter is not None:
        filtered = filtered[filtered[source_col].isin(source_filter)]
    if rule_filter:
        filtered = filtered[filtered["validation_rule"].isin(rule_filter)]

    st.caption(f"{len(filtered):,} of {len(detail):,} quarantine ledger rows shown")
    display_cols = [c for c in [
        "record_id", "validation_rule", "severity", "reason", "entity_id",
        source_col, "account_id", "transaction_date", "invoice_date", "timestamp",
    ] if c and c in filtered.columns]
    st.dataframe(filtered[display_cols].head(500), use_container_width=True, hide_index=True)
