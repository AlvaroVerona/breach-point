"""Shared chart styling: a fixed categorical color per entity (never
reassigned when a filter changes which entities are shown) and a status
palette for severity/risk, kept distinct from the categorical colors so a
"HIGH" badge is never confused with an entity's own color."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

ENTITY_COLORS = {
    "ENT_EU": "#4C78A8",
    "ENT_US": "#F58518",
    "ENT_UK": "#54A24B",
    "CONSOLIDATED": "#72727A",
}

STATUS_COLORS = {
    "LOW": "#54A24B", "PASS": "#54A24B", "OPTIMAL": "#54A24B",
    "MEDIUM": "#ECC94B", "FEASIBLE": "#ECC94B",
    "HIGH": "#F58518",
    "CRITICAL": "#E45756", "FAIL": "#E45756", "INFEASIBLE": "#E45756",
    "INFO": "#B0B0B8", "LOW_SEVERITY": "#54A24B",
}

PLOT_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(size=13),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)


def apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**{**PLOT_LAYOUT_DEFAULTS, **overrides})
    return fig


def entity_color(entity_id: str) -> str:
    return ENTITY_COLORS.get(entity_id, "#B0B0B8")


def status_color(status: str) -> str:
    return STATUS_COLORS.get(str(status).upper(), "#B0B0B8")


def status_badge(label: str, status: str) -> str:
    color = status_color(status)
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color};'
        f'border-radius:6px;padding:2px 10px;font-weight:600;font-size:0.85em">{label}</span>'
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
