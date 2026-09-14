"""Cached readers for the pipeline's output artifacts (reports/outputs/,
data/processed/, data/quarantine/). The dashboard reads what `make all`
already produced rather than recomputing the pipeline on every page
interaction -- Phase 7's forecasting alone fits a model per series, which
would make the UI unusably slow if re-run on every click."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.common.config import PROJECT_ROOT, load_config

REPORTS_DIR = PROJECT_ROOT / "reports" / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@st.cache_data
def get_config() -> dict:
    return load_config()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


@st.cache_data
def load_quality_report() -> dict:
    return _read_json(REPORTS_DIR / "quality_report.json")


@st.cache_data
def load_processed(name: str) -> pd.DataFrame:
    return _read_csv(PROCESSED_DIR / f"{name}.csv")


@st.cache_data
def load_quarantine(name: str) -> pd.DataFrame:
    return _read_csv(QUARANTINE_DIR / f"{name}.csv")


@st.cache_data
def load_raw(name: str) -> pd.DataFrame:
    return _read_csv(RAW_DIR / f"{name}.csv")


@st.cache_data
def load_quarantine_detail(name: str, id_column: str) -> pd.DataFrame:
    """Quarantine ledger rows joined back to the raw record's own fields
    (entity_id, source system/account, date) via row_uid == record_id, so
    the Data Quality page can filter by Entity/Source/Severity/Date the way
    the spec asks -- the quarantine ledger alone only carries the rule/
    severity/reason, not the record's business fields."""
    ledger = load_quarantine(name)
    if ledger.empty:
        return ledger
    raw = load_raw(name)
    raw = raw.reset_index().rename(columns={"index": "_raw_row"})
    raw.insert(0, "row_uid", [f"{name.upper()}_{i + 1:08d}" for i in range(len(raw))])
    return ledger.merge(raw, left_on="record_id", right_on="row_uid", how="left", suffixes=("", "_raw"))


@st.cache_data
def load_income_statement() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "income_statement.csv")


@st.cache_data
def load_balance_sheet() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "balance_sheet.csv")


@st.cache_data
def load_cash_flow() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "cash_flow.csv")


@st.cache_data
def load_working_capital() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "working_capital.csv")


@st.cache_data
def load_accruals() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "accruals.csv")


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "forecasts.csv")


@st.cache_data
def load_forecast_evaluation() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "forecast_evaluation.csv")


@st.cache_data
def load_monte_carlo_summary() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "monte_carlo_summary.csv")


@st.cache_data
def load_monte_carlo_paths() -> np.ndarray | None:
    path = REPORTS_DIR / "monte_carlo_consolidated_paths.npy"
    return np.load(path) if path.exists() else None


@st.cache_data
def load_liquidity_risk() -> dict:
    return _read_json(REPORTS_DIR / "liquidity_risk.json")


@st.cache_data
def load_liquidity_risk_by_month() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "liquidity_risk_by_month.csv")


@st.cache_data
def load_optimization_result() -> dict:
    return _read_json(REPORTS_DIR / "optimization_result.json")


@st.cache_data
def load_optimization_plan() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "optimization_plan.csv")


def outputs_available() -> bool:
    """True once `make all` (or at least the early phases) has run."""
    return (REPORTS_DIR / "quality_report.json").exists()
