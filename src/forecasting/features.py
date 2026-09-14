"""Assembles the per-entity monthly panel of forecast targets from the
Phase 4 statements (recomputed, not read from disk -- consistent with how
every later phase depends on earlier ones directly rather than trusting a
possibly-stale CSV) and builds lag/rolling features for the ML model.

Target metrics (§28): revenue, operating_expense, accounts_receivable,
accounts_payable, operating_cf, ending_cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.accounting.balance_sheet import build_balance_sheet
from src.accounting.cash_flow import build_cash_flow
from src.accounting.income_statement import build_income_statement, _load_processed

TARGET_METRICS = [
    "revenue", "operating_expense", "accounts_receivable",
    "accounts_payable", "operating_cf", "ending_cash",
]


def load_target_series() -> pd.DataFrame:
    transactions, coa, fx_rates = _load_processed()
    income_statement = build_income_statement(transactions, coa, fx_rates)
    balance_sheet = build_balance_sheet(transactions, coa, fx_rates, income_statement)
    cash_flow = build_cash_flow(transactions, coa, fx_rates, balance_sheet)

    panel = income_statement[["entity_id", "period", "revenue", "operating_expense"]].merge(
        balance_sheet[["entity_id", "period", "accounts_receivable", "accounts_payable", "cash"]],
        on=["entity_id", "period"], how="inner",
    ).merge(
        cash_flow[["entity_id", "period", "operating_cf"]],
        on=["entity_id", "period"], how="inner",
    )
    panel = panel.rename(columns={"cash": "ending_cash"})
    return panel.sort_values(["entity_id", "period"]).reset_index(drop=True)


def get_series(panel: pd.DataFrame, entity_id: str, metric: str) -> pd.Series:
    sub = panel[panel["entity_id"] == entity_id].sort_values("period")
    return pd.Series(sub[metric].to_numpy(), index=pd.PeriodIndex(sub["period"], freq="M"), name=metric)


def build_lag_features(
    series: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 12), rolling_window: int = 3,
) -> pd.DataFrame:
    """One row per period with lag_k / rolling_mean / a cyclical month
    encoding / a linear trend index as features and the period's own value
    as the target -- a row is only usable once all lag features exist, so
    the first `max(lags)` periods are dropped (NaN lag features, no
    fabricated fill)."""
    df = pd.DataFrame({"target": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    df["rolling_mean"] = series.shift(1).rolling(rolling_window).mean()
    month_num = series.index.month
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12)
    df["trend_index"] = np.arange(len(series))
    return df.dropna()
