"""Estimates expenses incurred but not yet fully reflected as expected,
for recurring expense categories (§25-26): Utilities, Professional
Services and Logistics (business-unit level) and Interest Expense
(corporate level).

Methodology (transparent, not arbitrary -- §26):

    Accrual_t = ExpectedExpense_t - RecognizedExpense_t

ExpectedExpense_t is estimated three ways for every period with enough
history -- historical (expanding-window) average, seasonal average (same
calendar month in prior years), and a trailing rolling average -- and the
rolling average is used as the primary estimate (most standard for a
short, 3-year history: responsive to recent trend while still smoothing
month-to-month noise). All three are reported for transparency so the
choice of primary method is auditable, not hidden.

RecognizedExpense_t is the amount actually posted to that account for
period t in the VALIDATED ledger. The dataset does not model an artificial
invoice-arrival lag (every AP invoice is dated within the month it
belongs to -- see Phase 2), so accruals here reflect genuine month-to-month
estimation variance in a recurring cost, exactly matching the "expected
vs. recognized" formula above, not a fabricated reporting gap.

A 95% confidence interval on ExpectedExpense_t is the standard error of
the rolling-window mean (std / sqrt(n)) at 1.96 sigma -- how much
uncertainty there is in the *estimate itself* given how much that expense
has varied historically, not the raw spread of historical values.

Run: python -m src.provisions.accrual_model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.accounting.chart_of_accounts import join_coa
from src.accounting.currency import convert_to_eur
from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger

log = get_logger("accrual_model")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "accruals.csv"

BUSINESS_UNIT_CATEGORIES = ["Utilities", "Professional Services", "Logistics"]
CORPORATE_CATEGORIES = ["Interest Expense"]
ROLLING_WINDOW = 3
MIN_HISTORY_PERIODS = 2
Z_95 = 1.96


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(PROCESSED_DIR / "transactions.csv")
    coa = pd.read_csv(PROCESSED_DIR / "chart_of_accounts.csv")
    fx_rates = pd.read_csv(PROCESSED_DIR / "fx_rates.csv")
    return transactions, coa, fx_rates


def compute_monthly_expense(
    transactions: pd.DataFrame, coa: pd.DataFrame, fx_rates: pd.DataFrame, subcategory: str,
) -> pd.DataFrame:
    t = join_coa(transactions, coa)
    t = t[t["coa_subcategory"] == subcategory].copy()
    t["debit_eur"] = convert_to_eur(t["debit"], t["currency"], t["transaction_date"], fx_rates)
    t["period"] = pd.to_datetime(t["transaction_date"]).dt.to_period("M")

    has_bu = subcategory in BUSINESS_UNIT_CATEGORIES
    group_cols = ["entity_id", "business_unit", "period"] if has_bu else ["entity_id", "period"]
    monthly = t.groupby(group_cols)["debit_eur"].sum().reset_index()
    monthly = monthly.rename(columns={"debit_eur": "recognized_cost"})
    if not has_bu:
        monthly["business_unit"] = None
    monthly["expense_category"] = subcategory
    return monthly


def _estimate_series(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("period").reset_index(drop=True)
    values = group["recognized_cost"].to_numpy()
    months = group["period"].dt.month.to_numpy()

    rows = []
    for i in range(len(group)):
        if i < MIN_HISTORY_PERIODS:
            continue
        prior = values[:i]
        historical_average = float(np.mean(prior))

        same_month_prior = prior[months[:i] == months[i]]
        seasonal_average = float(np.mean(same_month_prior)) if len(same_month_prior) >= 1 else np.nan

        window = values[max(0, i - ROLLING_WINDOW):i]
        rolling_average = float(np.mean(window))
        rolling_std = float(np.std(window, ddof=1)) if len(window) >= 2 else np.nan
        n = len(window)
        se = rolling_std / np.sqrt(n) if n >= 2 else np.nan

        expected_cost = rolling_average
        recognized_cost = float(values[i])
        rows.append({
            "period": group.loc[i, "period"],
            "historical_average": historical_average,
            "seasonal_average": seasonal_average,
            "rolling_average": rolling_average,
            "expected_cost": expected_cost,
            "recognized_cost": recognized_cost,
            "estimated_accrual": expected_cost - recognized_cost,
            "ci_lower": expected_cost - Z_95 * se if n >= 2 else np.nan,
            "ci_upper": expected_cost + Z_95 * se if n >= 2 else np.nan,
            "n_periods_in_estimate": n,
        })
    return pd.DataFrame(rows)


def build_accruals(transactions: pd.DataFrame, coa: pd.DataFrame, fx_rates: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for subcategory in BUSINESS_UNIT_CATEGORIES + CORPORATE_CATEGORIES:
        monthly = compute_monthly_expense(transactions, coa, fx_rates, subcategory)
        group_keys = ["entity_id", "business_unit"] if subcategory in BUSINESS_UNIT_CATEGORIES else ["entity_id"]
        for keys, group in monthly.groupby(group_keys):
            keys = keys if isinstance(keys, tuple) else (keys,)
            estimated = _estimate_series(group)
            if estimated.empty:
                continue
            for col, val in zip(group_keys, keys):
                estimated[col] = val
            estimated["expense_category"] = subcategory
            if "business_unit" not in estimated.columns:
                estimated["business_unit"] = None
            all_rows.append(estimated)

    result = pd.concat(all_rows, ignore_index=True)
    ordered = [
        "entity_id", "business_unit", "expense_category", "period",
        "historical_average", "seasonal_average", "rolling_average", "expected_cost",
        "recognized_cost", "estimated_accrual", "ci_lower", "ci_upper", "n_periods_in_estimate",
    ]
    return result[ordered].sort_values(["expense_category", "entity_id", "business_unit", "period"]).reset_index(drop=True)


def main() -> None:
    transactions, coa, fx_rates = _load_processed()
    accruals = build_accruals(transactions, coa, fx_rates)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    accruals.to_csv(OUTPUT_PATH, index=False)

    log.info("Wrote %d accrual estimates to %s", len(accruals), OUTPUT_PATH.relative_to(PROJECT_ROOT))
    by_category = accruals.groupby("expense_category")["estimated_accrual"].agg(["mean", "std", "count"]).round(2)
    log.info("Accrual summary by category:\n%s", by_category)


if __name__ == "__main__":
    main()
