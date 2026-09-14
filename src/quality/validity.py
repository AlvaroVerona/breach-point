"""Value-level validation rules (§18): amount consistency, date logic,
currency membership, disallowed negative amounts, syntactically invalid
dates, and statistical outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_CURRENCIES = {"EUR", "USD", "GBP"}
ISSUE_COLUMNS = ["row_uid", "dataset", "validation_rule", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _issues_for_mask(df: pd.DataFrame, dataset: str, mask: pd.Series, rule: str, severity: str, reason: str, field: str | None = None) -> pd.DataFrame:
    if not mask.any():
        return _empty()
    return pd.DataFrame({
        "row_uid": df.loc[mask, "row_uid"], "dataset": dataset,
        "validation_rule": rule, "severity": severity, "reason": reason, "field": field,
    })


def check_amount_consistency_ap(df: pd.DataFrame, tolerance: float, severity: str) -> pd.DataFrame:
    gross = pd.to_numeric(df["invoice_amount"], errors="coerce")
    net = pd.to_numeric(df["net_amount"], errors="coerce")
    tax = pd.to_numeric(df["tax_amount"], errors="coerce")
    all_present = gross.notna() & net.notna() & tax.notna()
    mismatch = all_present & ((gross - (net + tax)).abs() > tolerance)
    return _issues_for_mask(df, "accounts_payable", mismatch, "amount_mismatch", severity,
                             "invoice_amount does not equal net_amount + tax_amount",
                             field="invoice_amount")


def _to_datetime(series: pd.Series) -> pd.Series:
    """All dates are written to CSV as YYYY-MM-DD (see generate_data.py's
    _format_dates); pinning the format avoids per-row format inference and
    the resulting warning, while still coercing genuinely invalid strings
    (e.g. "2024-02-30") to NaT rather than raising."""
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


def check_date_order(df: pd.DataFrame, dataset: str, early_col: str, late_col: str, rule: str, severity: str, reason: str) -> pd.DataFrame:
    early = _to_datetime(df[early_col])
    late = _to_datetime(df[late_col])
    mask = early.notna() & late.notna() & (late < early)
    return _issues_for_mask(df, dataset, mask, rule, severity, reason, field=late_col)


def check_valid_currency(df: pd.DataFrame, dataset: str, severity: str, currency_col: str = "currency") -> pd.DataFrame:
    present = df[currency_col].notna()
    mask = present & ~df[currency_col].isin(VALID_CURRENCIES)
    return _issues_for_mask(df, dataset, mask, "invalid_currency", severity,
                             f"Currency not one of {sorted(VALID_CURRENCIES)}", field=currency_col)


def check_negative_amount(df: pd.DataFrame, dataset: str, amount_col: str, severity: str) -> pd.DataFrame:
    values = pd.to_numeric(df[amount_col], errors="coerce")
    mask = values.notna() & (values < 0)
    return _issues_for_mask(df, dataset, mask, "negative_amount_not_allowed", severity,
                             f"{amount_col} is negative, which is not a valid value for this field",
                             field=amount_col)


def check_invalid_dates(df: pd.DataFrame, dataset: str, date_cols: list[str], severity: str) -> pd.DataFrame:
    frames = []
    for col in date_cols:
        if col not in df.columns:
            continue
        raw_present = df[col].notna() & (df[col].astype(str).str.strip() != "")
        parsed = _to_datetime(df[col])
        mask = raw_present & parsed.isna()
        frames.append(_issues_for_mask(df, dataset, mask, "invalid_date", severity,
                                        f"'{col}' is not a syntactically valid calendar date", field=col))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def check_outliers(df: pd.DataFrame, dataset: str, amount_col: str, group_col: str, severity: str, z_threshold: float = 8.0) -> pd.DataFrame:
    """Robust (median/MAD-based) z-score outlier flag, computed within
    group_col so it compares like with like -- e.g. account_name (which
    encodes both line item and business unit) rather than the much coarser
    account_category, which would compare a small Travel expense against a
    large Salaries expense just because both are "Operating Expense"."""
    values = pd.to_numeric(df[amount_col], errors="coerce")
    flagged = pd.Series(False, index=df.index)
    groups = df.groupby(group_col).groups if group_col in df.columns else {"_all": df.index}
    for _, idx in groups.items():
        v = values.loc[idx].dropna()
        if len(v) < 20:
            continue
        median = v.median()
        mad = (v - median).abs().median()
        if mad == 0:
            continue
        z = 0.6745 * (v - median) / mad
        flagged.loc[v.index[z.abs() > z_threshold]] = True
    return _issues_for_mask(df, dataset, flagged, "statistical_outlier", severity,
                             f"{amount_col} is a statistical outlier relative to same-category records "
                             f"(robust z-score > {z_threshold})", field=amount_col)


def check_validity_transactions(df: pd.DataFrame, severity_cfg: dict, tolerance: float) -> pd.DataFrame:
    frames = [
        check_valid_currency(df, "transactions", severity_cfg["invalid_currency"]),
        check_negative_amount(df, "transactions", "amount", severity_cfg["negative_amount_not_allowed"]),
        check_invalid_dates(df, "transactions", ["transaction_date", "payment_due_date", "payment_date"],
                             severity_cfg["invalid_date"]),
        check_date_order(df, "transactions", "transaction_date", "payment_date", "payment_before_invoice",
                          severity_cfg["payment_before_invoice"],
                          "payment_date is earlier than transaction_date"),
        check_outliers(df, "transactions", "amount", "account_name", "MEDIUM"),
    ]
    return pd.concat(frames, ignore_index=True)


def check_validity_ar(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    frames = [
        check_valid_currency(df, "accounts_receivable", severity_cfg["invalid_currency"]),
        check_negative_amount(df, "accounts_receivable", "invoice_amount", severity_cfg["negative_amount_not_allowed"]),
        check_invalid_dates(df, "accounts_receivable", ["invoice_date", "due_date", "payment_date"],
                             severity_cfg["invalid_date"]),
        check_date_order(df, "accounts_receivable", "invoice_date", "due_date", "invalid_due_date",
                          severity_cfg["invalid_date"], "due_date is earlier than invoice_date"),
        check_date_order(df, "accounts_receivable", "invoice_date", "payment_date", "payment_before_invoice",
                          severity_cfg["payment_before_invoice"], "payment_date is earlier than invoice_date"),
        check_outliers(df, "accounts_receivable", "invoice_amount", "business_unit", "MEDIUM"),
    ]
    return pd.concat(frames, ignore_index=True)


def check_validity_ap(df: pd.DataFrame, severity_cfg: dict, tolerance: float) -> pd.DataFrame:
    frames = [
        check_valid_currency(df, "accounts_payable", severity_cfg["invalid_currency"]),
        check_negative_amount(df, "accounts_payable", "invoice_amount", severity_cfg["negative_amount_not_allowed"]),
        check_invalid_dates(df, "accounts_payable", ["invoice_date", "due_date", "payment_date"],
                             severity_cfg["invalid_date"]),
        check_date_order(df, "accounts_payable", "invoice_date", "due_date", "invalid_due_date",
                          severity_cfg["invalid_date"], "due_date is earlier than invoice_date"),
        check_date_order(df, "accounts_payable", "invoice_date", "payment_date", "payment_before_invoice",
                          severity_cfg["payment_before_invoice"], "payment_date is earlier than invoice_date"),
        check_amount_consistency_ap(df, tolerance, severity_cfg["amount_mismatch"]),
        check_outliers(df, "accounts_payable", "invoice_amount", "expense_category", "MEDIUM"),
    ]
    return pd.concat(frames, ignore_index=True)


def check_validity(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_cfg = config["quality"]["severity"]
    tolerance = config["quality"]["tolerance_eur"]
    frames = [
        check_validity_transactions(datasets["transactions"], severity_cfg, tolerance),
        check_validity_ar(datasets["accounts_receivable"], severity_cfg),
        check_validity_ap(datasets["accounts_payable"], severity_cfg, tolerance),
    ]
    return pd.concat(frames, ignore_index=True)
