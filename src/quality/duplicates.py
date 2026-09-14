"""Several levels of duplicate detection (§17): exact full-row duplicates,
business-key duplicates, duplicate invoice IDs, and a deliberately simple
near-duplicate heuristic (same party, close date, close amount) -- no fuzzy
string matching, per the project's "don't over-engineer this" instruction.
"""

from __future__ import annotations

import pandas as pd

ISSUE_COLUMNS = ["row_uid", "dataset", "validation_rule", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _issues_for_mask(df: pd.DataFrame, dataset: str, mask: pd.Series, rule: str, severity: str, reason: str) -> pd.DataFrame:
    if not mask.any():
        return _empty()
    return pd.DataFrame({
        "row_uid": df.loc[mask, "row_uid"], "dataset": dataset,
        "validation_rule": rule, "severity": severity, "reason": reason, "field": None,
    })


def check_exact_duplicates(df: pd.DataFrame, dataset: str, severity: str) -> pd.DataFrame:
    content_cols = [c for c in df.columns if c != "row_uid"]
    mask = df.duplicated(subset=content_cols, keep=False)
    return _issues_for_mask(df, dataset, mask, "duplicate_exact", severity,
                             "Full record is byte-identical to another record in the file")


def check_business_key_duplicates(df: pd.DataFrame, dataset: str, key_cols: list[str], severity: str) -> pd.DataFrame:
    present = [c for c in key_cols if c in df.columns]
    if len(present) < len(key_cols):
        return _empty()
    mask = df.duplicated(subset=present, keep=False) & df[present].notna().all(axis=1)
    return _issues_for_mask(df, dataset, mask, "duplicate_business_key", severity,
                             f"Same {', '.join(present)} combination appears on more than one record")


def check_duplicate_invoice_id(df: pd.DataFrame, dataset: str, severity: str) -> pd.DataFrame:
    if "invoice_id" not in df.columns:
        return _empty()
    mask = df["invoice_id"].notna() & df.duplicated(subset=["invoice_id"], keep=False)
    return _issues_for_mask(df, dataset, mask, "duplicate_invoice", severity,
                             "invoice_id appears on more than one record")


def check_near_duplicates(
    df: pd.DataFrame, dataset: str, party_col: str, amount_col: str, date_col: str, severity: str,
    amount_tolerance_pct: float = 0.005,
) -> pd.DataFrame:
    """Same counterparty, same calendar date, amount within 0.5%, but a
    *different* invoice_id -- catches a bill re-keyed under a new invoice
    number rather than a byte-identical duplicate. Deliberately strict
    (same day, tight tolerance): with only a few dozen vendors/customers
    and thousands of legitimately recurring invoices, a looser window
    mostly catches coincidence, not duplication."""
    work = df[["row_uid", "invoice_id", party_col, amount_col, date_col]].copy()
    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
    work[date_col] = pd.to_datetime(work[date_col], format="%Y-%m-%d", errors="coerce")
    work = work.dropna(subset=[party_col, amount_col, date_col])
    if work.empty:
        return _empty()

    flagged_uids: set[str] = set()
    for _, group in work.groupby([party_col, date_col]):
        if len(group) < 2:
            continue
        amounts = group[amount_col].to_numpy()
        for i in range(len(group)):
            close = (
                (group["invoice_id"].to_numpy() != group["invoice_id"].to_numpy()[i])
                & (abs(amounts - amounts[i]) <= amounts[i] * amount_tolerance_pct)
            )
            if close.any():
                flagged_uids.add(group["row_uid"].to_numpy()[i])

    mask = df["row_uid"].isin(flagged_uids)
    return _issues_for_mask(df, dataset, mask, "duplicate_near", severity,
                             f"Same {party_col} and {date_col}, {amount_col} within "
                             f"{amount_tolerance_pct:.1%} of another record under a different invoice_id")


def check_duplicates_transactions(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    return check_exact_duplicates(df, "transactions", severity_cfg["duplicate_exact"])


def check_duplicates_ar(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    frames = [
        check_exact_duplicates(df, "accounts_receivable", severity_cfg["duplicate_exact"]),
        check_duplicate_invoice_id(df, "accounts_receivable", severity_cfg["duplicate_invoice"]),
        check_business_key_duplicates(
            df, "accounts_receivable", ["customer_id", "invoice_date", "invoice_amount"],
            severity_cfg["duplicate_business_key"],
        ),
        check_near_duplicates(
            df, "accounts_receivable", "customer_id", "invoice_amount", "invoice_date",
            severity_cfg["duplicate_near"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def check_duplicates_ap(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    frames = [
        check_exact_duplicates(df, "accounts_payable", severity_cfg["duplicate_exact"]),
        check_duplicate_invoice_id(df, "accounts_payable", severity_cfg["duplicate_invoice"]),
        check_business_key_duplicates(
            df, "accounts_payable", ["vendor_id", "invoice_date", "invoice_amount"],
            severity_cfg["duplicate_business_key"],
        ),
        check_near_duplicates(
            df, "accounts_payable", "vendor_id", "invoice_amount", "invoice_date",
            severity_cfg["duplicate_near"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def check_duplicates(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_cfg = config["quality"]["severity"]
    frames = [
        check_duplicates_transactions(datasets["transactions"], severity_cfg),
        check_duplicates_ar(datasets["accounts_receivable"], severity_cfg),
        check_duplicates_ap(datasets["accounts_payable"], severity_cfg),
    ]
    return pd.concat(frames, ignore_index=True)
