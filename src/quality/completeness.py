"""Completeness checks: required fields must not be null, with severity and
business-context that varies by field -- a missing transaction_id is not
the same problem as a missing payment_date on an invoice nobody expects to
be paid yet (§16 of the project spec)."""

from __future__ import annotations

import pandas as pd

ISSUE_COLUMNS = ["row_uid", "dataset", "validation_rule", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def _missing_field_issues(df: pd.DataFrame, dataset: str, field: str, severity: str, rule_suffix: str = "") -> pd.DataFrame:
    if field not in df.columns:
        return _empty()
    mask = df[field].isna() | (df[field].astype(str).str.strip() == "")
    if not mask.any():
        return _empty()
    rule = f"missing_{field}{rule_suffix}"
    return pd.DataFrame({
        "row_uid": df.loc[mask, "row_uid"],
        "dataset": dataset,
        "validation_rule": rule,
        "severity": severity,
        "reason": f"Required field '{field}' is missing",
        "field": field,
    })


def check_completeness_transactions(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    issues = [
        _missing_field_issues(df, "transactions", "transaction_id", severity_cfg["missing_transaction_id"]),
        _missing_field_issues(df, "transactions", "entity_id", severity_cfg["missing_entity_id"]),
        _missing_field_issues(df, "transactions", "account_id", severity_cfg["missing_account_id"]),
        _missing_field_issues(df, "transactions", "amount", severity_cfg["missing_amount"]),
        _missing_field_issues(df, "transactions", "currency", severity_cfg["missing_currency"]),
        _missing_field_issues(df, "transactions", "business_unit", severity_cfg["missing_optional_metadata"]),
        _missing_field_issues(df, "transactions", "account_name", severity_cfg["missing_optional_metadata"]),
        _missing_field_issues(df, "transactions", "counterparty_id", severity_cfg["missing_optional_metadata"]),
    ]

    invoice_types = {"Revenue Invoice", "Vendor Invoice"}
    payment_leg_types = {"Cash Receipt", "Vendor Payment"}
    payment_missing = df["payment_date"].isna() | (df["payment_date"].astype(str).str.strip() == "")

    legit_unpaid = payment_missing & df["document_type"].isin(invoice_types)
    if legit_unpaid.any():
        issues.append(pd.DataFrame({
            "row_uid": df.loc[legit_unpaid, "row_uid"], "dataset": "transactions",
            "validation_rule": "missing_payment_date_unpaid",
            "severity": severity_cfg["missing_payment_date_unpaid"],
            "reason": "payment_date missing on an invoice line -- consistent with it still being open",
            "field": "payment_date",
        }))

    should_have_payment = payment_missing & df["document_type"].isin(payment_leg_types)
    if should_have_payment.any():
        reasons = "payment_date missing on a " + df.loc[should_have_payment, "document_type"] + \
            " line, which is itself the payment event"
        issues.append(pd.DataFrame({
            "row_uid": df.loc[should_have_payment, "row_uid"], "dataset": "transactions",
            "validation_rule": "missing_payment_date_inconsistent",
            "severity": severity_cfg["missing_payment_date_inconsistent"],
            "reason": reasons,
            "field": "payment_date",
        }))

    return pd.concat(issues, ignore_index=True) if issues else _empty()


def _check_completeness_subledger(df: pd.DataFrame, dataset: str, severity_cfg: dict, party_field: str, category_field: str) -> pd.DataFrame:
    issues = [
        _missing_field_issues(df, dataset, "invoice_id", severity_cfg["missing_transaction_id"]),
        _missing_field_issues(df, dataset, "entity_id", severity_cfg["missing_entity_id"]),
        _missing_field_issues(df, dataset, party_field, severity_cfg["missing_entity_id"]),
        _missing_field_issues(df, dataset, "invoice_amount", severity_cfg["missing_amount"]),
        _missing_field_issues(df, dataset, "currency", severity_cfg["missing_currency"]),
        _missing_field_issues(df, dataset, "business_unit", severity_cfg["missing_optional_metadata"]),
        _missing_field_issues(df, dataset, category_field, severity_cfg["missing_optional_metadata"]),
    ]

    payment_missing = df["payment_date"].isna() | (df["payment_date"].astype(str).str.strip() == "")
    is_paid = df["payment_status"] == "Paid"

    unpaid_missing = payment_missing & ~is_paid
    if unpaid_missing.any():
        issues.append(pd.DataFrame({
            "row_uid": df.loc[unpaid_missing, "row_uid"], "dataset": dataset,
            "validation_rule": "missing_payment_date_unpaid",
            "severity": severity_cfg["missing_payment_date_unpaid"],
            "reason": "payment_date missing, consistent with the invoice not being paid yet",
            "field": "payment_date",
        }))

    paid_missing = payment_missing & is_paid
    if paid_missing.any():
        issues.append(pd.DataFrame({
            "row_uid": df.loc[paid_missing, "row_uid"], "dataset": dataset,
            "validation_rule": "missing_payment_date_inconsistent",
            "severity": severity_cfg["missing_payment_date_inconsistent"],
            "reason": "payment_date missing despite payment_status='Paid'",
            "field": "payment_date",
        }))

    return pd.concat(issues, ignore_index=True) if issues else _empty()


def check_completeness_ar(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    return _check_completeness_subledger(df, "accounts_receivable", severity_cfg, "customer_id", "customer_segment")


def check_completeness_ap(df: pd.DataFrame, severity_cfg: dict) -> pd.DataFrame:
    return _check_completeness_subledger(df, "accounts_payable", severity_cfg, "vendor_id", "expense_category")


def check_completeness(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_cfg = config["quality"]["severity"]
    frames = [
        check_completeness_transactions(datasets["transactions"], severity_cfg),
        check_completeness_ar(datasets["accounts_receivable"], severity_cfg),
        check_completeness_ap(datasets["accounts_payable"], severity_cfg),
    ]
    return pd.concat(frames, ignore_index=True)
