"""Structural schema validation: does the raw file have the columns a
downstream consumer expects? This is intentionally coarse -- it does not
enforce non-null or value-level rules (that's completeness.py and
validity.py's job). A missing expected column is CRITICAL: everything
downstream assumes it exists."""

from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = {
    "transactions": [
        "transaction_id", "transaction_date", "entity_id", "business_unit",
        "account_id", "account_name", "account_category", "document_type",
        "document_id", "counterparty_id", "currency", "amount", "debit", "credit",
        "payment_due_date", "payment_date", "source_system",
    ],
    "accounts_receivable": [
        "invoice_id", "customer_id", "entity_id", "invoice_date", "due_date",
        "payment_date", "currency", "invoice_amount", "payment_status",
        "business_unit", "customer_segment",
    ],
    "accounts_payable": [
        "invoice_id", "vendor_id", "entity_id", "invoice_date", "due_date",
        "payment_date", "currency", "invoice_amount", "tax_amount", "net_amount",
        "payment_status", "business_unit", "expense_category",
    ],
    "entities": ["entity_id", "entity_name", "country", "functional_currency"],
    "chart_of_accounts": ["account_id", "account_name", "account_category", "normal_balance"],
    "vendors": ["vendor_id", "vendor_name", "entity_id", "currency", "payment_terms_days"],
    "customers": ["customer_id", "customer_name", "entity_id", "currency", "customer_segment"],
    "fx_rates": ["rate_date", "currency", "rate_to_eur"],
}


def check_schema(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Returns a dataset-level issues frame (one row per missing column),
    not row-level -- a missing column affects every row, not one."""
    expected = EXPECTED_COLUMNS.get(dataset_name, [])
    missing = [c for c in expected if c not in df.columns]
    if not missing:
        return pd.DataFrame(columns=["row_uid", "dataset", "validation_rule", "severity", "reason", "field"])
    return pd.DataFrame([{
        "row_uid": None, "dataset": dataset_name, "validation_rule": "schema_missing_column",
        "severity": "CRITICAL", "reason": f"Expected column '{col}' not found in {dataset_name}",
        "field": col,
    } for col in missing])
