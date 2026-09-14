"""Consistency and referential integrity checks (§13, §18 'Account'/
'Entity' rules): does the row's own account_category agree with what the
chart of accounts says for that account_id, and do foreign keys
(account_id, entity_id, vendor_id, customer_id) actually resolve against
master data?"""

from __future__ import annotations

import pandas as pd

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


def check_referential_integrity(df: pd.DataFrame, dataset: str, col: str, valid_values: set, severity: str) -> pd.DataFrame:
    if col not in df.columns:
        return _empty()
    present = df[col].notna()
    mask = present & ~df[col].isin(valid_values)
    return _issues_for_mask(df, dataset, mask, f"invalid_{col}", severity,
                             f"{col} does not exist in master data", field=col)


def check_account_category_consistency(transactions: pd.DataFrame, coa: pd.DataFrame, severity: str) -> pd.DataFrame:
    truth = coa.set_index("account_id")["account_category"]
    known_account = transactions["account_id"].isin(truth.index)
    true_category = transactions.loc[known_account, "account_id"].map(truth)
    stated_category = transactions.loc[known_account, "account_category"]
    mismatch = known_account.copy()
    mismatch.loc[known_account] = (
        stated_category.notna() & true_category.notna() & (stated_category != true_category)
    )
    return _issues_for_mask(transactions, "transactions", mismatch, "inconsistent_classification", severity,
                             "account_category on the record does not match the chart of accounts for this "
                             "account_id", field="account_category")


def check_consistency(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_cfg = config["quality"]["severity"]
    t, ar, ap = datasets["transactions"], datasets["accounts_receivable"], datasets["accounts_payable"]
    coa, entities = datasets["chart_of_accounts"], datasets["entities"]
    vendors, customers = datasets["vendors"], datasets["customers"]

    frames = [
        check_referential_integrity(t, "transactions", "account_id", set(coa["account_id"]), severity_cfg["invalid_account"]),
        check_referential_integrity(t, "transactions", "entity_id", set(entities["entity_id"]), severity_cfg["invalid_entity"]),
        check_account_category_consistency(t, coa, severity_cfg["inconsistent_classification"]),
        check_referential_integrity(ar, "accounts_receivable", "entity_id", set(entities["entity_id"]), severity_cfg["invalid_entity"]),
        check_referential_integrity(ar, "accounts_receivable", "customer_id", set(customers["customer_id"]), severity_cfg["invalid_account"]),
        check_referential_integrity(ap, "accounts_payable", "entity_id", set(entities["entity_id"]), severity_cfg["invalid_entity"]),
        check_referential_integrity(ap, "accounts_payable", "vendor_id", set(vendors["vendor_id"]), severity_cfg["invalid_account"]),
    ]
    return pd.concat(frames, ignore_index=True)
