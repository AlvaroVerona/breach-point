"""Accounting reconciliation checks (§15 item 7, §19-21). Two levels:

1. Journal-entry balance: sum(debit) == sum(credit) per document_id. This
   is a transaction-level check, distinct from the Balance Sheet equation
   (Assets = Liabilities + Equity), which is a statement-level check built
   in Phase 4 (src/accounting/) on the VALIDATED layer -- see CLAUDE.md.
2. Subledger tie-out: does the AR/AP subledger total for an entity-month
   agree with what actually got posted to the Accounts Receivable /
   Accounts Payable GL account for that entity-month? A material gap is a
   genuine reconciliation failure signal, independent of any single
   row-level defect.
"""

from __future__ import annotations

import pandas as pd

ISSUE_COLUMNS = ["row_uid", "dataset", "validation_rule", "severity", "reason", "field"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ISSUE_COLUMNS)


def check_journal_balance(transactions: pd.DataFrame, tolerance: float, severity: str) -> pd.DataFrame:
    df = transactions.copy()
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0.0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0)
    balance = df.groupby("document_id")[["debit", "credit"]].sum()
    diff = (balance["debit"] - balance["credit"]).abs()
    broken_docs = diff[diff > tolerance].index
    if len(broken_docs) == 0:
        return _empty()

    mask = df["document_id"].isin(broken_docs)
    reason = "document_id " + df.loc[mask, "document_id"] + " does not balance (sum(debit) != sum(credit))"
    return pd.DataFrame({
        "row_uid": df.loc[mask, "row_uid"], "dataset": "transactions",
        "validation_rule": "journal_entry_unbalanced", "severity": severity,
        "reason": reason, "field": None,
    })


def check_subledger_tie_out(
    sub_df: pd.DataFrame, transactions: pd.DataFrame, dataset_name: str, amount_col: str,
    date_col: str, invoice_document_type: str, gl_account_name: str, gl_side: str,
    tolerance: float, severity: str,
) -> pd.DataFrame:
    sub = sub_df.copy()
    sub[amount_col] = pd.to_numeric(sub[amount_col], errors="coerce")
    sub[date_col] = pd.to_datetime(sub[date_col], format="%Y-%m-%d", errors="coerce")
    sub["_period"] = sub[date_col].dt.to_period("M")
    sub_totals = sub.groupby(["entity_id", "_period"])[amount_col].sum()

    gl = transactions[transactions["document_type"] == invoice_document_type].copy()
    gl = gl[gl["account_name"] == gl_account_name]
    gl[gl_side] = pd.to_numeric(gl[gl_side], errors="coerce").fillna(0.0)
    gl["transaction_date"] = pd.to_datetime(gl["transaction_date"], format="%Y-%m-%d", errors="coerce")
    gl["_period"] = gl["transaction_date"].dt.to_period("M")
    gl_totals = gl.groupby(["entity_id", "_period"])[gl_side].sum()

    combined = pd.concat({"subledger": sub_totals, "gl": gl_totals}, axis=1).fillna(0.0)
    combined["diff"] = (combined["subledger"] - combined["gl"]).abs()
    breaches = combined[combined["diff"] > tolerance]
    if breaches.empty:
        return _empty()

    rows = []
    for (entity_id, period), row in breaches.iterrows():
        rows.append({
            "row_uid": None, "dataset": dataset_name, "validation_rule": "subledger_tie_out_failure",
            "severity": severity,
            "reason": (
                f"{dataset_name} total for {entity_id}/{period} is {row['subledger']:.2f} but "
                f"{gl_account_name} postings for that entity-month total {row['gl']:.2f} "
                f"(diff {row['diff']:.2f})"
            ),
            "field": amount_col,
        })
    return pd.DataFrame(rows)


def check_reconciliation(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    severity_cfg = config["quality"]["severity"]
    tolerance = config["quality"]["tolerance_eur"]
    t, ar, ap = datasets["transactions"], datasets["accounts_receivable"], datasets["accounts_payable"]

    frames = [
        check_journal_balance(t, tolerance, severity_cfg["journal_entry_unbalanced"]),
        check_subledger_tie_out(
            ar, t, "accounts_receivable", "invoice_amount", "invoice_date", "Revenue Invoice",
            "Accounts Receivable", "debit", tolerance, severity_cfg["subledger_tie_out_failure"],
        ),
        check_subledger_tie_out(
            ap, t, "accounts_payable", "net_amount", "invoice_date", "Vendor Invoice",
            "Accounts Payable", "credit", tolerance, severity_cfg["subledger_tie_out_failure"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)
