"""Classifies GL accounts into statement line items. Uses the chart of
accounts as the authoritative source of account_category/account_subcategory
-- not the (possibly still slightly noisy) denormalized fields carried on
each transaction row -- since every row surviving quarantine is already
guaranteed to have a consistent account_id, but re-deriving from the COA
join is one less thing to trust."""

from __future__ import annotations

import pandas as pd

PL_DEPRECIATION_SUBCATEGORY = "Depreciation Expense"
PL_INTEREST_SUBCATEGORY = "Interest Expense"
PL_TAX_SUBCATEGORY = "Tax Expense"

BALANCE_SHEET_CURRENT_ASSETS = ["Cash", "Accounts Receivable", "Inventory", "Other Current Assets"]
BALANCE_SHEET_FIXED_ASSETS = ["Fixed Assets"]
BALANCE_SHEET_CONTRA_ASSETS = ["Accumulated Depreciation"]
BALANCE_SHEET_LIABILITIES = ["Accounts Payable", "Debt", "Other Liabilities"]
BALANCE_SHEET_EQUITY = ["Common Stock", "Retained Earnings"]

# Corporate income tax is not journaled at the transaction level in Phase 2
# (no explicit tax-provisioning entries exist in the ledger); it is
# estimated here, at statement-build time, as a flat statutory rate on
# pre-tax income. This is a formula-driven estimate, not a fabricated
# number -- documented explicitly per the project's "never fabricate a
# result" principle.
CORPORATE_TAX_RATE_BY_ENTITY = {"ENT_EU": 0.30, "ENT_US": 0.25, "ENT_UK": 0.25}


def classify_pl_bucket(account_category: str, account_subcategory: str) -> str | None:
    if account_category == "Revenue":
        return "Revenue"
    if account_category == "COGS":
        return "COGS"
    if account_category == "Operating Expense":
        if account_subcategory == PL_DEPRECIATION_SUBCATEGORY:
            return "Depreciation"
        if account_subcategory == PL_INTEREST_SUBCATEGORY:
            return "Interest"
        if account_subcategory == PL_TAX_SUBCATEGORY:
            return "Tax"
        return "Operating Expense"
    return None


def join_coa(transactions: pd.DataFrame, coa: pd.DataFrame) -> pd.DataFrame:
    """Left-joins the authoritative account_category/account_subcategory/
    normal_balance from the chart of accounts onto each transaction row."""
    lookup = coa[["account_id", "account_category", "account_subcategory", "normal_balance"]]
    merged = transactions.merge(lookup, on="account_id", how="left", suffixes=("", "_coa"))
    return merged.rename(columns={
        "account_category_coa": "coa_category",
        "account_subcategory": "coa_subcategory",
        "normal_balance": "coa_normal_balance",
    })


def signed_amount(debit_eur: pd.Series, credit_eur: pd.Series, normal_balance: pd.Series) -> pd.Series:
    """Returns each row's contribution to its account's balance, positive
    when the account moves in its natural direction (a Debit-normal asset
    increasing on a debit, a Credit-normal liability increasing on a
    credit)."""
    return (debit_eur - credit_eur).where(normal_balance == "Debit", credit_eur - debit_eur)
