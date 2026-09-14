"""Builds a monthly Income Statement per entity from the VALIDATED ledger
(data/processed/transactions.csv), in EUR.

Revenue
COGS
Gross Profit
Operating Expense
EBITDA
Depreciation
EBIT
Interest
Taxes
Net Income

Run: python -m src.accounting.income_statement
"""

from __future__ import annotations

import pandas as pd

from src.accounting.chart_of_accounts import (
    CORPORATE_TAX_RATE_BY_ENTITY, classify_pl_bucket, join_coa, signed_amount,
)
from src.accounting.currency import convert_to_eur
from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger

log = get_logger("income_statement")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "income_statement.csv"


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(PROCESSED_DIR / "transactions.csv")
    coa = pd.read_csv(PROCESSED_DIR / "chart_of_accounts.csv")
    fx_rates = pd.read_csv(PROCESSED_DIR / "fx_rates.csv")
    return transactions, coa, fx_rates


def build_income_statement(
    transactions: pd.DataFrame, coa: pd.DataFrame, fx_rates: pd.DataFrame,
    tax_rate_by_entity: dict[str, float] = CORPORATE_TAX_RATE_BY_ENTITY,
) -> pd.DataFrame:
    t = join_coa(transactions, coa)
    t = t[t["document_type"] != "Opening Balance"].copy()

    t["debit_eur"] = convert_to_eur(t["debit"], t["currency"], t["transaction_date"], fx_rates)
    t["credit_eur"] = convert_to_eur(t["credit"], t["currency"], t["transaction_date"], fx_rates)
    t["contribution"] = signed_amount(t["debit_eur"], t["credit_eur"], t["coa_normal_balance"])
    t["pl_bucket"] = [classify_pl_bucket(cat, sub) for cat, sub in zip(t["coa_category"], t["coa_subcategory"])]
    t["period"] = pd.to_datetime(t["transaction_date"]).dt.to_period("M")

    pl = t.dropna(subset=["pl_bucket"])
    monthly = pl.groupby(["entity_id", "period", "pl_bucket"])["contribution"].sum().unstack(fill_value=0.0)
    for col in ["Revenue", "COGS", "Operating Expense", "Depreciation", "Interest", "Tax"]:
        if col not in monthly.columns:
            monthly[col] = 0.0

    stmt = monthly.reset_index()
    stmt["gross_profit"] = stmt["Revenue"] - stmt["COGS"]
    stmt["ebitda"] = stmt["gross_profit"] - stmt["Operating Expense"]
    stmt["ebit"] = stmt["ebitda"] - stmt["Depreciation"]
    stmt["pretax_income"] = stmt["ebit"] - stmt["Interest"]
    stmt["tax_rate"] = stmt["entity_id"].map(tax_rate_by_entity).fillna(0.25)
    stmt["taxes"] = stmt["pretax_income"].clip(lower=0) * stmt["tax_rate"]
    stmt["net_income"] = stmt["pretax_income"] - stmt["taxes"]

    stmt = stmt.rename(columns={
        "Revenue": "revenue", "COGS": "cogs", "Operating Expense": "operating_expense",
        "Depreciation": "depreciation", "Interest": "interest",
    }).drop(columns=["Tax", "tax_rate"])

    ordered_cols = [
        "entity_id", "period", "revenue", "cogs", "gross_profit", "operating_expense",
        "ebitda", "depreciation", "ebit", "interest", "pretax_income", "taxes", "net_income",
    ]
    return stmt[ordered_cols].sort_values(["entity_id", "period"]).reset_index(drop=True)


def main() -> None:
    transactions, coa, fx_rates = _load_processed()
    stmt = build_income_statement(transactions, coa, fx_rates)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stmt.to_csv(OUTPUT_PATH, index=False)
    log.info("Wrote income statement (%d entity-months) to %s", len(stmt), OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info(
        "Consolidated net income by entity: %s",
        stmt.groupby("entity_id")["net_income"].sum().round(0).to_dict(),
    )


if __name__ == "__main__":
    main()
