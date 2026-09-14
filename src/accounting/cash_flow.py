"""Builds a monthly (direct-method) Cash Flow Statement per entity from the
VALIDATED ledger, in EUR: Operating / Investing / Financing CF classified
by document_type, independently of the Balance Sheet's cumulative Cash
balance -- so comparing "Beginning Cash + Net Change = Ending Cash" against
the Balance Sheet's own cash figure is a genuine reconciliation check, not
a tautology derived from the same computation.

Financing CF is 0 historically: the generated ledger has no debt issuance/
repayment transactions (only interest on a constant opening debt balance --
see CLAUDE.md/ledger.py). Forward-looking financing decisions (new
borrowing) are a Phase 9 optimization lever, not part of these actuals.

Run: python -m src.accounting.cash_flow
"""

from __future__ import annotations

import pandas as pd

from src.accounting.chart_of_accounts import join_coa
from src.accounting.currency import convert_to_eur
from src.accounting.balance_sheet import build_balance_sheet
from src.accounting.income_statement import build_income_statement, _load_processed
from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger

log = get_logger("cash_flow")

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "cash_flow.csv"

OPERATING_DOCUMENT_TYPES = {"Cash Receipt", "Vendor Payment", "Payroll", "Interest Payment"}
INVESTING_DOCUMENT_TYPES = {"Capex"}
FINANCING_DOCUMENT_TYPES: set[str] = set()


def _opening_cash_by_entity(transactions: pd.DataFrame, coa: pd.DataFrame, fx_rates: pd.DataFrame) -> pd.Series:
    t = join_coa(transactions, coa)
    opening = t[t["document_id"].str.startswith("OPEN_", na=False) & (t["coa_subcategory"] == "Cash")].copy()
    opening["debit_eur"] = convert_to_eur(opening["debit"], opening["currency"], opening["transaction_date"], fx_rates)
    return opening.groupby("entity_id")["debit_eur"].sum()


def build_cash_flow(
    transactions: pd.DataFrame, coa: pd.DataFrame, fx_rates: pd.DataFrame,
    balance_sheet: pd.DataFrame, tolerance: float = 0.01,
) -> pd.DataFrame:
    t = join_coa(transactions, coa)
    t = t[(t["coa_subcategory"] == "Cash") & (t["document_type"] != "Opening Balance")].copy()

    t["debit_eur"] = convert_to_eur(t["debit"], t["currency"], t["transaction_date"], fx_rates)
    t["credit_eur"] = convert_to_eur(t["credit"], t["currency"], t["transaction_date"], fx_rates)
    t["cash_flow_eur"] = t["debit_eur"] - t["credit_eur"]
    t["period"] = pd.to_datetime(t["transaction_date"]).dt.to_period("M")

    def _section(doc_type: str) -> str:
        if doc_type in OPERATING_DOCUMENT_TYPES:
            return "operating_cf"
        if doc_type in INVESTING_DOCUMENT_TYPES:
            return "investing_cf"
        if doc_type in FINANCING_DOCUMENT_TYPES:
            return "financing_cf"
        return "other_cf"

    t["section"] = t["document_type"].map(_section)
    monthly = t.groupby(["entity_id", "period", "section"])["cash_flow_eur"].sum().unstack(fill_value=0.0)
    for col in ["operating_cf", "investing_cf", "financing_cf"]:
        if col not in monthly.columns:
            monthly[col] = 0.0

    cf = monthly.reset_index().sort_values(["entity_id", "period"])
    cf["net_change_in_cash"] = cf["operating_cf"] + cf["investing_cf"] + cf["financing_cf"]

    opening_cash = _opening_cash_by_entity(transactions, coa, fx_rates)
    bs_cash = balance_sheet.set_index(["entity_id", "period"])["cash"]

    beginning_cash = []
    for entity_id, group in cf.groupby("entity_id"):
        prior_ending = opening_cash.get(entity_id, 0.0)
        for period in group["period"]:
            beginning_cash.append(prior_ending)
            prior_ending = bs_cash.get((entity_id, period), prior_ending)
    cf["beginning_cash"] = beginning_cash
    cf["ending_cash"] = cf["beginning_cash"] + cf["net_change_in_cash"]

    cf = cf.merge(balance_sheet[["entity_id", "period", "cash"]].rename(columns={"cash": "balance_sheet_cash"}),
                  on=["entity_id", "period"], how="left")
    cf["reconciliation_diff"] = (cf["ending_cash"] - cf["balance_sheet_cash"]).abs()
    cf["status"] = (cf["reconciliation_diff"] <= tolerance).map({True: "PASS", False: "FAIL"})

    ordered = [
        "entity_id", "period", "operating_cf", "investing_cf", "financing_cf",
        "net_change_in_cash", "beginning_cash", "ending_cash", "balance_sheet_cash",
        "reconciliation_diff", "status",
    ]
    return cf[ordered].reset_index(drop=True)


def main() -> None:
    transactions, coa, fx_rates = _load_processed()
    income_statement = build_income_statement(transactions, coa, fx_rates)
    bs = build_balance_sheet(transactions, coa, fx_rates, income_statement)
    cf = build_cash_flow(transactions, coa, fx_rates, bs)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cf.to_csv(OUTPUT_PATH, index=False)

    n_fail = (cf["status"] == "FAIL").sum()
    log.info("Wrote cash flow statement (%d entity-months) to %s", len(cf), OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info("Cash flow reconciliation: %d/%d entity-months PASS", len(cf) - n_fail, len(cf))
    if n_fail:
        log.warning("%d entity-months FAIL to reconcile -- see status column", n_fail)


if __name__ == "__main__":
    main()
