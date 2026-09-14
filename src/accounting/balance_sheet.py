"""Builds a monthly Balance Sheet per entity from the VALIDATED ledger, in
EUR: cumulative account balances since inception (the opening entry is
part of that same cumulative sum, dated at the start of the window).

Retained Earnings is NOT read as a cumulative GL balance -- the ledger
posts it once, at opening, and never again (see CLAUDE.md: no closing
entries). It is reported here as opening Retained Earnings + cumulative Net
Income from the Income Statement, which is how retained earnings actually
works; Assets = Liabilities + Equity holds as a result of that, not despite
it, whenever every document is complete and balanced.

Run: python -m src.accounting.balance_sheet
"""

from __future__ import annotations

import pandas as pd

from src.accounting.chart_of_accounts import join_coa, signed_amount
from src.accounting.currency import convert_to_eur
from src.accounting.income_statement import build_income_statement, _load_processed
from src.common.config import PROJECT_ROOT, load_config
from src.common.logging_config import get_logger

log = get_logger("balance_sheet")

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "balance_sheet.csv"

CURRENT_ASSET_COLS = ["Cash", "Accounts Receivable", "Inventory", "Other Current Assets"]
LIABILITY_COLS = ["Accounts Payable", "Debt", "Other Liabilities"]


def build_balance_sheet(
    transactions: pd.DataFrame, coa: pd.DataFrame, fx_rates: pd.DataFrame,
    income_statement: pd.DataFrame, tolerance: float = 0.01,
) -> pd.DataFrame:
    t = join_coa(transactions, coa)
    t = t[t["coa_category"].isin(["Asset", "Liability", "Equity"])].copy()

    t["debit_eur"] = convert_to_eur(t["debit"], t["currency"], t["transaction_date"], fx_rates)
    t["credit_eur"] = convert_to_eur(t["credit"], t["currency"], t["transaction_date"], fx_rates)
    t["contribution"] = signed_amount(t["debit_eur"], t["credit_eur"], t["coa_normal_balance"])
    t["period"] = pd.to_datetime(t["transaction_date"]).dt.to_period("M")

    monthly_flow = t.groupby(["entity_id", "period", "coa_subcategory"])["contribution"].sum()
    wide = monthly_flow.unstack(fill_value=0.0)

    all_periods = income_statement["period"].unique()
    entities = wide.index.get_level_values("entity_id").unique()
    full_index = pd.MultiIndex.from_product([entities, sorted(all_periods)], names=["entity_id", "period"])
    wide = wide.reindex(full_index, fill_value=0.0)

    for col in ["Cash", "Accounts Receivable", "Inventory", "Other Current Assets", "Fixed Assets",
                "Accumulated Depreciation", "Accounts Payable", "Debt", "Other Liabilities",
                "Common Stock", "Retained Earnings"]:
        if col not in wide.columns:
            wide[col] = 0.0

    balances = wide.groupby(level="entity_id").cumsum().reset_index()

    ni = income_statement[["entity_id", "period", "net_income", "taxes"]].copy()
    ni["cumulative_net_income"] = ni.groupby("entity_id")["net_income"].cumsum()
    ni["income_tax_payable"] = ni.groupby("entity_id")["taxes"].cumsum()
    balances = balances.merge(
        ni[["entity_id", "period", "cumulative_net_income", "income_tax_payable"]],
        on=["entity_id", "period"], how="left",
    )
    balances["cumulative_net_income"] = balances["cumulative_net_income"].fillna(0.0)
    balances["income_tax_payable"] = balances["income_tax_payable"].fillna(0.0)

    opening_retained_earnings = balances.groupby("entity_id")["Retained Earnings"].transform("first")
    balances["retained_earnings"] = opening_retained_earnings + balances["cumulative_net_income"]

    balances["total_current_assets"] = balances[CURRENT_ASSET_COLS].sum(axis=1)
    balances["fixed_assets_net"] = balances["Fixed Assets"] - balances["Accumulated Depreciation"]
    balances["total_assets"] = balances["total_current_assets"] + balances["fixed_assets_net"]
    # Income tax is estimated at statement-build time (see chart_of_accounts.py)
    # rather than journaled in the ledger, so its offsetting liability has to
    # be added explicitly here -- otherwise the tax deduction lowers Retained
    # Earnings with no matching liability, and the Balance Sheet never
    # reconciles. This is standard accrual treatment: tax expensed but not
    # yet paid in cash is a payable, not a cash outflow.
    balances["total_liabilities"] = balances[LIABILITY_COLS].sum(axis=1) + balances["income_tax_payable"]
    balances["total_equity"] = balances["Common Stock"] + balances["retained_earnings"]
    balances["difference"] = balances["total_assets"] - (balances["total_liabilities"] + balances["total_equity"])
    balances["status"] = (balances["difference"].abs() <= tolerance).map({True: "PASS", False: "FAIL"})

    # The Balance Sheet equation reconciling (status == PASS) is a purely
    # arithmetic property of double-entry bookkeeping and holds regardless.
    # A negative balance on an account that can never legitimately be
    # negative (you cannot hold negative inventory or negative gross fixed
    # assets) is a *different*, economic-plausibility problem: quarantine
    # operates per-document, not per underlying economic relationship, so
    # it can remove more purchase-side records than consumption-side ones
    # for the same asset and push a derived balance negative even though
    # both sides of every remaining document still balance. Surfaced
    # explicitly rather than silently floored (which would itself break
    # the equation) or hidden.
    non_negative_cols = ["Cash", "Accounts Receivable", "Inventory", "Other Current Assets", "Fixed Assets"]
    balances["economically_implausible"] = (balances[non_negative_cols] < 0).any(axis=1)

    balances = balances.rename(columns={
        "Cash": "cash", "Accounts Receivable": "accounts_receivable", "Inventory": "inventory",
        "Other Current Assets": "other_current_assets", "Fixed Assets": "fixed_assets_gross",
        "Accumulated Depreciation": "accumulated_depreciation", "Accounts Payable": "accounts_payable",
        "Debt": "debt", "Other Liabilities": "other_liabilities", "Common Stock": "common_stock",
    }).drop(columns=["Retained Earnings", "cumulative_net_income"])

    ordered = [
        "entity_id", "period", "cash", "accounts_receivable", "inventory", "other_current_assets",
        "total_current_assets", "fixed_assets_gross", "accumulated_depreciation", "fixed_assets_net",
        "total_assets", "accounts_payable", "debt", "other_liabilities", "income_tax_payable",
        "total_liabilities", "common_stock", "retained_earnings", "total_equity", "difference", "status",
        "economically_implausible",
    ]
    return balances[ordered].sort_values(["entity_id", "period"]).reset_index(drop=True)


def main() -> None:
    transactions, coa, fx_rates = _load_processed()
    income_statement = build_income_statement(transactions, coa, fx_rates)
    bs = build_balance_sheet(transactions, coa, fx_rates, income_statement)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bs.to_csv(OUTPUT_PATH, index=False)

    n_fail = (bs["status"] == "FAIL").sum()
    log.info("Wrote balance sheet (%d entity-months) to %s", len(bs), OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info("Balance Sheet reconciliation: %d/%d entity-months PASS", len(bs) - n_fail, len(bs))
    if n_fail:
        log.warning("%d entity-months FAIL to reconcile -- see status column", n_fail)

    n_implausible = bs["economically_implausible"].sum()
    if n_implausible:
        affected = bs.loc[bs["economically_implausible"], "entity_id"].unique().tolist()
        log.warning(
            "%d entity-months have a negative balance on an account that should never be negative "
            "(entities affected: %s) -- the equation still reconciles; see "
            "'economically_implausible' column and CLAUDE.md", n_implausible, affected,
        )


if __name__ == "__main__":
    main()
