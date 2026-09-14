"""Phase 4 tests: Income Statement / Balance Sheet / Cash Flow, both on
small handcrafted fixtures (precise arithmetic) and as an integration check
against the real VALIDATED data (Balance Sheet reconciles, Cash Flow ties
to the Balance Sheet's own cash figure, Net Income is internally
consistent between statements)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.accounting.balance_sheet import build_balance_sheet
from src.accounting.cash_flow import build_cash_flow
from src.accounting.chart_of_accounts import classify_pl_bucket, signed_amount
from src.accounting.currency import convert_to_eur
from src.accounting.income_statement import build_income_statement, _load_processed


def test_classify_pl_bucket():
    assert classify_pl_bucket("Revenue", "Product Revenue") == "Revenue"
    assert classify_pl_bucket("COGS", "Cost of Goods Sold") == "COGS"
    assert classify_pl_bucket("Operating Expense", "Salaries") == "Operating Expense"
    assert classify_pl_bucket("Operating Expense", "Depreciation Expense") == "Depreciation"
    assert classify_pl_bucket("Operating Expense", "Interest Expense") == "Interest"
    assert classify_pl_bucket("Asset", "Cash") is None


def test_signed_amount_respects_normal_balance():
    debit = pd.Series([100.0, 0.0])
    credit = pd.Series([0.0, 100.0])
    normal_balance = pd.Series(["Debit", "Credit"])
    result = signed_amount(debit, credit, normal_balance)
    assert list(result) == [100.0, 100.0]


def test_convert_to_eur():
    fx_rates = pd.DataFrame({
        "rate_date": ["2024-01-01", "2024-01-01"], "currency": ["EUR", "USD"], "rate_to_eur": [1.0, 0.9],
    })
    amount = pd.Series([100.0, 100.0])
    currency = pd.Series(["EUR", "USD"])
    date = pd.Series(["2024-01-15", "2024-01-20"])
    result = convert_to_eur(amount, currency, date, fx_rates)
    assert list(result) == [100.0, 90.0]


@pytest.fixture(scope="module")
def processed():
    return _load_processed()


@pytest.fixture(scope="module")
def income_statement(processed):
    transactions, coa, fx_rates = processed
    return build_income_statement(transactions, coa, fx_rates)


@pytest.fixture(scope="module")
def balance_sheet(processed, income_statement):
    transactions, coa, fx_rates = processed
    return build_balance_sheet(transactions, coa, fx_rates, income_statement)


@pytest.fixture(scope="module")
def cash_flow(processed, balance_sheet):
    transactions, coa, fx_rates = processed
    return build_cash_flow(transactions, coa, fx_rates, balance_sheet)


def test_income_statement_covers_every_entity_month(income_statement):
    assert len(income_statement) == 3 * 36
    assert (income_statement["revenue"] > 0).all()
    assert (income_statement["cogs"] >= 0).all()


def test_income_statement_arithmetic_identities(income_statement):
    s = income_statement
    assert (s["gross_profit"] - (s["revenue"] - s["cogs"])).abs().max() < 0.01
    assert (s["ebitda"] - (s["gross_profit"] - s["operating_expense"])).abs().max() < 0.01
    assert (s["ebit"] - (s["ebitda"] - s["depreciation"])).abs().max() < 0.01
    assert (s["net_income"] - (s["pretax_income"] - s["taxes"])).abs().max() < 0.01
    assert (s["taxes"] >= 0).all()


def test_balance_sheet_reconciles(balance_sheet):
    assert (balance_sheet["status"] == "PASS").all()
    assert balance_sheet["difference"].abs().max() < 0.01


def test_balance_sheet_flags_economic_implausibility_without_breaking_equation(balance_sheet):
    # The equation must hold even for the entity-months flagged implausible
    # (see CLAUDE.md) -- the flag is informational, not a reconciliation failure.
    assert (balance_sheet["status"] == "PASS").all()
    assert "economically_implausible" in balance_sheet.columns


def test_cash_flow_reconciles_to_balance_sheet(cash_flow):
    assert (cash_flow["status"] == "PASS").all()
    assert cash_flow["reconciliation_diff"].abs().max() < 0.01


def test_cash_flow_financing_is_zero_and_documented(cash_flow):
    # No debt issuance/repayment transactions exist in the historical
    # ledger (see cash_flow.py docstring / CLAUDE.md) -- financing is a
    # forward-looking optimization lever, not part of these actuals.
    assert (cash_flow["financing_cf"] == 0).all()


def test_balance_sheet_opening_documents_survive_quarantine(processed):
    """Regression test: opening-balance journal entries must never be
    individually corrupted by injection (see generate_data.py) -- if one
    were quarantined, the whole entity's Balance Sheet would restart from
    zero instead of its true opening position."""
    transactions, _, _ = processed
    for entity_id in transactions["entity_id"].dropna().unique():
        doc_id = f"OPEN_{entity_id}"
        rows = transactions[transactions["document_id"] == doc_id]
        assert len(rows) == 13, f"{doc_id} has {len(rows)} surviving legs, expected 13"
