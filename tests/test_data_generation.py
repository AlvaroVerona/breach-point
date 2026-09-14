"""Phase 2 tests: master data, the monthly generative series, ledger
double-entry integrity on clean (pre-injection) data, and the full
generate_data pipeline (columns, row counts, reproducibility)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import financial_series, master_data
from src.data import transaction_events as events
from src.data.generate_data import generate_all
from src.data.ledger import LedgerBuilder, opening_balances

SEED = 42


@pytest.fixture(scope="module")
def entities():
    return master_data.generate_entities()


@pytest.fixture(scope="module")
def coa():
    return master_data.generate_chart_of_accounts()


def test_entities_shape(entities):
    assert len(entities) == 3
    assert set(entities["entity_id"]) == {"ENT_EU", "ENT_US", "ENT_UK"}
    assert set(entities.columns) >= {"entity_id", "entity_name", "country", "functional_currency"}


def test_chart_of_accounts_has_100_plus_accounts(coa):
    assert len(coa) >= 100
    assert coa["account_id"].is_unique
    assert set(coa["account_category"]) <= {"Revenue", "COGS", "Operating Expense", "Asset", "Liability", "Equity"}


def test_vendors_and_customers_valid(entities):
    rng = np.random.default_rng(SEED)
    vendors = master_data.generate_vendors(rng, entities, n_vendors=20)
    customers = master_data.generate_customers(rng, entities, n_customers=30)
    assert vendors["vendor_id"].is_unique
    assert customers["customer_id"].is_unique
    assert set(vendors["entity_id"]) <= set(entities["entity_id"])
    assert set(customers["entity_id"]) <= set(entities["entity_id"])


def test_fx_rates_eur_is_one():
    rng = np.random.default_rng(SEED)
    fx = master_data.generate_fx_rates(rng, "2023-01-01", "2023-03-01")
    eur_rates = fx.loc[fx["currency"] == "EUR", "rate_to_eur"]
    assert (eur_rates == 1.0).all()
    assert set(fx["currency"]) == {"EUR", "USD", "GBP"}


def test_monthly_series_nonnegative(entities):
    rng = np.random.default_rng(SEED)
    series = financial_series.generate_monthly_series(rng, entities, "2023-01-01", "2023-06-01")
    assert len(series) > 0
    assert (series["target_amount"] >= 0).all()
    assert set(series["account_category"]) <= {"Revenue", "COGS", "Operating Expense"}


def test_small_ledger_documents_balance(coa):
    """Builds a minimal one-entity, one-month ledger through the same code
    path as the full pipeline (pre-injection) and checks that every
    journal document balances exactly -- the property injected corruption
    (Phase 2's quality_injection) is designed to selectively break."""
    rng = np.random.default_rng(SEED)
    entities = master_data.generate_entities().iloc[[0]].reset_index(drop=True)
    vendors = master_data.generate_vendors(rng, entities, n_vendors=10)
    customers = master_data.generate_customers(rng, entities, n_customers=15)
    start_date, end_date = "2023-01-01", "2023-01-01"
    monthly_series = financial_series.generate_monthly_series(rng, entities, start_date, end_date)

    builder = LedgerBuilder(rng, coa, start_date)
    opening = opening_balances(rng, entities)
    events.post_opening_balances(builder, opening)

    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=60)
    events.generate_ar(rng, builder, monthly_series, customers, end_ts)
    events.generate_ap_opex(rng, builder, monthly_series, vendors, end_ts)
    events.generate_inventory_and_cogs(rng, builder, monthly_series, vendors, end_ts)
    events.generate_salaries(builder, monthly_series, entities)

    transactions = builder.to_frame()
    assert len(transactions) > 0

    balance = transactions.groupby("document_id")[["debit", "credit"]].sum()
    diff = (balance["debit"] - balance["credit"]).abs()
    assert (diff < 0.01).all(), f"{(diff >= 0.01).sum()} documents do not balance"


def test_small_ledger_has_valid_ids(coa):
    """Every ID the clean (pre-injection) ledger produces should actually
    resolve: transaction_id is unique, account_id exists in the chart of
    accounts, entity_id matches the entity it was generated for. Injected
    invalid IDs (Phase 2's quality_injection) are a separate, deliberate
    concern tested in test_quality.py -- this covers the ledger builder
    itself, before any corruption is applied."""
    rng = np.random.default_rng(SEED)
    entities = master_data.generate_entities().iloc[[0]].reset_index(drop=True)
    vendors = master_data.generate_vendors(rng, entities, n_vendors=10)
    customers = master_data.generate_customers(rng, entities, n_customers=15)
    start_date, end_date = "2023-01-01", "2023-01-01"
    monthly_series = financial_series.generate_monthly_series(rng, entities, start_date, end_date)

    builder = LedgerBuilder(rng, coa, start_date)
    opening = opening_balances(rng, entities)
    events.post_opening_balances(builder, opening)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=60)
    events.generate_ar(rng, builder, monthly_series, customers, end_ts)
    events.generate_ap_opex(rng, builder, monthly_series, vendors, end_ts)
    transactions = builder.to_frame()

    assert transactions["transaction_id"].is_unique
    assert transactions["transaction_id"].notna().all()
    valid_accounts = set(coa["account_id"])
    assert set(transactions["account_id"]) <= valid_accounts
    assert set(transactions["entity_id"]) == {"ENT_EU"}


@pytest.fixture(scope="module")
def generated_datasets():
    return generate_all()


def test_generate_all_expected_columns(generated_datasets):
    expected_transaction_cols = {
        "transaction_id", "transaction_date", "entity_id", "business_unit",
        "account_id", "account_name", "account_category", "document_type",
        "document_id", "counterparty_id", "currency", "amount", "debit", "credit",
        "payment_due_date", "payment_date", "source_system",
    }
    expected_ar_cols = {
        "invoice_id", "customer_id", "entity_id", "invoice_date", "due_date",
        "payment_date", "currency", "invoice_amount", "payment_status",
        "business_unit", "customer_segment",
    }
    expected_ap_cols = {
        "invoice_id", "vendor_id", "entity_id", "invoice_date", "due_date",
        "payment_date", "currency", "invoice_amount", "tax_amount", "net_amount",
        "payment_status", "business_unit", "expense_category",
    }
    assert expected_transaction_cols <= set(generated_datasets["transactions"].columns)
    assert expected_ar_cols <= set(generated_datasets["accounts_receivable"].columns)
    assert expected_ap_cols <= set(generated_datasets["accounts_payable"].columns)


def test_generate_all_row_counts(generated_datasets):
    assert len(generated_datasets["transactions"]) >= 100_000
    assert len(generated_datasets["accounts_receivable"]) > 0
    assert len(generated_datasets["accounts_payable"]) > 0


def test_generate_all_reproducible():
    run1 = generate_all()
    run2 = generate_all()
    pd.testing.assert_frame_equal(
        run1["transactions"].reset_index(drop=True),
        run2["transactions"].reset_index(drop=True),
    )
