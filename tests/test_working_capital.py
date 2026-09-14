"""Phase 6 tests: DSO/DPO/DIO/CCC arithmetic on a handcrafted fixture,
plus sanity checks on the real generated data (no missing values, the CCC
identity holds everywhere, the commentary is genuinely computed from the
data rather than templated)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.accounting.balance_sheet import build_balance_sheet
from src.accounting.income_statement import build_income_statement, _load_processed
from src.accounting.working_capital import build_working_capital, working_capital_commentary


def test_dso_dpo_dio_formulas_on_handcrafted_fixture():
    income_statement = pd.DataFrame({
        "entity_id": ["E1"], "period": [pd.Period("2024-01", freq="M")],
        "revenue": [300.0], "cogs": [150.0], "operating_expense": [50.0],
    })
    balance_sheet = pd.DataFrame({
        "entity_id": ["E1"], "period": [pd.Period("2024-01", freq="M")],
        "accounts_receivable": [100.0], "inventory": [50.0], "accounts_payable": [80.0],
    })
    wc = build_working_capital(income_statement, balance_sheet)
    days = 31  # January
    row = wc.iloc[0]
    assert row["dso"] == pytest.approx((100.0 / 300.0) * days)
    assert row["dpo"] == pytest.approx((80.0 / (150.0 + 50.0)) * days)
    assert row["dio"] == pytest.approx((50.0 / 150.0) * days)
    assert row["ccc"] == pytest.approx(row["dso"] + row["dio"] - row["dpo"])


def test_first_period_3m_avg_equals_own_value():
    income_statement = pd.DataFrame({
        "entity_id": ["E1"], "period": [pd.Period("2024-01", freq="M")],
        "revenue": [300.0], "cogs": [150.0], "operating_expense": [50.0],
    })
    balance_sheet = pd.DataFrame({
        "entity_id": ["E1"], "period": [pd.Period("2024-01", freq="M")],
        "accounts_receivable": [100.0], "inventory": [50.0], "accounts_payable": [80.0],
    })
    wc = build_working_capital(income_statement, balance_sheet)
    assert wc.iloc[0]["dso_3m_avg"] == pytest.approx(wc.iloc[0]["dso"])


@pytest.fixture(scope="module")
def working_capital():
    transactions, coa, fx_rates = _load_processed()
    income_statement = build_income_statement(transactions, coa, fx_rates)
    balance_sheet = build_balance_sheet(transactions, coa, fx_rates, income_statement)
    return build_working_capital(income_statement, balance_sheet)


def test_ccc_identity_holds_everywhere(working_capital):
    diff = (working_capital["ccc"] - (working_capital["dso"] + working_capital["dio"] - working_capital["dpo"])).abs()
    assert diff.max() < 1e-6


def test_no_missing_core_metrics(working_capital):
    core = ["dso", "dpo", "dio", "ccc", "dso_3m_avg", "dpo_3m_avg", "dio_3m_avg", "ccc_3m_avg"]
    assert not working_capital[core].isna().any().any()


def test_covers_every_entity_month(working_capital):
    assert len(working_capital) == 3 * 36


def test_commentary_is_computed_from_data_not_templated(working_capital):
    notes = working_capital_commentary(working_capital)
    assert len(notes) == working_capital["entity_id"].nunique() + 1  # one per entity + the CCC-max summary

    first_entity = working_capital["entity_id"].iloc[0]
    entity_group = working_capital[working_capital["entity_id"] == first_entity].sort_values("period")
    expected_first_dso = f"{entity_group.iloc[0]['dso_3m_avg']:.0f}"
    matching_note = next(n for n in notes if n.startswith(first_entity))
    assert expected_first_dso in matching_note
