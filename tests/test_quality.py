"""Phase 3 tests: each check function against small handcrafted fixtures
(precise, fast), plus an integration test of the full quality engine
against the real generated data (RAW = VALIDATED + QUARANTINED, quarantine
schema, score bounds)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.config import load_config
from src.data.ingestion import load_all_raw
from src.quality import completeness, consistency, duplicates, reconciliation, validity
from src.quality.quality_score import (
    QUARANTINE_SEVERITIES, add_document_cascade_issues, build_quarantine_ledger,
    compute_quality_score, run_all_checks, run_quality_engine, split_validated_and_quarantine,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_missing_transaction_id_is_critical(config):
    df = pd.DataFrame({
        "row_uid": ["T1", "T2"], "transaction_id": [None, "TXN_1"],
        "entity_id": ["ENT_EU", "ENT_EU"], "account_id": ["1000", "1000"],
        "amount": [10.0, 20.0], "currency": ["EUR", "EUR"],
        "business_unit": ["Sales", "Sales"], "account_name": ["Cash", "Cash"],
        "counterparty_id": [None, None], "document_type": ["Payroll", "Payroll"],
        "payment_date": [None, None],
    })
    issues = completeness.check_completeness_transactions(df, config["quality"]["severity"])
    hit = issues[(issues["row_uid"] == "T1") & (issues["validation_rule"] == "missing_transaction_id")]
    assert len(hit) == 1
    assert hit.iloc[0]["severity"] == "CRITICAL"


def test_unpaid_invoice_missing_payment_date_is_low_not_flagged_as_paid_inconsistency(config):
    df = pd.DataFrame({
        "row_uid": ["A1"], "invoice_id": ["INV1"], "entity_id": ["ENT_EU"],
        "customer_id": ["CUST_0001"], "invoice_amount": [100.0], "currency": ["EUR"],
        "business_unit": ["Sales"], "customer_segment": ["SMB"],
        "payment_status": ["Open"], "payment_date": [None],
    })
    issues = completeness.check_completeness_ar(df, config["quality"]["severity"])
    rules = set(issues["validation_rule"])
    assert "missing_payment_date_unpaid" in rules
    assert "missing_payment_date_inconsistent" not in rules


def test_paid_invoice_missing_payment_date_is_flagged_high(config):
    df = pd.DataFrame({
        "row_uid": ["A1"], "invoice_id": ["INV1"], "entity_id": ["ENT_EU"],
        "customer_id": ["CUST_0001"], "invoice_amount": [100.0], "currency": ["EUR"],
        "business_unit": ["Sales"], "customer_segment": ["SMB"],
        "payment_status": ["Paid"], "payment_date": [None],
    })
    issues = completeness.check_completeness_ar(df, config["quality"]["severity"])
    hit = issues[issues["validation_rule"] == "missing_payment_date_inconsistent"]
    assert len(hit) == 1
    assert hit.iloc[0]["severity"] == config["quality"]["severity"]["missing_payment_date_inconsistent"]


def test_exact_duplicate_detected():
    df = pd.DataFrame({
        "row_uid": ["R1", "R2", "R3"],
        "a": [1, 1, 2], "b": ["x", "x", "y"],
    })
    issues = duplicates.check_exact_duplicates(df, "dummy", "HIGH")
    assert set(issues["row_uid"]) == {"R1", "R2"}


def test_duplicate_invoice_id_detected():
    df = pd.DataFrame({"row_uid": ["R1", "R2", "R3"], "invoice_id": ["INV1", "INV1", "INV2"]})
    issues = duplicates.check_duplicate_invoice_id(df, "dummy", "HIGH")
    assert set(issues["row_uid"]) == {"R1", "R2"}


def test_invalid_currency_detected():
    df = pd.DataFrame({"row_uid": ["R1", "R2"], "currency": ["XYZ", "EUR"]})
    issues = validity.check_valid_currency(df, "dummy", "HIGH")
    assert list(issues["row_uid"]) == ["R1"]


def test_negative_amount_detected():
    df = pd.DataFrame({"row_uid": ["R1", "R2"], "amount": [-5.0, 5.0]})
    issues = validity.check_negative_amount(df, "dummy", "amount", "MEDIUM")
    assert list(issues["row_uid"]) == ["R1"]


def test_invalid_date_string_detected():
    df = pd.DataFrame({"row_uid": ["R1", "R2"], "d": ["2024-02-30", "2024-02-15"]})
    issues = validity.check_invalid_dates(df, "dummy", ["d"], "HIGH")
    assert list(issues["row_uid"]) == ["R1"]


def test_payment_before_invoice_date_detected():
    df = pd.DataFrame({
        "row_uid": ["R1", "R2"],
        "invoice_date": ["2024-02-15", "2024-02-15"],
        "payment_date": ["2024-02-10", "2024-02-20"],
    })
    issues = validity.check_date_order(df, "dummy", "invoice_date", "payment_date", "rule", "HIGH", "reason")
    assert list(issues["row_uid"]) == ["R1"]


def test_amount_consistency_ap_detected():
    df = pd.DataFrame({
        "row_uid": ["R1", "R2"], "invoice_amount": [119.0, 119.0],
        "net_amount": [100.0, 100.0], "tax_amount": [19.0, 10.0],
    })
    issues = validity.check_amount_consistency_ap(df, tolerance=0.01, severity="MEDIUM")
    assert list(issues["row_uid"]) == ["R2"]


def test_inconsistent_account_category_detected():
    transactions = pd.DataFrame({
        "row_uid": ["R1", "R2"], "account_id": ["4000", "4000"],
        "account_category": ["Operating Expense", "Revenue"],
    })
    coa = pd.DataFrame({"account_id": ["4000"], "account_category": ["Revenue"]})
    issues = consistency.check_account_category_consistency(transactions, coa, "HIGH")
    assert list(issues["row_uid"]) == ["R1"]


def test_referential_integrity_detected():
    df = pd.DataFrame({"row_uid": ["R1", "R2"], "account_id": ["9999", "4000"]})
    issues = consistency.check_referential_integrity(df, "dummy", "account_id", {"4000"}, "HIGH")
    assert list(issues["row_uid"]) == ["R1"]


def test_journal_balance_detects_unbalanced_document():
    df = pd.DataFrame({
        "row_uid": ["R1", "R2", "R3", "R4"],
        "document_id": ["D1", "D1", "D2", "D2"],
        "debit": [100.0, 0.0, 50.0, 0.0], "credit": [0.0, 90.0, 0.0, 50.0],
    })
    issues = reconciliation.check_journal_balance(df, tolerance=0.01, severity="CRITICAL")
    assert set(issues["row_uid"]) == {"R1", "R2"}


@pytest.fixture(scope="module")
def real_datasets():
    return load_all_raw()


@pytest.fixture(scope="module")
def real_issues(real_datasets, config):
    issues = run_all_checks(real_datasets, config)
    return add_document_cascade_issues(real_datasets["transactions"], issues)


def test_run_all_checks_on_real_data(real_issues):
    assert len(real_issues) > 0
    assert set(real_issues["severity"]) <= {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
    row_issues = real_issues[real_issues["row_uid"].notna()]
    assert row_issues["dataset"].isin(["transactions", "accounts_receivable", "accounts_payable"]).all()


def test_split_partitions_without_loss_or_duplication(real_datasets, real_issues):
    validated, quarantined = split_validated_and_quarantine(real_datasets, real_issues)
    for name in ["transactions", "accounts_receivable", "accounts_payable"]:
        assert len(validated[name]) + len(quarantined[name]) == len(real_datasets[name])
        assert set(validated[name]["row_uid"]).isdisjoint(set(quarantined[name]["row_uid"]))


def test_quarantine_ledger_schema(real_datasets, real_issues):
    _, quarantined = split_validated_and_quarantine(real_datasets, real_issues)
    ledgers = build_quarantine_ledger(quarantined, real_issues)
    expected_cols = {"record_id", "validation_rule", "severity", "reason", "timestamp"}
    for name, ledger in ledgers.items():
        assert expected_cols <= set(ledger.columns)
        if len(ledger) > 0:
            assert ledger["severity"].isin(QUARANTINE_SEVERITIES).all()


def test_quality_score_within_bounds(real_datasets, real_issues, config):
    score = compute_quality_score(real_issues, real_datasets, config)
    assert 0 <= score["overall_score"] <= 100
    for dim_score in score["dimension_scores"].values():
        assert 0 <= dim_score <= 100


def test_quality_score_not_hardcoded_differs_with_fewer_issues(real_datasets, real_issues, config):
    full_score = compute_quality_score(real_issues, real_datasets, config)
    half_issues = real_issues.iloc[: len(real_issues) // 4]
    partial_score = compute_quality_score(half_issues, real_datasets, config)
    assert partial_score["overall_score"] != full_score["overall_score"]


def test_validated_transactions_have_no_unbalanced_documents(real_datasets, real_issues):
    """Regression test for the asymmetric-quarantine bug: a document whose
    only broken leg was flagged for an unrelated reason must be removed as
    a whole, not left as an orphaned single leg in the validated layer."""
    validated, _ = split_validated_and_quarantine(real_datasets, real_issues)
    t = validated["transactions"].copy()
    t["debit"] = pd.to_numeric(t["debit"], errors="coerce").fillna(0.0)
    t["credit"] = pd.to_numeric(t["credit"], errors="coerce").fillna(0.0)
    balance = t.groupby("document_id")[["debit", "credit"]].sum()
    assert ((balance["debit"] - balance["credit"]).abs() < 0.01).all()


def test_document_cascade_quarantines_orphaned_sibling_leg():
    transactions = pd.DataFrame({
        "row_uid": ["T1", "T2", "T3", "T4"],
        "document_id": ["D1", "D1", "D2", "D2"],
    })
    # T1 has an unrelated CRITICAL issue; T2 (its balanced sibling) does not.
    issues = pd.DataFrame({
        "row_uid": ["T1"], "dataset": ["transactions"],
        "validation_rule": ["invalid_currency"], "severity": ["CRITICAL"],
        "reason": ["bad currency"], "field": ["currency"], "dimension": ["validity"],
    })
    augmented = add_document_cascade_issues(transactions, issues)
    quarantined_uids = set(augmented.loc[augmented["severity"] == "CRITICAL", "row_uid"])
    assert quarantined_uids == {"T1", "T2"}
    assert "T3" not in quarantined_uids and "T4" not in quarantined_uids


def test_full_quality_engine_run_produces_processed_and_quarantine_files():
    report = run_quality_engine()
    assert "overall_score" in report
    from src.quality.quality_score import PROCESSED_DIR, QUARANTINE_DIR
    for name in ["transactions", "accounts_receivable", "accounts_payable"]:
        assert (PROCESSED_DIR / f"{name}.csv").exists()
        assert (QUARANTINE_DIR / f"{name}.csv").exists()
