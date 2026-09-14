"""Entry point for synthetic data generation.

Run: python -m src.data.generate_data

Produces data/raw/{transactions,accounts_payable,accounts_receivable,
entities,chart_of_accounts,vendors,customers,fx_rates}.csv, including
intentionally injected data-quality issues for the Phase 3 quality engine
to detect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, PROJECT_ROOT
from src.common.logging_config import get_logger
from src.data import master_data, financial_series, quality_injection
from src.data.ledger import LedgerBuilder, opening_balances
from src.data import transaction_events as events

log = get_logger("generate_data")

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _format_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    return df


def generate_all() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    seed = cfg["data"]["seed"]
    rng = np.random.default_rng(seed)
    start_date, end_date = cfg["data"]["start_date"], cfg["data"]["end_date"]
    end_ts = pd.Timestamp(end_date)

    log.info("Generating master data (seed=%s)", seed)
    entities = master_data.generate_entities()
    coa = master_data.generate_chart_of_accounts()
    vendors = master_data.generate_vendors(rng, entities)
    customers = master_data.generate_customers(rng, entities)
    fx_rates = master_data.generate_fx_rates(rng, start_date, end_date)
    log.info(
        "Master data: %d entities, %d accounts, %d vendors, %d customers, %d fx rows",
        len(entities), len(coa), len(vendors), len(customers), len(fx_rates),
    )

    log.info("Generating monthly financial series (revenue/expense ground truth)")
    monthly_series = financial_series.generate_monthly_series(rng, entities, start_date, end_date)
    log.info("Monthly series: %d target rows", len(monthly_series))

    log.info("Building double-entry ledger")
    builder = LedgerBuilder(rng, coa, start_date)
    opening = opening_balances(rng, entities)
    events.post_opening_balances(builder, opening)

    ar = events.generate_ar(rng, builder, monthly_series, customers, end_ts)
    ap_opex = events.generate_ap_opex(rng, builder, monthly_series, vendors, end_ts)
    ap_inventory = events.generate_inventory_and_cogs(rng, builder, monthly_series, vendors, end_ts)
    events.generate_salaries(builder, monthly_series, entities)
    months = pd.date_range(start_date, end_date, freq="MS")
    events.generate_depreciation_and_capex(rng, builder, entities, opening, months)
    events.generate_debt_interest(builder, entities, opening, months)

    ap = pd.concat([ap_opex, ap_inventory], ignore_index=True)
    transactions = builder.to_frame()
    log.info(
        "Clean ledger built: %d transaction lines, %d AR invoices, %d AP invoices",
        len(transactions), len(ar), len(ap),
    )

    log.info("Injecting data quality issues")
    dq = cfg["data"]
    mr = dq["missing_rate"]

    # Opening-balance documents seed each entity's entire Balance Sheet
    # (see ledger.opening_balances / CLAUDE.md) and are excluded from
    # injection entirely -- a real source system does not randomly corrupt
    # a controlled one-time ledger-migration entry, and if it did, the
    # quality engine's document-quarantine cascade (Phase 3) would remove
    # the whole entity's opening position, leaving every subsequent period
    # reconciled but economically meaningless (e.g. inventory or debt stuck
    # at exactly what accumulated from zero instead of the true opening
    # balance). That failure mode is realistic but not an interesting one
    # to demonstrate, so it's scoped out here rather than left to chance.
    is_opening = transactions["document_id"].str.startswith("OPEN_", na=False)
    opening_rows, transactions = transactions[is_opening], transactions[~is_opening].reset_index(drop=True)

    transactions = quality_injection.inject_exact_duplicates(transactions, rng, dq["duplicate_rate"])
    transactions = quality_injection.inject_missing_values(transactions, rng, {
        "transaction_id": mr["default"] / 4, "entity_id": mr["default"] / 4,
        "account_id": mr["default"] / 2, "currency": mr["currency"],
        "business_unit": mr["optional_metadata"], "counterparty_id": mr["optional_metadata"],
        "account_name": mr["optional_metadata"], "amount": mr["amount"],
    })
    transactions = quality_injection.inject_negative_amounts(transactions, rng, dq["invalid_rate"] / 4, "amount")
    transactions = quality_injection.inject_invalid_currency(transactions, rng, dq["invalid_rate"] / 4, "currency")
    transactions = quality_injection.inject_invalid_account(transactions, rng, dq["invalid_rate"] / 4)
    transactions = quality_injection.inject_inconsistent_account_category(transactions, rng, dq["invalid_rate"] / 2)
    transactions = quality_injection.inject_unbalanced_journal_entries(
        transactions, rng, dq["unbalanced_journal_rate"]
    )
    transactions = pd.concat([opening_rows, transactions], ignore_index=True)

    ar = quality_injection.inject_exact_duplicates(ar, rng, dq["duplicate_rate"])
    ar = quality_injection.inject_missing_values(ar, rng, {
        "customer_segment": mr["optional_metadata"], "business_unit": mr["optional_metadata"],
        "currency": mr["currency"], "invoice_amount": mr["amount"],
    })
    ar = quality_injection.inject_negative_amounts(ar, rng, dq["invalid_rate"] / 4, "invoice_amount")
    ar = quality_injection.inject_invalid_currency(ar, rng, dq["invalid_rate"] / 4, "currency")
    ar = quality_injection.inject_payment_before_document_date(
        ar, rng, dq["invalid_rate"] / 4, "invoice_date", "payment_date"
    )

    ap = quality_injection.inject_exact_duplicates(ap, rng, dq["duplicate_rate"])
    ap = quality_injection.inject_missing_values(ap, rng, {
        "expense_category": mr["optional_metadata"], "business_unit": mr["optional_metadata"],
        "currency": mr["currency"], "tax_amount": mr["optional_metadata"],
        "net_amount": mr["amount"], "invoice_amount": mr["amount"],
    })
    ap = quality_injection.inject_negative_amounts(ap, rng, dq["invalid_rate"] / 4, "invoice_amount")
    ap = quality_injection.inject_invalid_currency(ap, rng, dq["invalid_rate"] / 4, "currency")
    ap = quality_injection.inject_payment_before_document_date(
        ap, rng, dq["invalid_rate"] / 4, "invoice_date", "payment_date"
    )
    ap = quality_injection.inject_missing_periods(ap, rng, dq["missing_period_rate"], "invoice_date", "entity_id")

    log.info(
        "After injection: %d transaction lines, %d AR invoices, %d AP invoices",
        len(transactions), len(ar), len(ap),
    )

    transactions = _format_dates(transactions, ["transaction_date", "payment_due_date", "payment_date"])
    not_opening = ~transactions["document_id"].str.startswith("OPEN_", na=False)
    transactions.loc[not_opening, "transaction_date"] = quality_injection.inject_invalid_date_strings(
        transactions.loc[not_opening, "transaction_date"], rng, dq["invalid_rate"] / 8
    )
    ar = _format_dates(ar, ["invoice_date", "due_date", "payment_date"])
    ap = _format_dates(ap, ["invoice_date", "due_date", "payment_date"])
    ap["invoice_date"] = quality_injection.inject_invalid_date_strings(
        ap["invoice_date"], rng, dq["invalid_rate"] / 8
    )
    fx_rates = _format_dates(fx_rates, ["rate_date"])

    return {
        "transactions": transactions, "accounts_receivable": ar, "accounts_payable": ap,
        "entities": entities, "chart_of_accounts": coa, "vendors": vendors,
        "customers": customers, "fx_rates": fx_rates,
    }


def write_raw(datasets: dict[str, pd.DataFrame]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        path = RAW_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info("Wrote %s (%d rows)", path.relative_to(PROJECT_ROOT), len(df))


def main() -> None:
    datasets = generate_all()
    write_raw(datasets)
    log.info("Data generation complete.")


if __name__ == "__main__":
    main()
