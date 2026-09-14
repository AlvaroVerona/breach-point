"""Master data generation: entities, chart of accounts, vendors, customers,
and FX rates. This is the reference data that referential-integrity checks
in the quality engine validate transactional data against."""

from __future__ import annotations

import numpy as np
import pandas as pd

BUSINESS_UNITS = ["Sales", "Operations", "Marketing", "R&D", "Corporate"]
REVENUE_BUSINESS_UNITS = ["Sales", "Operations", "Marketing"]

REVENUE_TYPES = ["Product Revenue", "Service Revenue", "Subscription Revenue"]
COGS_TYPES = ["Cost of Goods Sold", "Cost of Services"]
FIXED_OPEX_TYPES = [
    "Salaries", "Rent", "Insurance", "Software Subscriptions",
    "Professional Services", "Utilities",
]
VARIABLE_OPEX_TYPES = [
    "Marketing", "Logistics", "Raw Materials", "Sales Commissions",
    "Travel", "IT Infrastructure",
]
CORPORATE_PL_ACCOUNTS = [
    "Depreciation Expense", "Interest Expense", "Bank Fees",
    "FX Gain/Loss", "Tax Expense", "Audit Fees", "Legal Fees",
    "Bad Debt Expense",
]
BALANCE_SHEET_ACCOUNTS = [
    ("Cash", "Asset", "Cash", "Debit"),
    ("Accounts Receivable", "Asset", "Accounts Receivable", "Debit"),
    ("Inventory", "Asset", "Inventory", "Debit"),
    ("Other Current Assets", "Asset", "Other Current Assets", "Debit"),
    ("Fixed Assets - Equipment", "Asset", "Fixed Assets", "Debit"),
    ("Fixed Assets - Buildings", "Asset", "Fixed Assets", "Debit"),
    ("Accumulated Depreciation", "Asset", "Accumulated Depreciation", "Credit"),
    ("Accounts Payable", "Liability", "Accounts Payable", "Credit"),
    ("Accrued Liabilities", "Liability", "Other Liabilities", "Credit"),
    ("Short-term Debt", "Liability", "Debt", "Credit"),
    ("Long-term Debt", "Liability", "Debt", "Credit"),
    ("Other Liabilities", "Liability", "Other Liabilities", "Credit"),
    ("Common Stock", "Equity", "Common Stock", "Credit"),
    ("Retained Earnings", "Equity", "Retained Earnings", "Credit"),
]


def generate_entities() -> pd.DataFrame:
    return pd.DataFrame([
        {"entity_id": "ENT_EU", "entity_name": "Breach Point Europe GmbH",
         "country": "DE", "functional_currency": "EUR",
         "annual_revenue_scale_eur": 8_000_000},
        {"entity_id": "ENT_US", "entity_name": "Breach Point Inc.",
         "country": "US", "functional_currency": "USD",
         "annual_revenue_scale_eur": 9_300_000},
        {"entity_id": "ENT_UK", "entity_name": "Breach Point UK Ltd.",
         "country": "GB", "functional_currency": "GBP",
         "annual_revenue_scale_eur": 5_800_000},
    ])


def generate_chart_of_accounts() -> pd.DataFrame:
    rows = []
    account_id = 4000
    for acc_type in REVENUE_TYPES:
        for bu in BUSINESS_UNITS:
            rows.append((account_id, f"{acc_type} - {bu}", "Revenue", acc_type, "Credit", bu))
            account_id += 1

    account_id = 5000
    for acc_type in COGS_TYPES:
        for bu in BUSINESS_UNITS:
            rows.append((account_id, f"{acc_type} - {bu}", "COGS", acc_type, "Debit", bu))
            account_id += 1

    account_id = 6000
    for acc_type in FIXED_OPEX_TYPES + VARIABLE_OPEX_TYPES:
        for bu in BUSINESS_UNITS:
            rows.append((account_id, f"{acc_type} Expense - {bu}", "Operating Expense", acc_type, "Debit", bu))
            account_id += 1

    account_id = 6900
    for acc_type in CORPORATE_PL_ACCOUNTS:
        rows.append((account_id, acc_type, "Operating Expense", acc_type, "Debit", None))
        account_id += 1

    account_id = 1000
    liab_start, equity_start = 2000, 3000
    for name, category, subcat, normal_balance in BALANCE_SHEET_ACCOUNTS:
        if category == "Asset":
            aid = account_id
            account_id += 1
        elif category == "Liability":
            aid = liab_start
            liab_start += 1
        else:
            aid = equity_start
            equity_start += 1
        rows.append((aid, name, category, subcat, normal_balance, None))

    coa = pd.DataFrame(rows, columns=[
        "account_id", "account_name", "account_category",
        "account_subcategory", "normal_balance", "expected_business_unit",
    ])
    coa["account_id"] = coa["account_id"].astype(str)
    return coa.sort_values("account_id").reset_index(drop=True)


def generate_vendors(rng: np.random.Generator, entities: pd.DataFrame, n_vendors: int = 45) -> pd.DataFrame:
    entity_ids = entities["entity_id"].to_numpy()
    entity_ccy = dict(zip(entities["entity_id"], entities["functional_currency"]))
    terms = rng.choice([15, 30, 45, 60], size=n_vendors, p=[0.15, 0.5, 0.2, 0.15])
    assigned_entity = rng.choice(entity_ids, size=n_vendors)
    cross_currency = rng.random(n_vendors) < 0.15
    currencies = np.array([entity_ccy[e] for e in assigned_entity])
    other_ccy = rng.choice(["EUR", "USD", "GBP"], size=n_vendors)
    currencies = np.where(cross_currency, other_ccy, currencies)
    categories = rng.choice(
        VARIABLE_OPEX_TYPES + ["Rent", "Insurance", "Software Subscriptions", "Professional Services", "Utilities"],
        size=n_vendors,
    )
    return pd.DataFrame({
        "vendor_id": [f"VEND_{i:04d}" for i in range(1, n_vendors + 1)],
        "vendor_name": [f"Vendor {i} {c.split()[0]}" for i, c in enumerate(categories, start=1)],
        "entity_id": assigned_entity,
        "currency": currencies,
        "payment_terms_days": terms,
        "default_expense_category": categories,
    })


def generate_customers(rng: np.random.Generator, entities: pd.DataFrame, n_customers: int = 180) -> pd.DataFrame:
    entity_ids = entities["entity_id"].to_numpy()
    entity_ccy = dict(zip(entities["entity_id"], entities["functional_currency"]))
    assigned_entity = rng.choice(entity_ids, size=n_customers)
    cross_currency = rng.random(n_customers) < 0.10
    currencies = np.array([entity_ccy[e] for e in assigned_entity])
    other_ccy = rng.choice(["EUR", "USD", "GBP"], size=n_customers)
    currencies = np.where(cross_currency, other_ccy, currencies)
    segment = rng.choice(["Enterprise", "SMB", "Retail"], size=n_customers, p=[0.2, 0.45, 0.35])
    terms = np.select(
        [segment == "Enterprise", segment == "SMB", segment == "Retail"],
        [60, 30, 15],
    )
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(1, n_customers + 1)],
        "customer_name": [f"Customer {i}" for i in range(1, n_customers + 1)],
        "entity_id": assigned_entity,
        "currency": currencies,
        "customer_segment": segment,
        "payment_terms_days": terms,
    })


def generate_fx_rates(rng: np.random.Generator, start_date: str, end_date: str) -> pd.DataFrame:
    """Monthly EUR-per-1-unit-of-currency rates via a bounded random walk
    around a realistic 2023-2025 baseline. EUR is trivially 1.0."""
    months = pd.date_range(start_date, end_date, freq="MS")
    baselines = {"EUR": 1.00, "USD": 0.92, "GBP": 1.16}
    vol = {"EUR": 0.0, "USD": 0.015, "GBP": 0.012}
    rows = []
    for ccy, base in baselines.items():
        rate = base
        for month in months:
            if ccy != "EUR":
                rate = rate * (1 + rng.normal(0, vol[ccy]))
                rate = float(np.clip(rate, base * 0.85, base * 1.15))
            rows.append({"rate_date": month, "currency": ccy, "rate_to_eur": round(rate, 5)})
    return pd.DataFrame(rows)
