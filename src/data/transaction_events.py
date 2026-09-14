"""Turns monthly financial targets (financial_series.py) into individual
GL transactions, accounts_receivable.csv rows and accounts_payable.csv
rows, via the double-entry LedgerBuilder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ledger import LedgerBuilder, _random_dates_in_month, _split_amount, TAX_RATE_BY_ENTITY

AR_INVOICE_COUNT_RANGE = (10, 28)
AP_INVOICE_COUNT_RANGE = (2, 6)
# Fraction of invoices that are genuinely disputed/stuck rather than just
# slow, independent of the normal timing distribution below. AR's residual
# is cleaned up via bad-debt write-off (a customer can genuinely default);
# AP's stays intentionally much smaller since a going-concern company
# eventually pays its vendors -- with no write-off equivalent on the payables
# side, a higher rate would leave AP (and DPO) growing without bound.
COLLECTION_PROBABILITY = 0.97
AP_PAYMENT_PROBABILITY = 0.99
CAPEX_QUARTER_PCT_RANGE = (0.010, 0.022)
BAD_DEBT_WRITEOFF_DAYS_PAST_DUE = 120


def _segment_offset(rng: np.random.Generator, segment: str, n: int) -> np.ndarray:
    params = {
        "Enterprise": (5, 10), "SMB": (10, 15), "Retail": (-2, 8),
    }
    mean, sd = params[segment]
    return rng.normal(mean, sd, size=n)


def post_opening_balances(builder: LedgerBuilder, opening: pd.DataFrame) -> None:
    for _, row in opening.iterrows():
        entity_id = row["entity_id"]
        ccy = row["functional_currency"]
        b = row["balances"]
        legs = [
            (builder.account_by_name("Cash"), b["Cash"], "debit"),
            (builder.account_by_name("Accounts Receivable"), b["Accounts Receivable"], "debit"),
            (builder.account_by_name("Inventory"), b["Inventory"], "debit"),
            (builder.account_by_name("Other Current Assets"), b["Other Current Assets"], "debit"),
            (builder.account_by_name("Fixed Assets - Equipment"), b["Fixed Assets - Equipment"], "debit"),
            (builder.account_by_name("Accumulated Depreciation"), b["Accumulated Depreciation"], "credit"),
            (builder.account_by_name("Accounts Payable"), b["Accounts Payable"], "credit"),
            (builder.account_by_name("Accrued Liabilities"), b["Accrued Liabilities"], "credit"),
            (builder.account_by_name("Short-term Debt"), b["Short-term Debt"], "credit"),
            (builder.account_by_name("Long-term Debt"), b["Long-term Debt"], "credit"),
            (builder.account_by_name("Other Liabilities"), b["Other Liabilities"], "credit"),
            (builder.account_by_name("Common Stock"), b["Common Stock"], "credit"),
            (builder.account_by_name("Retained Earnings"), b["Retained Earnings"], "credit"),
        ]
        builder.post_multi(
            date=builder.start_date, entity_id=entity_id, business_unit="Corporate",
            legs=legs, currency=ccy, document_id=f"OPEN_{entity_id}",
            document_type="Opening Balance", source_system="ERP",
        )


def generate_ar(
    rng: np.random.Generator, builder: LedgerBuilder, monthly_series: pd.DataFrame,
    customers: pd.DataFrame, end_date: pd.Timestamp,
) -> pd.DataFrame:
    revenue_targets = monthly_series[monthly_series["account_category"] == "Revenue"]
    ar_rows = []
    customers_by_entity = {eid: df for eid, df in customers.groupby("entity_id")}

    for _, target in revenue_targets.iterrows():
        entity_id, bu, month = target["entity_id"], target["business_unit"], target["month"]
        line_item, total = target["line_item"], target["target_amount"]
        pool = customers_by_entity.get(entity_id)
        if pool is None or pool.empty or total <= 0:
            continue

        n = int(rng.integers(*AR_INVOICE_COUNT_RANGE))
        amounts = _split_amount(rng, total, n)
        chosen = pool.sample(n=n, replace=True, random_state=rng.integers(0, 2**31 - 1))
        invoice_dates = _random_dates_in_month(rng, month, n)

        debit_acc = builder.account_by_name("Accounts Receivable")
        credit_acc = builder.account("Revenue", line_item, bu)

        for i in range(n):
            cust = chosen.iloc[i]
            invoice_date = invoice_dates[i]
            due_date = invoice_date + pd.Timedelta(days=int(cust["payment_terms_days"]))
            offset = float(_segment_offset(rng, cust["customer_segment"], 1)[0])
            would_be_payment = due_date + pd.Timedelta(days=offset)

            paid = rng.random() < COLLECTION_PROBABILITY and would_be_payment <= end_date
            payment_date = would_be_payment if paid else pd.NaT
            writeoff_date = due_date + pd.Timedelta(days=BAD_DEBT_WRITEOFF_DAYS_PAST_DUE)
            written_off = (not paid) and writeoff_date <= end_date
            if paid:
                status = "Paid"
            elif written_off:
                status = "Written Off"
            elif due_date < end_date:
                status = "Overdue"
            else:
                status = "Open"

            invoice_id = f"ARINV_{len(ar_rows) + 1:08d}"
            ar_rows.append({
                "invoice_id": invoice_id, "customer_id": cust["customer_id"],
                "entity_id": entity_id, "invoice_date": invoice_date, "due_date": due_date,
                "payment_date": payment_date, "currency": cust["currency"],
                "invoice_amount": round(amounts[i], 2), "payment_status": status,
                "business_unit": bu, "customer_segment": cust["customer_segment"],
            })

            doc_id = builder.new_document_id("ARINV")
            builder.post(
                date=invoice_date, entity_id=entity_id, business_unit=bu,
                debit_account=debit_acc, credit_account=credit_acc,
                amount=amounts[i], currency=cust["currency"], document_id=doc_id,
                document_type="Revenue Invoice", counterparty_id=cust["customer_id"],
                source_system="AR_SYSTEM", due_date=due_date, payment_date=payment_date,
            )
            if paid:
                doc_id_c = builder.new_document_id("ARCOL")
                builder.post(
                    date=payment_date, entity_id=entity_id, business_unit=bu,
                    debit_account=builder.account_by_name("Cash"), credit_account=debit_acc,
                    amount=amounts[i], currency=cust["currency"], document_id=doc_id_c,
                    document_type="Cash Receipt", counterparty_id=cust["customer_id"],
                    source_system="AR_SYSTEM", due_date=due_date, payment_date=payment_date,
                )
            elif written_off:
                doc_id_w = builder.new_document_id("ARWO")
                builder.post(
                    date=writeoff_date, entity_id=entity_id, business_unit=bu,
                    debit_account=builder.account_by_name("Bad Debt Expense"), credit_account=debit_acc,
                    amount=amounts[i], currency=cust["currency"], document_id=doc_id_w,
                    document_type="Bad Debt Writeoff", counterparty_id=cust["customer_id"],
                    source_system="AR_SYSTEM", due_date=due_date, payment_date=None,
                )

    return pd.DataFrame(ar_rows)


def generate_ap_opex(
    rng: np.random.Generator, builder: LedgerBuilder, monthly_series: pd.DataFrame,
    vendors: pd.DataFrame, end_date: pd.Timestamp,
) -> pd.DataFrame:
    """AP invoices for every Operating Expense line item except Salaries
    (paid directly, see generate_salaries) and Raw Materials (routed
    through Inventory, see generate_inventory_and_cogs)."""
    targets = monthly_series[
        (monthly_series["account_category"] == "Operating Expense")
        & (~monthly_series["line_item"].isin(["Salaries", "Raw Materials"]))
    ]
    return _generate_ap_from_targets(rng, builder, targets, vendors, end_date, expense_account_fn=(
        lambda bt: builder.account("Operating Expense", bt["line_item"], bt["business_unit"])
    ))


def _generate_ap_from_targets(rng, builder, targets, vendors, end_date, expense_account_fn) -> pd.DataFrame:
    ap_rows = []
    vendors_by_entity = {eid: df for eid, df in vendors.groupby("entity_id")}

    for _, target in targets.iterrows():
        entity_id, bu, month = target["entity_id"], target["business_unit"], target["month"]
        line_item, total = target["line_item"], target["target_amount"]
        pool = vendors_by_entity.get(entity_id)
        if pool is None or pool.empty or total <= 0:
            continue

        n = int(rng.integers(*AP_INVOICE_COUNT_RANGE))
        net_amounts = _split_amount(rng, total, n)
        chosen = pool.sample(n=n, replace=True, random_state=rng.integers(0, 2**31 - 1))
        invoice_dates = _random_dates_in_month(rng, month, n)

        debit_acc = expense_account_fn(target)
        tax_rate = TAX_RATE_BY_ENTITY[entity_id]

        for i in range(n):
            vend = chosen.iloc[i]
            invoice_date = invoice_dates[i]
            due_date = invoice_date + pd.Timedelta(days=int(vend["payment_terms_days"]))
            offset = rng.normal(3, 7)
            would_be_payment = due_date + pd.Timedelta(days=float(offset))

            paid = rng.random() < AP_PAYMENT_PROBABILITY and would_be_payment <= end_date
            payment_date = would_be_payment if paid else pd.NaT
            if paid:
                status = "Paid"
            elif due_date < end_date:
                status = "Overdue"
            else:
                status = "Open"

            net_amount = round(net_amounts[i], 2)
            tax_amount = round(net_amount * tax_rate, 2)
            invoice_amount = round(net_amount + tax_amount, 2)
            invoice_id = f"APINV_{len(ap_rows) + 1:08d}"

            ap_rows.append({
                "invoice_id": invoice_id, "vendor_id": vend["vendor_id"], "entity_id": entity_id,
                "invoice_date": invoice_date, "due_date": due_date, "payment_date": payment_date,
                "currency": vend["currency"], "invoice_amount": invoice_amount,
                "tax_amount": tax_amount, "net_amount": net_amount, "payment_status": status,
                "business_unit": bu, "expense_category": line_item,
            })

            doc_id = builder.new_document_id("APINV")
            builder.post(
                date=invoice_date, entity_id=entity_id, business_unit=bu,
                debit_account=debit_acc, credit_account=builder.account_by_name("Accounts Payable"),
                amount=net_amount, currency=vend["currency"], document_id=doc_id,
                document_type="Vendor Invoice", counterparty_id=vend["vendor_id"],
                source_system="AP_SYSTEM", due_date=due_date, payment_date=payment_date,
            )
            if paid:
                doc_id_p = builder.new_document_id("APPAY")
                builder.post(
                    date=payment_date, entity_id=entity_id, business_unit=bu,
                    debit_account=builder.account_by_name("Accounts Payable"),
                    credit_account=builder.account_by_name("Cash"),
                    amount=net_amount, currency=vend["currency"], document_id=doc_id_p,
                    document_type="Vendor Payment", counterparty_id=vend["vendor_id"],
                    source_system="AP_SYSTEM", due_date=due_date, payment_date=payment_date,
                )

    return pd.DataFrame(ap_rows)


def generate_inventory_and_cogs(
    rng: np.random.Generator, builder: LedgerBuilder, monthly_series: pd.DataFrame,
    vendors: pd.DataFrame, end_date: pd.Timestamp,
) -> pd.DataFrame:
    """COGS recognition (Dr COGS / Cr Inventory) plus the AP-funded
    inventory replenishment (Dr Inventory / Cr AP) that keeps the Inventory
    balance moving realistically instead of staying frozen at its opening
    value."""
    cogs_targets = monthly_series[monthly_series["account_category"] == "COGS"]
    ap_rows = []
    vendors_by_entity = {eid: df for eid, df in vendors.groupby("entity_id")}

    for _, target in cogs_targets.iterrows():
        entity_id, bu, month = target["entity_id"], target["business_unit"], target["month"]
        line_item, cogs_amount = target["line_item"], target["target_amount"]
        if cogs_amount <= 0:
            continue

        consumption_date = month + pd.Timedelta(days=27)
        builder.post(
            date=consumption_date, entity_id=entity_id, business_unit=bu,
            debit_account=builder.account("COGS", line_item, bu),
            credit_account=builder.account_by_name("Inventory"),
            amount=cogs_amount, currency="EUR" if entity_id == "ENT_EU" else (
                "USD" if entity_id == "ENT_US" else "GBP"),
            document_id=builder.new_document_id("COGSREC"),
            document_type="Inventory Consumption", source_system="ERP",
        )

        pool = vendors_by_entity.get(entity_id)
        if pool is None or pool.empty:
            continue
        purchase_amount = cogs_amount * rng.lognormal(0, 0.08)
        vend = pool.sample(n=1, random_state=rng.integers(0, 2**31 - 1)).iloc[0]
        invoice_date = month + pd.Timedelta(days=int(rng.integers(0, 10)))
        due_date = invoice_date + pd.Timedelta(days=int(vend["payment_terms_days"]))
        offset = rng.normal(3, 7)
        would_be_payment = due_date + pd.Timedelta(days=float(offset))
        paid = rng.random() < AP_PAYMENT_PROBABILITY and would_be_payment <= end_date
        payment_date = would_be_payment if paid else pd.NaT
        status = "Paid" if paid else ("Overdue" if due_date < end_date else "Open")

        tax_rate = TAX_RATE_BY_ENTITY[entity_id]
        net_amount = round(purchase_amount, 2)
        tax_amount = round(net_amount * tax_rate, 2)
        invoice_amount = round(net_amount + tax_amount, 2)
        ap_rows.append({
            "invoice_id": f"APINV_INV_{len(ap_rows) + 1:08d}", "vendor_id": vend["vendor_id"],
            "entity_id": entity_id, "invoice_date": invoice_date, "due_date": due_date,
            "payment_date": payment_date, "currency": vend["currency"],
            "invoice_amount": invoice_amount, "tax_amount": tax_amount, "net_amount": net_amount,
            "payment_status": status, "business_unit": bu, "expense_category": "Raw Materials",
        })

        doc_id = builder.new_document_id("INVPUR")
        builder.post(
            date=invoice_date, entity_id=entity_id, business_unit=bu,
            debit_account=builder.account_by_name("Inventory"),
            credit_account=builder.account_by_name("Accounts Payable"),
            amount=net_amount, currency=vend["currency"], document_id=doc_id,
            document_type="Vendor Invoice", counterparty_id=vend["vendor_id"],
            source_system="AP_SYSTEM", due_date=due_date, payment_date=payment_date,
        )
        if paid:
            builder.post(
                date=payment_date, entity_id=entity_id, business_unit=bu,
                debit_account=builder.account_by_name("Accounts Payable"),
                credit_account=builder.account_by_name("Cash"),
                amount=net_amount, currency=vend["currency"],
                document_id=builder.new_document_id("APPAY"),
                document_type="Vendor Payment", counterparty_id=vend["vendor_id"],
                source_system="AP_SYSTEM", due_date=due_date, payment_date=payment_date,
            )

    return pd.DataFrame(ap_rows)


def generate_salaries(builder: LedgerBuilder, monthly_series: pd.DataFrame, entities: pd.DataFrame) -> None:
    ccy_by_entity = dict(zip(entities["entity_id"], entities["functional_currency"]))
    targets = monthly_series[monthly_series["line_item"] == "Salaries"]
    for _, t in targets.iterrows():
        pay_date = t["month"] + pd.Timedelta(days=24)
        builder.post(
            date=pay_date, entity_id=t["entity_id"], business_unit=t["business_unit"],
            debit_account=builder.account("Operating Expense", "Salaries", t["business_unit"]),
            credit_account=builder.account_by_name("Cash"),
            amount=t["target_amount"], currency=ccy_by_entity[t["entity_id"]],
            document_id=builder.new_document_id("PAYROLL"),
            document_type="Payroll", source_system="PAYROLL_SYSTEM",
        )


def generate_depreciation_and_capex(
    rng: np.random.Generator, builder: LedgerBuilder, entities: pd.DataFrame,
    opening: pd.DataFrame, months: pd.DatetimeIndex,
) -> None:
    ccy_by_entity = dict(zip(entities["entity_id"], entities["functional_currency"]))
    scale_by_entity = dict(zip(entities["entity_id"], entities["annual_revenue_scale_eur"]))
    fixed_assets_gross = {
        row["entity_id"]: row["balances"]["Fixed Assets - Equipment"] for _, row in opening.iterrows()
    }
    from src.data.ledger import DEPRECIATION_MONTHLY_RATE

    for entity_id, ccy in ccy_by_entity.items():
        for m_idx, month in enumerate(months):
            if m_idx % 3 == 0:
                capex = scale_by_entity[entity_id] * rng.uniform(*CAPEX_QUARTER_PCT_RANGE)
                builder.post(
                    date=month + pd.Timedelta(days=10), entity_id=entity_id, business_unit="Corporate",
                    debit_account=builder.account_by_name("Fixed Assets - Equipment"),
                    credit_account=builder.account_by_name("Cash"),
                    amount=capex, currency=ccy, document_id=builder.new_document_id("CAPEX"),
                    document_type="Capex", source_system="ERP",
                )
                fixed_assets_gross[entity_id] += capex

            depreciation = fixed_assets_gross[entity_id] * DEPRECIATION_MONTHLY_RATE
            builder.post(
                date=month + pd.Timedelta(days=28), entity_id=entity_id, business_unit="Corporate",
                debit_account=builder.account_by_name("Depreciation Expense"),
                credit_account=builder.account_by_name("Accumulated Depreciation"),
                amount=depreciation, currency=ccy, document_id=builder.new_document_id("DEPR"),
                document_type="Depreciation", source_system="ERP",
            )


def generate_debt_interest(
    builder: LedgerBuilder, entities: pd.DataFrame, opening: pd.DataFrame, months: pd.DatetimeIndex,
) -> None:
    from src.data.ledger import DEBT_ANNUAL_RATE
    ccy_by_entity = dict(zip(entities["entity_id"], entities["functional_currency"]))
    debt_by_entity = {
        row["entity_id"]: row["balances"]["Short-term Debt"] + row["balances"]["Long-term Debt"]
        for _, row in opening.iterrows()
    }
    for entity_id, debt in debt_by_entity.items():
        interest_monthly = debt * DEBT_ANNUAL_RATE / 12
        for month in months:
            builder.post(
                date=month + pd.Timedelta(days=29 if month.days_in_month > 29 else month.days_in_month - 1),
                entity_id=entity_id, business_unit="Corporate",
                debit_account=builder.account_by_name("Interest Expense"),
                credit_account=builder.account_by_name("Cash"),
                amount=interest_monthly, currency=ccy_by_entity[entity_id],
                document_id=builder.new_document_id("INT"),
                document_type="Interest Payment", source_system="BANKING_SYSTEM",
            )
