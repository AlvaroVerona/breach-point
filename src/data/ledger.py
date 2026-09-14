"""Double-entry GL ledger generation, plus the AR/AP sub-ledgers.

Every economic event posts a balanced journal entry (one or more debit
lines and matching credit lines sharing a document_id). Because Assets,
Liabilities and Revenue/Expense all flow through the same balanced legs,
the Balance Sheet equation (Assets = Liabilities + Equity, with Equity's
Retained Earnings component derived as cumulative net income) holds by
construction on clean data -- exactly the property the quality engine's
reconciliation checks are designed to verify, and which the intentionally
injected broken entries (see quality_injection.py) are designed to break.

Tax on AP invoices (tax_amount) is carried on the accounts_payable.csv
record for validation purposes (gross = net + tax) but is not separately
posted to a VAT/sales-tax payable account in the GL -- a deliberate scope
simplification; only net_amount hits the expense/inventory leg.
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

TAX_RATE_BY_ENTITY = {"ENT_EU": 0.19, "ENT_UK": 0.20, "ENT_US": 0.07}
SOURCE_AR = "AR_SYSTEM"
SOURCE_AP = "AP_SYSTEM"
SOURCE_ERP = "ERP"
SOURCE_PAYROLL = "PAYROLL_SYSTEM"
SOURCE_BANK = "BANKING_SYSTEM"

DEPRECIATION_MONTHLY_RATE = 0.015
DEBT_ANNUAL_RATE = 0.065


def _coa_index(coa: pd.DataFrame) -> dict:
    idx = {}
    for _, row in coa.iterrows():
        key_bu = (row["account_category"], row["account_subcategory"], row["expected_business_unit"])
        idx[key_bu] = (row["account_id"], row["account_name"])
        idx.setdefault(("name", row["account_name"]), (row["account_id"], row["account_name"]))
    return idx


def _days_in_month(ts: pd.Timestamp) -> int:
    return calendar.monthrange(ts.year, ts.month)[1]


def _split_amount(rng: np.random.Generator, total: float, n: int, alpha: float = 3.0) -> np.ndarray:
    if n <= 1 or total <= 0:
        return np.array([max(total, 0.0)])
    weights = rng.dirichlet(np.full(n, alpha))
    return weights * total


def _random_dates_in_month(rng: np.random.Generator, month: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    days = _days_in_month(month)
    offsets = rng.integers(0, days, size=n)
    return pd.DatetimeIndex([month + pd.Timedelta(days=int(o)) for o in offsets])


def opening_balances(rng: np.random.Generator, entities: pd.DataFrame) -> pd.DataFrame:
    """One balanced opening journal entry per entity at the start of the
    window. Retained Earnings is solved as the plug that balances the
    opening trial balance -- standard practice when seeding a new ledger,
    representing accumulated history before the observed window."""
    rows = []
    for _, entity in entities.iterrows():
        scale = entity["annual_revenue_scale_eur"]
        cash = scale * rng.uniform(0.08, 0.14)
        ar = scale * rng.uniform(0.12, 0.18)
        inventory = scale * rng.uniform(0.05, 0.09)
        other_ca = scale * rng.uniform(0.02, 0.04)
        fixed_assets = scale * rng.uniform(0.35, 0.55)
        accum_depr = fixed_assets * rng.uniform(0.15, 0.30)
        ap = scale * rng.uniform(0.08, 0.13)
        accrued = scale * rng.uniform(0.02, 0.04)
        st_debt = scale * rng.uniform(0.04, 0.07)
        lt_debt = scale * rng.uniform(0.12, 0.20)
        other_liab = scale * rng.uniform(0.01, 0.03)
        common_stock = scale * rng.uniform(0.20, 0.30)

        total_assets = cash + ar + inventory + other_ca + (fixed_assets - accum_depr)
        total_liab = ap + accrued + st_debt + lt_debt + other_liab
        retained_earnings = total_assets - total_liab - common_stock

        balances = {
            "Cash": cash, "Accounts Receivable": ar, "Inventory": inventory,
            "Other Current Assets": other_ca, "Fixed Assets - Equipment": fixed_assets,
            "Accumulated Depreciation": accum_depr, "Accounts Payable": ap,
            "Accrued Liabilities": accrued, "Short-term Debt": st_debt,
            "Long-term Debt": lt_debt, "Other Liabilities": other_liab,
            "Common Stock": common_stock, "Retained Earnings": retained_earnings,
        }
        rows.append({"entity_id": entity["entity_id"], "balances": balances,
                      "functional_currency": entity["functional_currency"]})
    return pd.DataFrame(rows)


class LedgerBuilder:
    def __init__(self, rng: np.random.Generator, coa: pd.DataFrame, start_date: str):
        self.rng = rng
        self.coa_idx = _coa_index(coa)
        self._name_to_category = dict(zip(coa["account_name"], coa["account_category"]))
        self.start_date = pd.Timestamp(start_date)
        self.rows: list[dict] = []
        self._txn_seq = 0
        self._doc_seq = 0

    def _next_txn_id(self) -> str:
        self._txn_seq += 1
        return f"TXN_{self._txn_seq:08d}"

    def _next_doc_id(self, prefix: str = "DOC") -> str:
        self._doc_seq += 1
        return f"{prefix}_{self._doc_seq:08d}"

    def post(
        self, *, date, entity_id, business_unit, debit_account, credit_account,
        amount, currency, document_id, document_type, counterparty_id=None,
        source_system=SOURCE_ERP, due_date=None, payment_date=None,
    ) -> None:
        if amount <= 0:
            return
        acc_id_d, acc_name_d = debit_account
        acc_id_c, acc_name_c = credit_account
        common = dict(
            transaction_date=date, entity_id=entity_id, business_unit=business_unit,
            document_type=document_type, document_id=document_id,
            counterparty_id=counterparty_id, currency=currency,
            payment_due_date=due_date, payment_date=payment_date,
            source_system=source_system,
        )
        self.rows.append({
            "transaction_id": self._next_txn_id(), "account_id": acc_id_d,
            "account_name": acc_name_d, "account_category": self._category_of(acc_name_d),
            "amount": round(amount, 2), "debit": round(amount, 2), "credit": 0.0, **common,
        })
        self.rows.append({
            "transaction_id": self._next_txn_id(), "account_id": acc_id_c,
            "account_name": acc_name_c, "account_category": self._category_of(acc_name_c),
            "amount": round(amount, 2), "debit": 0.0, "credit": round(amount, 2), **common,
        })

    def post_multi(
        self, *, date, entity_id, business_unit, legs, currency, document_id,
        document_type, counterparty_id=None, source_system=SOURCE_ERP,
        due_date=None, payment_date=None,
    ) -> None:
        """legs: list of (account_tuple, amount, side) where side is 'debit'
        or 'credit'. Caller is responsible for balancing debits and credits;
        used for the multi-line opening balance entry."""
        common = dict(
            transaction_date=date, entity_id=entity_id, business_unit=business_unit,
            document_type=document_type, document_id=document_id,
            counterparty_id=counterparty_id, currency=currency,
            payment_due_date=due_date, payment_date=payment_date,
            source_system=source_system,
        )
        # Round each leg to the cent independently, then push the residual
        # rounding error (at most a few cents) onto the last leg so the
        # document balances exactly -- the standard "penny plug" real
        # ledgers use, rather than letting per-leg rounding drift the
        # document out of balance. A negative amount (e.g. an opening
        # Retained Earnings deficit) flips the leg to its opposite side
        # instead of being silently dropped.
        resolved = []
        for account_tuple, amount, side in legs:
            if amount < 0:
                side = "credit" if side == "debit" else "debit"
                amount = -amount
            resolved.append([account_tuple, round(amount, 2), side])
        if not resolved:
            return
        signed_total = sum(a if s == "debit" else -a for _, a, s in resolved)
        last_side = resolved[-1][2]
        resolved[-1][1] += signed_total if last_side == "credit" else -signed_total

        for account_tuple, amount, side in resolved:
            if amount <= 0:
                continue
            acc_id, acc_name = account_tuple
            self.rows.append({
                "transaction_id": self._next_txn_id(), "account_id": acc_id,
                "account_name": acc_name, "account_category": self._category_of(acc_name),
                "amount": amount,
                "debit": amount if side == "debit" else 0.0,
                "credit": amount if side == "credit" else 0.0,
                **common,
            })

    def new_document_id(self, prefix: str = "DOC") -> str:
        return self._next_doc_id(prefix)

    def _category_of(self, account_name: str) -> str:
        return self._name_to_category.get(account_name, "Unknown")

    def account(self, category: str, subcategory: str, bu: str | None):
        return self.coa_idx[(category, subcategory, bu)]

    def account_by_name(self, name: str):
        return self.coa_idx[("name", name)]

    def to_frame(self) -> pd.DataFrame:
        cols = [
            "transaction_id", "transaction_date", "entity_id", "business_unit",
            "account_id", "account_name", "account_category", "document_type",
            "document_id", "counterparty_id", "currency", "amount", "debit", "credit",
            "payment_due_date", "payment_date", "source_system",
        ]
        return pd.DataFrame(self.rows)[cols]
