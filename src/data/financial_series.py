"""Generative process for monthly revenue and expense targets, per entity
and business unit. This is the source of truth: GL transactions in
ledger.py are generated to sum to these targets, so financial statements
built from transactions stay consistent with the series forecasting models
are trained on (see CLAUDE.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.master_data import (
    BUSINESS_UNITS, REVENUE_BUSINESS_UNITS, REVENUE_TYPES, COGS_TYPES,
    FIXED_OPEX_TYPES, VARIABLE_OPEX_TYPES,
)

REVENUE_BU_SHARE = {"Sales": 0.55, "Operations": 0.30, "Marketing": 0.15}
REVENUE_TYPE_WEIGHT = {
    "Sales": {"Product Revenue": 0.6, "Service Revenue": 0.3, "Subscription Revenue": 0.1},
    "Operations": {"Product Revenue": 0.5, "Service Revenue": 0.4, "Subscription Revenue": 0.1},
    "Marketing": {"Product Revenue": 0.2, "Service Revenue": 0.2, "Subscription Revenue": 0.6},
}
GROSS_MARGIN = {"Sales": 0.62, "Operations": 0.55, "Marketing": 0.70}
COGS_TYPE_WEIGHT = {"Cost of Goods Sold": 0.7, "Cost of Services": 0.3}

# Fixed opex monthly base as a fraction of entity annual revenue scale, per BU.
FIXED_OPEX_BU_WEIGHT = {
    "Sales": 0.028, "Operations": 0.022, "Marketing": 0.018, "R&D": 0.030, "Corporate": 0.035,
}
FIXED_OPEX_TYPE_WEIGHT = {
    "Salaries": 0.55, "Rent": 0.12, "Insurance": 0.06,
    "Software Subscriptions": 0.10, "Professional Services": 0.10, "Utilities": 0.07,
}

# Variable opex: base (fraction of entity annual scale) + pct-of-revenue.
VARIABLE_OPEX_BASE_WEIGHT = {
    "Sales": 0.006, "Operations": 0.006, "Marketing": 0.006, "R&D": 0.004, "Corporate": 0.004,
}
VARIABLE_OPEX_TYPE_WEIGHT = {
    "Marketing": 0.30, "Logistics": 0.20, "Raw Materials": 0.20,
    "Sales Commissions": 0.15, "Travel": 0.08, "IT Infrastructure": 0.07,
}
VARIABLE_OPEX_REVENUE_PCT = {
    "Marketing": 0.04, "Logistics": 0.03, "Raw Materials": 0.05,
    "Sales Commissions": 0.06, "Travel": 0.005, "IT Infrastructure": 0.005,
}

MONTHLY_TREND = 0.006
SLOWDOWN_START_MONTHS_FROM_END = 5
SLOWDOWN_MONTHLY_DROP = 0.03


def _seasonality_index(month_of_year: int) -> float:
    base = 1.0 + 0.06 * np.sin(2 * np.pi * (month_of_year - 3) / 12)
    if month_of_year in (11, 12):
        base *= 1.12
    if month_of_year in (1, 8):
        base *= 0.92
    return base


def generate_monthly_series(
    rng: np.random.Generator,
    entities: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Returns a long-format DataFrame of monthly targets:
    entity_id, business_unit, month, account_category, line_item,
    target_amount (entity functional currency)."""
    months = pd.date_range(start_date, end_date, freq="MS")
    n_months = len(months)
    rows: list[dict] = []

    for _, entity in entities.iterrows():
        entity_id = entity["entity_id"]
        annual_scale = entity["annual_revenue_scale_eur"]
        monthly_base = annual_scale / 12

        revenue_by_bu_month: dict[tuple[str, pd.Timestamp], float] = {}

        for m_idx, month in enumerate(months):
            trend_factor = (1 + MONTHLY_TREND) ** m_idx
            months_from_end = n_months - 1 - m_idx
            if months_from_end < SLOWDOWN_START_MONTHS_FROM_END:
                slowdown_periods = SLOWDOWN_START_MONTHS_FROM_END - months_from_end
                trend_factor *= (1 - SLOWDOWN_MONTHLY_DROP) ** slowdown_periods
            season = _seasonality_index(month.month)

            for bu in REVENUE_BUSINESS_UNITS:
                bu_share = REVENUE_BU_SHARE[bu]
                noise = rng.lognormal(mean=0, sigma=0.05)
                bu_revenue = monthly_base * bu_share * trend_factor * season * noise
                revenue_by_bu_month[(bu, month)] = bu_revenue

                for rtype, wt in REVENUE_TYPE_WEIGHT[bu].items():
                    amt = bu_revenue * wt * rng.lognormal(0, 0.03)
                    rows.append({
                        "entity_id": entity_id, "business_unit": bu, "month": month,
                        "account_category": "Revenue", "line_item": rtype,
                        "target_amount": round(amt, 2),
                    })

                cogs_total = bu_revenue * (1 - GROSS_MARGIN[bu]) * rng.lognormal(0, 0.04)
                for ctype, wt in COGS_TYPE_WEIGHT.items():
                    rows.append({
                        "entity_id": entity_id, "business_unit": bu, "month": month,
                        "account_category": "COGS", "line_item": ctype,
                        "target_amount": round(cogs_total * wt, 2),
                    })

            for bu in BUSINESS_UNITS:
                fixed_base = monthly_base * FIXED_OPEX_BU_WEIGHT[bu] * (1.02 ** (m_idx / 12))
                for ftype, wt in FIXED_OPEX_TYPE_WEIGHT.items():
                    amt = fixed_base * wt * rng.lognormal(0, 0.03)
                    rows.append({
                        "entity_id": entity_id, "business_unit": bu, "month": month,
                        "account_category": "Operating Expense", "line_item": ftype,
                        "target_amount": round(amt, 2),
                    })

                bu_revenue = revenue_by_bu_month.get((bu, month), 0.0)
                var_base = monthly_base * VARIABLE_OPEX_BASE_WEIGHT[bu]
                for vtype, wt in VARIABLE_OPEX_TYPE_WEIGHT.items():
                    pct = VARIABLE_OPEX_REVENUE_PCT[vtype]
                    amt = (var_base * wt + bu_revenue * pct) * rng.lognormal(0, 0.04)
                    rows.append({
                        "entity_id": entity_id, "business_unit": bu, "month": month,
                        "account_category": "Operating Expense", "line_item": vtype,
                        "target_amount": round(amt, 2),
                    })

    return pd.DataFrame(rows)
