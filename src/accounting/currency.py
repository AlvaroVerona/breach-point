"""Converts multi-currency GL amounts to the base reporting currency (EUR)
using the monthly fx_rates.csv generated in Phase 2. Every transaction row
carries its own currency; conversion happens at the month of the
transaction date, matching how the ledger was generated (see CLAUDE.md)."""

from __future__ import annotations

import pandas as pd

BASE_CURRENCY = "EUR"


def convert_to_eur(amount: pd.Series, currency: pd.Series, date: pd.Series, fx_rates: pd.DataFrame) -> pd.Series:
    period = pd.to_datetime(date).dt.to_period("M")
    fx = fx_rates.copy()
    fx["_period"] = pd.to_datetime(fx["rate_date"]).dt.to_period("M")
    rate_lookup = fx.set_index(["currency", "_period"])["rate_to_eur"]

    key = pd.MultiIndex.from_arrays([currency, period])
    rates = rate_lookup.reindex(key).to_numpy()
    return pd.to_numeric(amount, errors="coerce") * rates
