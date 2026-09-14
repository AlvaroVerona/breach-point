"""Injects realistic, controlled data-quality problems into otherwise-clean
generated data. Every function takes an explicit rng and rate so injection
is reproducible and each issue type is independently testable.

These functions corrupt data in place conceptually (they return a new,
modified copy) -- they are applied to the RAW layer only. The clean data
generated in transaction_events.py is the ground truth the quality engine
(Phase 3) is expected to recover by quarantining what these functions break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_CURRENCIES = {"EUR", "USD", "GBP"}


def inject_exact_duplicates(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    n = len(df)
    n_dupes = int(n * rate)
    if n_dupes == 0:
        return df
    dupe_idx = rng.choice(df.index, size=n_dupes, replace=False)
    return pd.concat([df, df.loc[dupe_idx]], ignore_index=True)


def inject_missing_values(df: pd.DataFrame, rng: np.random.Generator, rates: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for col, rate in rates.items():
        if col not in df.columns or rate <= 0:
            continue
        n_missing = int(len(df) * rate)
        if n_missing == 0:
            continue
        idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_negative_amounts(df: pd.DataFrame, rng: np.random.Generator, rate: float, amount_col: str) -> pd.DataFrame:
    df = df.copy()
    n = int(len(df) * rate)
    if n == 0:
        return df
    idx = rng.choice(df.index, size=n, replace=False)
    df.loc[idx, amount_col] = -df.loc[idx, amount_col].abs()
    return df


def inject_invalid_currency(df: pd.DataFrame, rng: np.random.Generator, rate: float, currency_col: str) -> pd.DataFrame:
    df = df.copy()
    n = int(len(df) * rate)
    if n == 0:
        return df
    idx = rng.choice(df.index, size=n, replace=False)
    bogus = rng.choice(["CHF", "XYZ", "JPY", "N/A"], size=n)
    df.loc[idx, currency_col] = bogus
    return df


def inject_invalid_account(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    df = df.copy()
    n = int(len(df) * rate)
    if n == 0 or "account_id" not in df.columns:
        return df
    idx = rng.choice(df.index, size=n, replace=False)
    bogus_ids = rng.choice(["9999", "0000", "ACC_UNKNOWN"], size=n)
    df.loc[idx, "account_id"] = bogus_ids
    return df


def inject_payment_before_document_date(
    df: pd.DataFrame, rng: np.random.Generator, rate: float, doc_date_col: str, payment_col: str,
) -> pd.DataFrame:
    df = df.copy()
    eligible = df.index[df[payment_col].notna()]
    n = int(len(eligible) * rate)
    if n == 0:
        return df
    idx = rng.choice(eligible, size=n, replace=False)
    df.loc[idx, payment_col] = df.loc[idx, doc_date_col] - pd.to_timedelta(
        rng.integers(1, 15, size=n), unit="D"
    )
    return df


def inject_inconsistent_account_category(
    transactions: pd.DataFrame, rng: np.random.Generator, rate: float,
) -> pd.DataFrame:
    """Denormalization drift: the row's account_category no longer matches
    what the chart of accounts says for that account_id (§13 example)."""
    df = transactions.copy()
    categories = df["account_category"].dropna().unique().tolist()
    n = int(len(df) * rate)
    if n == 0 or len(categories) < 2:
        return df
    idx = rng.choice(df.index, size=n, replace=False)

    def _swap(cat):
        choices = [c for c in categories if c != cat]
        return rng.choice(choices) if choices else cat

    df.loc[idx, "account_category"] = df.loc[idx, "account_category"].apply(_swap)
    return df


def inject_unbalanced_journal_entries(
    transactions: pd.DataFrame, rng: np.random.Generator, rate: float,
) -> pd.DataFrame:
    """Breaks sum(debit) == sum(credit) for a sample of document_ids by
    perturbing one leg's amount. Opening-balance documents are excluded so
    the balance sheet stays reconcilable from month 1 for the vast majority
    of periods -- only the sampled periods/entities should show a break."""
    df = transactions.copy()
    candidate_docs = df.loc[~df["document_id"].str.startswith("OPEN_"), "document_id"].unique()
    n = int(len(candidate_docs) * rate)
    if n == 0:
        return df
    chosen_docs = rng.choice(candidate_docs, size=n, replace=False)
    for doc_id in chosen_docs:
        doc_rows = df.index[df["document_id"] == doc_id]
        if len(doc_rows) < 2:
            continue
        target_row = rng.choice(doc_rows)
        factor = rng.uniform(0.5, 0.95)
        for col in ("amount", "debit", "credit"):
            df.loc[target_row, col] = df.loc[target_row, col] * factor
    return df


def inject_missing_periods(
    df: pd.DataFrame, rng: np.random.Generator, rate: float, date_col: str, entity_col: str,
) -> pd.DataFrame:
    """Simulates a source system failing to send a full month of records
    for a given entity."""
    periods = df[[entity_col]].copy()
    periods["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M")
    combos = periods.drop_duplicates().reset_index(drop=True)
    n = max(1, int(len(combos) * rate)) if rate > 0 else 0
    if n == 0 or combos.empty:
        return df
    chosen = combos.sample(n=min(n, len(combos)), random_state=rng.integers(0, 2**31 - 1))
    mask = pd.Series(False, index=df.index)
    for _, row in chosen.iterrows():
        this_month = pd.to_datetime(df[date_col]).dt.to_period("M") == row["_month"]
        mask |= this_month & (df[entity_col] == row[entity_col])
    return df.loc[~mask].reset_index(drop=True)


def inject_invalid_date_strings(series: pd.Series, rng: np.random.Generator, rate: float) -> pd.Series:
    """Operates on a column already formatted as strings (post date->str
    conversion at CSV-export time) and replaces some with syntactically
    invalid calendar dates -- something a real datetime column could never
    hold, which is why this must run on the string representation."""
    series = series.copy()
    n = int(len(series) * rate)
    if n == 0:
        return series
    idx = np.random.default_rng(rng.integers(0, 2**31 - 1)).choice(series.index, size=n, replace=False)
    bogus_dates = ["2024-02-30", "2024-13-05", "2023-00-15", "not-a-date"]
    series.loc[idx] = rng.choice(bogus_dates, size=n)
    return series
