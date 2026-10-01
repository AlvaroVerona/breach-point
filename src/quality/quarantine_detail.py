"""Quarantine ledger joined back to the raw records' own fields.

The ledger alone only carries rule/severity/reason; the Data Quality page also
filters by entity, source system and date, which live on the raw record
(row_uid == record_id). Joining needs the full raw files (25 MB for transactions),
so `main()` precomputes the joined result into `reports/outputs/` as small CSVs the
dashboard can use when the raw data isn't available (e.g. the hosted demo).
"""

from __future__ import annotations

import pandas as pd

from src.common.config import PROJECT_ROOT

DATASETS = ["transactions", "accounts_receivable", "accounts_payable"]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "outputs"


def build_quarantine_detail(ledger: pd.DataFrame, raw: pd.DataFrame, name: str) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    raw = raw.reset_index().rename(columns={"index": "_raw_row"})
    raw.insert(0, "row_uid", [f"{name.upper()}_{i + 1:08d}" for i in range(len(raw))])
    return ledger.merge(raw, left_on="record_id", right_on="row_uid", how="left", suffixes=("", "_raw"))


def detail_path(name: str):
    return OUTPUT_DIR / f"quarantine_detail_{name}.csv"


def main() -> None:
    for name in DATASETS:
        ledger = pd.read_csv(PROJECT_ROOT / "data" / "quarantine" / f"{name}.csv")
        raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / f"{name}.csv")
        detail = build_quarantine_detail(ledger, raw, name)
        detail.to_csv(detail_path(name), index=False)
        print(f"{name}: {len(detail):,} quarantined rows -> {detail_path(name).name}")


if __name__ == "__main__":
    main()
