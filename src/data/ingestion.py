"""Reads data/raw/*.csv and assigns the row-level lineage key (row_uid)
every quality check and the quarantine layer key off. Dates are kept as raw
strings here on purpose -- parsing them now with errors='coerce' would
silently turn an intentionally-invalid date string (e.g. "2024-02-30") into
the same NaT as a genuinely missing date, losing exactly the distinction
the quality engine's validity checks (src/quality/validity.py) need to draw.
"""

from __future__ import annotations

import pandas as pd

from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger

log = get_logger("ingestion")

RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_FILES = {
    "transactions": "transactions.csv",
    "accounts_receivable": "accounts_receivable.csv",
    "accounts_payable": "accounts_payable.csv",
    "entities": "entities.csv",
    "chart_of_accounts": "chart_of_accounts.csv",
    "vendors": "vendors.csv",
    "customers": "customers.csv",
    "fx_rates": "fx_rates.csv",
}


def load_raw(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {list(DATASET_FILES)}.")
    path = RAW_DIR / DATASET_FILES[dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {path}. Run `make generate-data` (or "
            "`python -m src.data.generate_data`) first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.insert(0, "row_uid", [f"{dataset_name.upper()}_{i + 1:08d}" for i in range(len(df))])
    log.info("Ingested %s: %d rows from %s", dataset_name, len(df), path.name)
    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    return {name: load_raw(name) for name in DATASET_FILES}
