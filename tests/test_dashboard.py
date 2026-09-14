"""Phase 10 tests: runs every dashboard page headlessly via Streamlit's
AppTest harness and asserts it executes without raising -- the automated
equivalent of the manual browser walkthrough used to build these pages."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"

PAGES = [
    APP_DIR / "app.py",
    APP_DIR / "pages" / "01_Data_Quality.py",
    APP_DIR / "pages" / "02_Financial_Statements.py",
    APP_DIR / "pages" / "03_Working_Capital.py",
    APP_DIR / "pages" / "04_Forecasting.py",
    APP_DIR / "pages" / "05_Liquidity_Risk.py",
    APP_DIR / "pages" / "06_Optimization.py",
]


@pytest.mark.parametrize("page_path", PAGES, ids=[p.stem for p in PAGES])
def test_page_runs_without_exception(page_path):
    at = AppTest.from_file(str(page_path), default_timeout=60)
    at.run()
    assert not at.exception, f"{page_path.name} raised: {[str(e) for e in at.exception]}"


def test_data_quality_page_renders_quarantine_table():
    at = AppTest.from_file(str(APP_DIR / "pages" / "01_Data_Quality.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 2  # rule-count table + quarantine detail table


def test_financial_statements_page_has_tabs():
    at = AppTest.from_file(str(APP_DIR / "pages" / "02_Financial_Statements.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 3
