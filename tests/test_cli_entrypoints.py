"""Phase 11: closes a coverage gap left by every other test file, which
calls each module's underlying functions directly rather than its main()
CLI entrypoint. main() is the actual `python -m src.X` / `make X` path --
it's the one that writes reports/outputs/*, so it deserves its own
(lightweight) regression test: call it, check the file(s) it promises to
write actually exist with the expected shape. Not a duplicate of the
other test files, which cover the underlying logic in detail; this covers
the orchestration/IO glue around it."""

from __future__ import annotations

import json

from src.accounting import balance_sheet, cash_flow, income_statement, working_capital
from src.data import generate_data
from src.optimization import cash_management
from src.provisions import accrual_model
from src.risk import liquidity
from src.simulation import monte_carlo


def test_generate_data_main_writes_raw_csvs():
    """Same seed, so this rewrites data/raw/ with byte-identical content
    (see test_generate_all_reproducible) -- safe to run here, and it's the
    one path that actually exercises write_raw()'s file-per-dataset loop."""
    generate_data.main()
    for name in ["transactions", "accounts_receivable", "accounts_payable", "entities"]:
        path = generate_data.RAW_DIR / f"{name}.csv"
        assert path.exists()
        assert path.stat().st_size > 0


def test_monte_carlo_main_writes_summary_and_paths():
    monte_carlo.main()
    assert monte_carlo.OUTPUT_PATH.exists()
    assert monte_carlo.PATHS_OUTPUT_PATH.exists()


def test_income_statement_main_writes_csv():
    income_statement.main()
    assert income_statement.OUTPUT_PATH.exists()
    assert income_statement.OUTPUT_PATH.stat().st_size > 0


def test_balance_sheet_main_writes_csv():
    balance_sheet.main()
    assert balance_sheet.OUTPUT_PATH.exists()
    assert balance_sheet.OUTPUT_PATH.stat().st_size > 0


def test_cash_flow_main_writes_csv():
    cash_flow.main()
    assert cash_flow.OUTPUT_PATH.exists()
    assert cash_flow.OUTPUT_PATH.stat().st_size > 0


def test_working_capital_main_writes_csv():
    working_capital.main()
    assert working_capital.OUTPUT_PATH.exists()
    assert working_capital.OUTPUT_PATH.stat().st_size > 0


def test_accrual_model_main_writes_csv():
    accrual_model.main()
    assert accrual_model.OUTPUT_PATH.exists()
    assert accrual_model.OUTPUT_PATH.stat().st_size > 0


def test_liquidity_main_writes_json_with_all_scenarios():
    liquidity.main()
    assert liquidity.OUTPUT_PATH.exists()
    with liquidity.OUTPUT_PATH.open() as f:
        report = json.load(f)
    assert {"base", "optimistic", "pessimistic", "risk_thresholds"} <= set(report.keys())
    assert liquidity.MONTHLY_OUTPUT_PATH.exists()


def test_cash_management_main_writes_result_and_plan():
    cash_management.main()
    assert cash_management.OUTPUT_PATH.exists()
    with cash_management.OUTPUT_PATH.open() as f:
        report = json.load(f)
    assert {"real_minimum_cash", "demo_minimum_cash", "before", "after"} <= set(report.keys())
    assert cash_management.PLAN_OUTPUT_PATH.exists()
