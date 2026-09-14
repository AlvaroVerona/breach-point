"""Phase 5 tests: the accrual estimator's arithmetic on handcrafted series
(precise), plus sanity checks on the real generated data (no missing
values once enough history exists, the estimator isn't systematically
biased, business-unit vs. corporate-level categories are shaped correctly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.provisions.accrual_model import (
    BUSINESS_UNIT_CATEGORIES, CORPORATE_CATEGORIES, _estimate_series,
    _load_processed, build_accruals, compute_monthly_expense,
)


def _series(values: list[float]) -> pd.DataFrame:
    periods = pd.period_range("2023-01", periods=len(values), freq="M")
    return pd.DataFrame({"period": periods, "recognized_cost": values})


def test_accrual_arithmetic_identity():
    est = _estimate_series(_series([100, 110, 90, 105, 95, 100]))
    assert (est["estimated_accrual"] - (est["expected_cost"] - est["recognized_cost"])).abs().max() < 1e-9


def test_rolling_average_matches_manual_calculation():
    est = _estimate_series(_series([100, 200, 300, 400, 500]))
    # At index 3 (value=400), window is the trailing 3 prior values [100,200,300].
    row = est[est["period"] == pd.Period("2023-04", freq="M")].iloc[0]
    assert row["rolling_average"] == pytest.approx((100 + 200 + 300) / 3)
    assert row["recognized_cost"] == 400


def test_no_estimate_before_minimum_history():
    est = _estimate_series(_series([100, 110]))
    assert est.empty  # MIN_HISTORY_PERIODS=2, so the 2 periods available aren't enough


def test_confidence_interval_widens_with_variance():
    stable = _estimate_series(_series([100, 101, 99, 100, 100]))
    volatile = _estimate_series(_series([100, 300, 10, 250, 50]))
    stable_width = (stable["ci_upper"] - stable["ci_lower"]).iloc[-1]
    volatile_width = (volatile["ci_upper"] - volatile["ci_lower"]).iloc[-1]
    assert volatile_width > stable_width


def test_seasonal_average_requires_a_prior_same_month():
    values = list(range(100, 100 + 14))  # 14 months, so month 13 has a prior "month 1" match
    est = _estimate_series(_series(values))
    early = est.iloc[0]
    assert np.isnan(early["seasonal_average"])
    later = est.iloc[-1]
    assert not np.isnan(later["seasonal_average"])


@pytest.fixture(scope="module")
def processed():
    return _load_processed()


@pytest.fixture(scope="module")
def accruals(processed):
    transactions, coa, fx_rates = processed
    return build_accruals(transactions, coa, fx_rates)


def test_business_unit_categories_have_business_unit(accruals):
    bu_rows = accruals[accruals["expense_category"].isin(BUSINESS_UNIT_CATEGORIES)]
    assert bu_rows["business_unit"].notna().all()


def test_corporate_categories_have_no_business_unit(accruals):
    corp_rows = accruals[accruals["expense_category"].isin(CORPORATE_CATEGORIES)]
    assert corp_rows["business_unit"].isna().all()


def test_no_missing_core_estimates(accruals):
    core_cols = ["expected_cost", "recognized_cost", "estimated_accrual", "ci_lower", "ci_upper"]
    assert not accruals[core_cols].isna().any().any()


def test_estimator_is_not_systematically_biased(accruals):
    """A rolling-average estimator on a genuinely noisy-but-stationary-ish
    recurring expense should average out close to zero over many periods --
    if it didn't, that would indicate an arbitrary/fabricated accrual
    methodology rather than one grounded in the actual historical pattern (§26)."""
    by_category = accruals.groupby("expense_category").agg(
        mean_accrual=("estimated_accrual", "mean"), mean_recognized=("recognized_cost", "mean"),
    )
    for category, row in by_category.iterrows():
        assert abs(row["mean_accrual"]) < 0.15 * row["mean_recognized"], (
            f"{category}: mean accrual {row['mean_accrual']:.2f} is not small relative to "
            f"mean recognized cost {row['mean_recognized']:.2f}"
        )


def test_compute_monthly_expense_covers_all_entities(processed):
    transactions, coa, fx_rates = processed
    monthly = compute_monthly_expense(transactions, coa, fx_rates, "Utilities")
    assert set(monthly["entity_id"]) == {"ENT_EU", "ENT_US", "ENT_UK"}
    assert (monthly["recognized_cost"] > 0).all()


def test_no_pathological_accruals(accruals):
    """estimated_accrual can legitimately be negative (recognized exceeded
    what was expected that period) -- that's not the bug this guards
    against. What must never happen is a NaN/inf estimate, or one wildly
    disproportionate to the actual recognized cost (which would indicate a
    broken calculation, not real variance -- see
    test_estimator_is_not_systematically_biased for the aggregate-bias
    check this complements at the individual-row level)."""
    assert accruals["estimated_accrual"].apply(np.isfinite).all()
    # Compared against expected_cost (the rolling average), not
    # recognized_cost -- a single month's recognized cost can legitimately
    # dip close to zero, which would make even a normal-sized accrual look
    # like a huge multiple of it. expected_cost is the stable scale of the
    # series.
    reasonable = accruals["estimated_accrual"].abs() <= 5 * accruals["expected_cost"].abs().clip(lower=1.0)
    assert reasonable.all(), accruals.loc[~reasonable, ["entity_id", "expense_category", "period", "estimated_accrual", "expected_cost"]]
