"""Phase 7 tests: metrics arithmetic, the time-based split (never
shuffled), baseline/statistical/ML forecast shapes, no-leakage (a model's
prediction must depend only on the train prefix, never on the held-out
validation values), and integration checks against the real data (every
series produces a full-horizon forecast with no missing values)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baseline import naive_forecast, seasonal_naive_forecast
from src.forecasting.evaluation import (
    build_all_forecasts, evaluate_series, forecast_series, mae, mape, rmse, smape, time_split,
)
from src.forecasting.features import TARGET_METRICS, get_series, load_target_series
from src.forecasting.ml_models import gradient_boosting_forecast
from src.forecasting.statistical_models import exponential_smoothing_forecast


def _series(values: list[float], start: str = "2023-01") -> pd.Series:
    idx = pd.period_range(start, periods=len(values), freq="M")
    return pd.Series(values, index=idx)


def test_mae_rmse_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 320.0])
    assert mae(y_true, y_pred) == pytest.approx((10 + 10 + 20) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((100 + 100 + 400) / 3))


def test_mape_smape_basic():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    assert mape(y_true, y_pred) == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)
    assert smape(y_true, y_pred) > 0


def test_time_split_preserves_order_and_sizes():
    s = _series(list(range(36)))
    train, val = time_split(s, validation_months=6)
    assert len(train) == 30
    assert len(val) == 6
    assert list(train.index) + list(val.index) == list(s.index)  # never shuffled


def test_naive_forecast():
    s = _series([10, 20, 30])
    result = naive_forecast(s, horizon=4)
    assert list(result) == [30, 30, 30, 30]


def test_seasonal_naive_cycles():
    s = _series(list(range(1, 13)))  # 12 months, values 1..12
    result = seasonal_naive_forecast(s, horizon=14, season_length=12)
    assert list(result[:12]) == list(range(1, 13))
    assert result[12] == 1  # cycles back into the same season
    assert result[13] == 2


def test_seasonal_naive_falls_back_to_naive_with_insufficient_history():
    s = _series([10, 20, 30])
    result = seasonal_naive_forecast(s, horizon=2, season_length=12)
    assert list(result) == [30, 30]


def test_exponential_smoothing_returns_correct_horizon():
    s = _series([100 + i * 2 + (5 if i % 12 < 6 else -5) for i in range(30)])
    result = exponential_smoothing_forecast(s, horizon=6)
    assert len(result) == 6
    assert np.isfinite(result).all()


def test_exponential_smoothing_falls_back_on_degenerate_series():
    s = _series([5.0, 5.0])
    result = exponential_smoothing_forecast(s, horizon=3)
    assert len(result) == 3
    assert np.isfinite(result).all()


def test_gradient_boosting_forecast_shape():
    s = _series([100 + i + 10 * np.sin(i / 12 * 2 * np.pi) for i in range(30)])
    result = gradient_boosting_forecast(s, horizon=5, seed=42)
    assert len(result) == 5
    assert np.isfinite(result).all()


def test_gradient_boosting_falls_back_with_too_little_history():
    s = _series([10.0, 12.0, 11.0])
    result = gradient_boosting_forecast(s, horizon=3, seed=42)
    assert len(result) == 3
    assert np.isfinite(result).all()


def test_evaluate_series_covers_all_models():
    s = _series([100 + i for i in range(30)])
    table = evaluate_series(s, validation_months=6, seed=42)
    assert set(table["model"]) == {"naive", "seasonal_naive", "exponential_smoothing", "gradient_boosting"}
    assert {"mae", "rmse", "mape", "smape"} <= set(table.columns)
    assert table[["mae", "rmse"]].notna().all().all()


def test_no_leakage_validation_forecast_depends_only_on_train():
    base = _series([100 + i for i in range(30)])
    train, _ = time_split(base, validation_months=6)

    altered = base.copy()
    altered.iloc[-6:] = 999999.0  # corrupt only the held-out validation values

    pred_original = naive_forecast(train, horizon=6)
    train_altered, _ = time_split(altered, validation_months=6)
    pred_altered = naive_forecast(train_altered, horizon=6)

    assert list(pred_original) == list(pred_altered)  # unaffected by val-period corruption


def test_forecast_series_selects_lowest_rmse_model():
    s = _series([100 + i for i in range(30)])
    eval_table, best_model, future, residual_std = forecast_series(s, horizon=6, validation_months=6, seed=42)
    assert best_model == eval_table.sort_values("rmse").iloc[0]["model"]
    assert len(future) == 6
    assert residual_std >= 0


@pytest.fixture(scope="module")
def panel():
    return load_target_series()


def test_load_target_series_has_all_metrics_and_entities(panel):
    assert set(TARGET_METRICS) <= set(panel.columns)
    assert set(panel["entity_id"]) == {"ENT_EU", "ENT_US", "ENT_UK"}
    assert len(panel) == 3 * 36


@pytest.fixture(scope="module")
def all_forecasts():
    return build_all_forecasts()


def test_build_all_forecasts_no_missing_values(all_forecasts):
    forecasts, evaluation = all_forecasts
    assert not forecasts[["forecast", "ci_lower", "ci_upper"]].isna().any().any()
    assert not evaluation[["mae", "rmse"]].isna().any().any()


def test_build_all_forecasts_full_horizon_every_series(all_forecasts):
    forecasts, _ = all_forecasts
    counts = forecasts.groupby(["entity_id", "metric"]).size()
    assert (counts == 12).all()


def test_confidence_interval_widens_with_horizon(all_forecasts):
    forecasts, _ = all_forecasts
    sample = forecasts[(forecasts["entity_id"] == "ENT_EU") & (forecasts["metric"] == "revenue")]
    sample = sample.sort_values("step")
    widths = (sample["ci_upper"] - sample["ci_lower"]).to_numpy()
    assert widths[-1] >= widths[0]
