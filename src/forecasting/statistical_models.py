"""Exponential Smoothing (Holt-Winters) (§29). With only 36 months of
history per entity, a full seasonal model needs at least two full cycles;
below that this degrades gracefully to a trend-only (Holt's linear) model,
and if statsmodels fails to fit at all (e.g. a degenerate/near-constant
series), falls back to the naive baseline -- logged, not silently swapped."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.common.logging_config import get_logger
from src.forecasting.baseline import naive_forecast

log = get_logger("statistical_models")

MIN_SEASONAL_PERIODS = 24


def exponential_smoothing_forecast(train: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    use_seasonal = len(train) >= MIN_SEASONAL_PERIODS
    try:
        model = ExponentialSmoothing(
            train.to_numpy(),
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=season_length if use_seasonal else None,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        return np.asarray(fitted.forecast(horizon))
    except Exception as exc:  # statsmodels can raise on degenerate/short series
        log.warning("ExponentialSmoothing failed (%s); falling back to naive forecast", exc)
        return naive_forecast(train, horizon)
