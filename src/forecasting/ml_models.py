"""Gradient Boosting forecast (§29-30), scikit-learn only -- see CLAUDE.md
for why (no XGBoost dependency). Multi-step-ahead forecasting uses the
recursive strategy: predict one step, feed that prediction back in as the
next step's lag_1, and so on -- the standard approach for a model that
only ever learned to predict one step ahead from lag features.

Small-sample caveat: 36 months of history with lag_12 warmed up leaves at
most ~2 dozen trainable rows per series, so hyperparameters are kept
deliberately shallow (few, shallow trees) to reduce overfitting risk on a
sample this small -- this is not a production-scale forecasting setup."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.common.config import load_config
from src.forecasting.features import build_lag_features


def _make_model(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=seed,
    )


def gradient_boosting_forecast(train: pd.Series, horizon: int, seed: int | None = None) -> np.ndarray:
    if seed is None:
        seed = load_config()["data"]["seed"]

    features = build_lag_features(train)
    if len(features) < 6:
        # Not enough post-lag-warmup rows to fit anything meaningful.
        from src.forecasting.baseline import naive_forecast
        return naive_forecast(train, horizon)

    feature_cols = [c for c in features.columns if c != "target"]
    model = _make_model(seed)
    model.fit(features[feature_cols], features["target"])

    history = train.copy()
    predictions = []
    for _ in range(horizon):
        next_period = history.index[-1] + 1
        # build_lag_features can't be reused here: it drops rows with a NaN
        # target, but the row we want to predict is exactly the one whose
        # target doesn't exist yet -- so its lag features are computed by
        # hand instead, from the same recipe.
        row = _next_step_features(history, next_period)
        pred = float(model.predict(row[feature_cols])[0])
        predictions.append(pred)
        history = pd.concat([history, pd.Series([pred], index=[next_period])])

    return np.array(predictions)


def _next_step_features(history: pd.Series, next_period: pd.Period) -> pd.DataFrame:
    lags = (1, 2, 3, 12)
    row = {}
    for lag in lags:
        idx = next_period - lag
        row[f"lag_{lag}"] = history.get(idx, np.nan)
    row["rolling_mean"] = history.iloc[-3:].mean() if len(history) >= 3 else history.mean()
    row["month_sin"] = np.sin(2 * np.pi * next_period.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * next_period.month / 12)
    row["trend_index"] = len(history)
    return pd.DataFrame([row])
