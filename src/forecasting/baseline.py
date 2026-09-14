"""Naive and seasonal-naive baselines (§29). Every other model has to beat
these to be worth using."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeats the last observed value for every future step."""
    return np.full(horizon, train.iloc[-1])


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    """Repeats the value from `season_length` periods ago, cycling further
    back into the same seasonal history for steps beyond one season. Falls
    back to naive_forecast if there isn't a full season of history yet."""
    if len(train) < season_length:
        return naive_forecast(train, horizon)
    season = train.iloc[-season_length:].to_numpy()
    return np.array([season[i % season_length] for i in range(horizon)])
