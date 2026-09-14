"""Metrics, the time-based (never shuffled -- §29/§45) train/validation
split, and the orchestrator that runs all three model families per series,
picks a winner on held-out validation error, and produces the genuine
12-month-ahead future forecast with a confidence interval.

Data-leakage prevention (§45): the validation forecast for months
N-5..N is produced by fitting every model on months 1..N-6 ONLY, then
forecasting forward -- no walk-forward re-fitting on validation-period
actuals, no peeking at the target period's own value in any feature.
Confidence intervals are derived from those same held-out validation
residuals (widening with sqrt(step) under a random-walk-error assumption),
never from in-sample fit.

Run: python -m src.forecasting.evaluation
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT, load_config
from src.common.logging_config import get_logger
from src.forecasting.baseline import naive_forecast, seasonal_naive_forecast
from src.forecasting.features import TARGET_METRICS, get_series, load_target_series
from src.forecasting.ml_models import gradient_boosting_forecast
from src.forecasting.statistical_models import exponential_smoothing_forecast

log = get_logger("forecast_evaluation")

MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "forecasts.csv"
EVAL_OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "forecast_evaluation.csv"
Z_95 = 1.96


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    nonzero = denom != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(2 * np.abs(y_true[nonzero] - y_pred[nonzero]) / denom[nonzero]) * 100)


def time_split(series: pd.Series, validation_months: int) -> tuple[pd.Series, pd.Series]:
    return series.iloc[:-validation_months], series.iloc[-validation_months:]


MODEL_FUNCS = {
    "naive": lambda train, h, seed: naive_forecast(train, h),
    "seasonal_naive": lambda train, h, seed: seasonal_naive_forecast(train, h),
    "exponential_smoothing": lambda train, h, seed: exponential_smoothing_forecast(train, h),
    "gradient_boosting": lambda train, h, seed: gradient_boosting_forecast(train, h, seed),
}


def evaluate_series(series: pd.Series, validation_months: int, seed: int) -> pd.DataFrame:
    train, val = time_split(series, validation_months)
    y_true = val.to_numpy()
    rows = []
    for name, fn in MODEL_FUNCS.items():
        y_pred = np.asarray(fn(train, len(val), seed))
        rows.append({
            "model": name, "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred), "smape": smape(y_true, y_pred),
        })
    return pd.DataFrame(rows)


def forecast_series(
    series: pd.Series, horizon: int, validation_months: int, seed: int,
) -> tuple[pd.DataFrame, str, np.ndarray, float]:
    """Returns (evaluation_table, best_model_name, future_forecast, residual_std)."""
    eval_table = evaluate_series(series, validation_months, seed)
    best_model = eval_table.sort_values("rmse").iloc[0]["model"]

    residual_std = _validation_residual_std(series, validation_months, seed, best_model)
    future = np.asarray(MODEL_FUNCS[best_model](series, horizon, seed))
    return eval_table, best_model, future, residual_std


def _validation_residual_std(series: pd.Series, validation_months: int, seed: int, model_name: str) -> float:
    train, val = time_split(series, validation_months)
    y_pred = np.asarray(MODEL_FUNCS[model_name](train, len(val), seed))
    residuals = val.to_numpy() - y_pred
    return float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.abs(residuals).mean())


def build_all_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    seed = config["data"]["seed"]
    horizon = config["forecast"]["horizon_months"]
    validation_months = config["forecast"]["validation_months"]

    panel = load_target_series()
    entities = panel["entity_id"].unique()

    forecast_rows, eval_rows = [], []
    for entity_id in entities:
        for metric in TARGET_METRICS:
            series = get_series(panel, entity_id, metric)
            eval_table, best_model, future, residual_std = forecast_series(
                series, horizon, validation_months, seed,
            )
            eval_table["entity_id"] = entity_id
            eval_table["metric"] = metric
            eval_table["selected"] = eval_table["model"] == best_model
            eval_rows.append(eval_table)

            last_period = series.index[-1]
            for step, value in enumerate(future, start=1):
                se = residual_std * np.sqrt(step)
                forecast_rows.append({
                    "entity_id": entity_id, "metric": metric, "period": last_period + step,
                    "step": step, "forecast": value, "model": best_model,
                    "ci_lower": value - Z_95 * se, "ci_upper": value + Z_95 * se,
                })

            if best_model == "gradient_boosting":
                _save_ml_model(entity_id, metric, series, seed)

    forecasts = pd.DataFrame(forecast_rows)
    evaluation = pd.concat(eval_rows, ignore_index=True)
    return forecasts, evaluation


def _save_ml_model(entity_id: str, metric: str, series: pd.Series, seed: int) -> None:
    from src.forecasting.ml_models import _make_model
    from src.forecasting.features import build_lag_features

    features = build_lag_features(series)
    if len(features) < 6:
        return
    feature_cols = [c for c in features.columns if c != "target"]
    model = _make_model(seed)
    model.fit(features[feature_cols], features["target"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"gbr_{entity_id}_{metric}.joblib"
    joblib.dump(model, path)


def main() -> None:
    forecasts, evaluation = build_all_forecasts()

    FORECAST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(FORECAST_OUTPUT_PATH, index=False)
    evaluation.to_csv(EVAL_OUTPUT_PATH, index=False)

    log.info("Wrote %d forecast rows to %s", len(forecasts), FORECAST_OUTPUT_PATH.relative_to(PROJECT_ROOT))
    log.info("Wrote model evaluation table to %s", EVAL_OUTPUT_PATH.relative_to(PROJECT_ROOT))

    selected = evaluation[evaluation["selected"]]
    log.info("Selected model by series:\n%s", selected[["entity_id", "metric", "model", "rmse", "mape"]].to_string(index=False))
    log.info("Model selection frequency: %s", selected["model"].value_counts().to_dict())


if __name__ == "__main__":
    main()
