import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / (denom + 1e-8) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"mape": np.nan, "wape": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    wape_val = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100)

    return {"mape": mape_val, "wape": wape_val, "rmse": rmse, "mae": mae_val, "r2": r2}
