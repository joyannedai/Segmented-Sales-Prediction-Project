import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seasonal_naive_forecast(train_series: np.ndarray, test_len: int, period: int = 12) -> np.ndarray:
    if len(train_series) >= period:
        pattern = train_series[-period:]
        repeats = test_len // period + 1
        return np.tile(pattern, repeats)[:test_len]
    return np.full(test_len, np.mean(train_series))


def compute_baseline(df: pd.DataFrame, group_cols: list, target: str, test_ratio: float = 0.2) -> Dict[str, float]:
    mean_wapes = []
    seasonal_wapes = []

    for _, group in df.groupby(group_cols):
        group = group.sort_values("month")
        n = len(group)
        if n < 6:
            continue
        split_idx = int(n * (1 - test_ratio))
        train_series = group.iloc[:split_idx][target].values
        test_series = group.iloc[split_idx:][target].values
        if len(test_series) == 0:
            continue
        mean_pred = np.full(len(test_series), np.mean(train_series))
        mean_wapes.append(wape(test_series, mean_pred))
        seasonal_pred = seasonal_naive_forecast(train_series, len(test_series))
        seasonal_wapes.append(wape(test_series, seasonal_pred))

    mean_wape = float(np.mean(mean_wapes)) if mean_wapes else np.nan
    seasonal_wape = float(np.mean(seasonal_wapes)) if seasonal_wapes else np.nan
    best_baseline = min(mean_wape, seasonal_wape)

    logger.info(f"Baseline: mean={mean_wape:.2f}%, seasonal_naive={seasonal_wape:.2f}%, best={best_baseline:.2f}%")
    return {"mean": mean_wape, "seasonal": seasonal_wape, "best": best_baseline}


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / (denom + 1e-8) * 100)
