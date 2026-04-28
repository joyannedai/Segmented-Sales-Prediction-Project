import logging
from typing import Dict

import numpy as np

from src.evaluation import wape

logger = logging.getLogger(__name__)


def ensemble_average(preds: Dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(list(preds.values()), axis=0)


def ensemble_weighted(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    result = np.zeros_like(list(preds.values())[0])
    total_w = 0.0
    for name, pred in preds.items():
        w = weights.get(name, 1.0)
        result += w * pred
        total_w += w
    return result / total_w if total_w > 0 else result


def ensemble_median(preds: Dict[str, np.ndarray]) -> np.ndarray:
    return np.median(list(preds.values()), axis=0)


def ensemble_trimmed(preds: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.array(list(preds.values()))
    if arr.shape[0] <= 2:
        return np.mean(arr, axis=0)
    sorted_arr = np.sort(arr, axis=0)
    return np.mean(sorted_arr[1:-1], axis=0)


def run_fusion(test_preds: Dict[str, np.ndarray], y_test: np.ndarray, val_wapes: Dict[str, float] = None):
    min_len = min(len(pred) for pred in test_preds.values())
    y_test_aligned = y_test[-min_len:]
    aligned_preds = {k: v[-min_len:] for k, v in test_preds.items()}

    results = {}
    results["avg"] = wape(y_test_aligned, ensemble_average(aligned_preds))
    results["median"] = wape(y_test_aligned, ensemble_median(aligned_preds))
    results["trimmed"] = wape(y_test_aligned, ensemble_trimmed(aligned_preds))

    if val_wapes:
        weights = {name: 1.0 / max(val_wapes.get(name, 1.0), 1e-6) for name in aligned_preds}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        results["weighted"] = wape(y_test_aligned, ensemble_weighted(aligned_preds, weights))
    else:
        results["weighted"] = results["avg"]

    logger.info(f"Fusion results: avg={results['avg']:.2f}%, weighted={results['weighted']:.2f}%, median={results['median']:.2f}%, trimmed={results['trimmed']:.2f}%")
    return results
