import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.models.base import compute_baseline
from src.models.dl_models import run_dl_experiment
from src.models.ensemble import run_fusion
from src.models.traditional import train_prophet_sample, train_ridge
from src.models.tree_models import train_gbdt, train_lightgbm, train_random_forest, train_xgboost
from src.tuning import optimize_gbdt, optimize_lightgbm, optimize_random_forest, optimize_xgboost
from src.utils import set_random_seed

logger = logging.getLogger(__name__)


def load_tuned_params(tuned_params_path: str, grp: str, model_name: str) -> dict:
    """Load best tuned parameters from project-local JSON."""
    if not tuned_params_path or not os.path.exists(tuned_params_path):
        return None
    try:
        import json
        with open(tuned_params_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        params = tuned.get(grp, {}).get(model_name)
        if params:
            # random_state is managed by train_fn itself
            params = {k: v for k, v in params.items() if k != "random_state"}
            return params
        return None
    except Exception as e:
        logger.warning(f"Failed to load tuned params for {grp}/{model_name}: {e}")
        return None


def run_modeling(
    processed: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    grp: str,
    seed: int = 42,
) -> Tuple[List[dict], dict, dict, dict]:
    set_random_seed(seed)
    data_cfg = config["data"]
    model_cfg = config["modeling"]
    enable_tuning = model_cfg.get("enable_tuning", True)

    X_train = processed["X_train"]
    y_train = processed["y_train"]
    X_test = processed["X_test"]
    y_test = processed["y_test"]
    X_train_strict = processed["X_train_strict"]
    y_train_strict = processed["y_train_strict"]
    X_val = processed["X_val"]
    y_val = processed["y_val"]

    full_group = pd.concat([train_df, test_df]).sort_values("month")
    baseline = compute_baseline(full_group, data_cfg["group_cols"], data_cfg["target_col"], data_cfg["test_ratio"])

    all_results = []
    all_test_preds = {}
    trained_models = {}
    val_wapes = {}

    tuned_params_path = config.get("paths", {}).get("tuned_params")

    # Tree models
    tree_trainers = {
        "RandomForest": (train_random_forest, optimize_random_forest),
        "GBDT": (train_gbdt, optimize_gbdt),
        "LightGBM": (train_lightgbm, optimize_lightgbm),
        "XGBoost": (train_xgboost, optimize_xgboost),
    }

    for name, (train_fn, tune_fn) in tree_trainers.items():
        try:
            if enable_tuning:
                logger.info(f"[{grp}/{name}] Tuning on strict train/val...")
                best_params, best_val_wape = tune_fn(X_train_strict, y_train_strict, X_val, y_val, n_trials=30, seed=seed)
                if best_params is not None:
                    logger.info(f"[{grp}/{name}] Best val WAPE={best_val_wape:.2f}%, params={best_params}")
                    model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test, **best_params)
                    val_wapes[name] = best_val_wape
                else:
                    logger.info(f"[{grp}/{name}] Tuning unavailable, using defaults")
                    model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test)
            else:
                tuned_params = load_tuned_params(tuned_params_path, grp, name)
                if tuned_params:
                    logger.info(f"[{grp}/{name}] Using tuned params from tuned_params.json")
                    model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test, **tuned_params)
                else:
                    logger.info(f"[{grp}/{name}] Tuned params not found, falling back to defaults...")
                    model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test)

            all_test_preds[name] = y_pred
            trained_models[name] = model
            metrics["model"] = name
            all_results.append(metrics)
            logger.info(f"[{grp}/{name}] Test WAPE={metrics['wape']:.2f}%")
        except Exception as e:
            logger.warning(f"[{grp}/{name}] failed: {e}")

    # Non-tree models
    try:
        model, y_pred, metrics = train_ridge(X_train, y_train, X_test, y_test)
        all_test_preds["Ridge"] = y_pred
        trained_models["Ridge"] = model
        metrics["model"] = "Ridge"
        all_results.append(metrics)
        logger.info(f"[{grp}/Ridge] Test WAPE={metrics['wape']:.2f}%")
    except Exception as e:
        logger.warning(f"[{grp}/Ridge] failed: {e}")

    try:
        _, _, prophet_wape, _ = train_prophet_sample(
            train_df, test_df, data_cfg["group_cols"],
            sample_size=model_cfg.get("prophet_sample_size", 50), random_state=seed,
        )
        if prophet_wape is not None:
            all_results.append({"model": "Prophet", "wape": prophet_wape, "mape": prophet_wape})
            logger.info(f"[{grp}/Prophet] Test WAPE={prophet_wape:.2f}%")
    except Exception as e:
        logger.warning(f"[{grp}/Prophet] failed: {e}")

    # DL models
    try:
        for dl_name, dl_type in [("LSTM", "lstm"), ("Transformer", "transformer")]:
            _, _, wape_dl, dl_model = run_dl_experiment(
                train_df.copy(), test_df.copy(),
                model_type=dl_type,
                seq_length=model_cfg["dl"]["seq_length"],
                epochs=model_cfg["dl"]["epochs"],
                batch_size=model_cfg["dl"]["batch_size"],
                patience=model_cfg["dl"]["patience"],
                lr=model_cfg["dl"]["lr"],
                weight_decay=model_cfg["dl"]["weight_decay"],
            )
            all_results.append({"model": dl_name, "wape": wape_dl, "mape": wape_dl})
            logger.info(f"[{grp}/{dl_name}] Test WAPE={wape_dl:.2f}%")
    except Exception as e:
        logger.warning(f"[{grp}/DL] failed: {e}")

    # Fusion
    if len(all_test_preds) >= 2:
        min_len = min(len(pred) for pred in all_test_preds.values())
        aligned_preds = {k: v[-min_len:] for k, v in all_test_preds.items()}
        y_test_aligned = y_test.values[-min_len:]
        fusion = run_fusion(aligned_preds, y_test_aligned, val_wapes=val_wapes if val_wapes else None)
        for method, val in fusion.items():
            all_results.append({"model": f"Ensemble_{method.capitalize()}", "wape": val, "mape": val})

    return all_results, trained_models, all_test_preds, baseline, val_wapes
