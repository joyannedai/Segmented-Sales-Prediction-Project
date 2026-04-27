import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

from src.evaluation import compute_baseline
from src.models.base import evaluate, wape
from src.models.dl_models import run_dl_experiment
from src.models.ensemble import run_fusion
from src.models.traditional import train_prophet_sample
from src.models.tree_models import train_gbdt, train_lightgbm, train_random_forest, train_xgboost
from src.tuning import optimize_gbdt, optimize_lightgbm, optimize_random_forest, optimize_xgboost
from src.utils import set_random_seed

logger = logging.getLogger(__name__)


def prepare_short_term_data(df: pd.DataFrame, cat_features: list, test_ratio: float = 0.2, val_ratio: float = 0.15, seed: int = 42):
    short_term_df = df[df["predictability_level"].isna()].copy()
    logger.info(f"Short-term data: {len(short_term_df):,} rows, {short_term_df.groupby(['ref_branch_code', 'material_nature_sum_desc']).ngroups} combos")

    group_cols = ["ref_branch_code", "material_nature_sum_desc"]
    target = "monthly_sales"

    def create_time_features(df_sub):
        df_sub = df_sub.copy()
        df_sub["month"] = pd.to_datetime(df_sub["month"])
        df_sub["year"] = df_sub["month"].dt.year
        df_sub["month_num"] = df_sub["month"].dt.month
        df_sub["quarter"] = df_sub["month"].dt.quarter
        df_sub["month_sin"] = np.sin(2 * np.pi * df_sub["month_num"] / 12)
        df_sub["month_cos"] = np.cos(2 * np.pi * df_sub["month_num"] / 12)
        return df_sub

    def encode_cats(train, test):
        for col in cat_features:
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]], axis=0).astype(str).unique()
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
        return train, test

    holiday_cols = [col for col in short_term_df.columns if col.startswith("holiday_") and col.endswith("_flag")]
    holiday_cols.append("total_holiday_days")
    base_features = ["year", "month_num", "quarter", "month_sin", "month_cos", "price"] + holiday_cols + cat_features

    train_list, test_list = [], []
    train_raw_list, test_raw_list = [], []
    for _, group in short_term_df.groupby(group_cols):
        group = group.sort_values("month")
        n = len(group)
        if n < 6:
            continue
        split_idx = int(n * (1 - test_ratio))
        train_list.append(group.iloc[:split_idx].copy())
        test_list.append(group.iloc[split_idx:].copy())
        train_raw_list.append(group.iloc[:split_idx][group_cols + ["month", target]].copy())
        test_raw_list.append(group.iloc[split_idx:][group_cols + ["month", target]].copy())

    train_all = pd.concat(train_list, ignore_index=True)
    test_all = pd.concat(test_list, ignore_index=True)
    train_raw_all = pd.concat(train_raw_list, ignore_index=True)
    test_raw_all = pd.concat(test_raw_list, ignore_index=True)

    train_all = create_time_features(train_all)
    test_all = create_time_features(test_all)
    train_all, test_all = encode_cats(train_all, test_all)

    X_train = train_all[base_features].fillna(0)
    y_train = train_all[target]
    X_test = test_all[base_features].fillna(0)
    y_test = test_all[target]

    n_train = len(X_train)
    val_size = int(n_train * val_ratio)
    X_val = X_train.iloc[n_train - val_size:]
    y_val = y_train.iloc[n_train - val_size:]
    X_train_strict = X_train.iloc[:n_train - val_size]
    y_train_strict = y_train.iloc[:n_train - val_size]

    baseline = compute_baseline(short_term_df, group_cols, target, test_ratio)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_val": X_val, "y_val": y_val,
        "X_train_strict": X_train_strict, "y_train_strict": y_train_strict,
        "train_raw": train_raw_all, "test_raw": test_raw_all,
        "baseline": baseline, "group_cols": group_cols,
    }


def run_short_term_pipeline(data_dict: dict, use_tuning: bool = True, seed: int = 42) -> Dict:
    set_random_seed(seed)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    X_train_strict = data_dict["X_train_strict"]
    y_train_strict = data_dict["y_train_strict"]
    train_raw = data_dict["train_raw"]
    test_raw = data_dict["test_raw"]
    group_cols = data_dict["group_cols"]
    baseline = data_dict["baseline"]

    all_test_preds = {}
    test_results = []
    val_wapes = {}

    model_trainers = {
        "RandomForest": (train_random_forest, optimize_random_forest),
        "GBDT": (train_gbdt, optimize_gbdt),
        "LightGBM": (train_lightgbm, optimize_lightgbm),
        "XGBoost": (train_xgboost, optimize_xgboost),
    }

    for name, (train_fn, tune_fn) in model_trainers.items():
        try:
            if use_tuning:
                logger.info(f"[{name}] Tuning on strict train/val...")
                best_params, best_val_wape = tune_fn(X_train_strict, y_train_strict, X_val, y_val, n_trials=30, seed=seed)
                if best_params is not None:
                    logger.info(f"[{name}] Best val WAPE={best_val_wape:.2f}%, params={best_params}")
                    model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test, **best_params)
                    val_wapes[name] = best_val_wape
                else:
                    logger.info(f"[{name}] Tuning unavailable, using defaults")
                    model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test)
            else:
                logger.info(f"[{name}] Training with defaults...")
                model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test)

            all_test_preds[name] = y_pred
            metrics = evaluate(y_test, y_pred)
            test_results.append({"model": name, **metrics})
            logger.info(f"[{name}] Test WAPE={metrics['wape']:.2f}%")
        except Exception as e:
            logger.warning(f"[{name}] failed: {e}")

    # Ridge
    try:
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(X_train, y_train)
        y_pred_ridge = np.maximum(0, ridge.predict(X_test))
        all_test_preds["Ridge"] = y_pred_ridge
        metrics = evaluate(y_test, y_pred_ridge)
        test_results.append({"model": "Ridge", **metrics})
        logger.info(f"[Ridge] Test WAPE={metrics['wape']:.2f}%")
    except Exception as e:
        logger.warning(f"Ridge failed: {e}")

    # Prophet sample
    try:
        _, _, prophet_wape, _ = train_prophet_sample(train_raw, test_raw, group_cols, sample_size=50, random_state=seed)
        if prophet_wape is not None:
            test_results.append({"model": "Prophet", "wape": prophet_wape})
            logger.info(f"[Prophet] Test WAPE={prophet_wape:.2f}%")
    except Exception as e:
        logger.warning(f"Prophet failed: {e}")

    # DL
    try:
        group_lengths = train_raw.groupby(group_cols).size()
        usable_seq = min(12, max(3, int(group_lengths.quantile(0.25))))
        for dl_name, dl_type in [("LSTM", "lstm"), ("Transformer", "transformer")]:
            _, _, wape_dl, _ = run_dl_experiment(
                train_raw.copy(), test_raw.copy(),
                model_type=dl_type, seq_length=usable_seq,
                epochs=100, batch_size=32, patience=15, lr=0.001, weight_decay=1e-5,
            )
            test_results.append({"model": dl_name, "wape": wape_dl})
            logger.info(f"[{dl_name}] Test WAPE={wape_dl:.2f}%")
    except Exception as e:
        logger.warning(f"DL failed: {e}")

    # Fusion
    if len(all_test_preds) >= 2:
        min_len = min(len(pred) for pred in all_test_preds.values())
        aligned_preds = {k: v[-min_len:] for k, v in all_test_preds.items()}
        y_test_aligned = y_test.values[-min_len:]
        fusion = run_fusion(aligned_preds, y_test_aligned, val_wapes=val_wapes if val_wapes else None)
        for method, val in fusion.items():
            test_results.append({"model": f"Ensemble_{method.capitalize()}", "wape": val})

    results_df = pd.DataFrame(test_results)
    if not np.isnan(baseline.get("best", np.nan)):
        results_df["improvement_vs_baseline"] = (baseline["best"] - results_df["wape"]) / baseline["best"] * 100
    results_df = results_df.sort_values("wape")
    return results_df
