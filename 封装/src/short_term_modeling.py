import logging
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.evaluation import evaluate, wape
from src.modeling import load_tuned_params
from src.models.ensemble import run_fusion
from src.models.traditional import train_ridge
from src.models.tree_models import train_gbdt, train_lightgbm, train_random_forest, train_xgboost
from src.tuning import optimize_gbdt, optimize_lightgbm, optimize_random_forest, optimize_xgboost
from src.utils import ensure_dir, set_random_seed
from src.visualization import plot_feature_importance, plot_model_comparison

logger = logging.getLogger(__name__)


def _safe_mean(values: np.ndarray, fallback: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else fallback


def split_short_term_train_test(
    df: pd.DataFrame,
    group_cols: List[str],
    min_months: int = 6,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for _, group in df.groupby(group_cols, sort=False):
        group = group.sort_values("month")
        n = len(group)
        if n < min_months:
            continue
        test_len = max(1, int(np.ceil(n * test_ratio)))
        split_idx = n - test_len
        if split_idx < 2:
            continue
        train_parts.append(group.iloc[:split_idx].copy())
        test_parts.append(group.iloc[split_idx:].copy())

    if not train_parts or not test_parts:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, cat_features: List[str]):
    train, test = train.copy(), test.copy()
    for col in cat_features:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], ignore_index=True).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    return train, test


def add_group_codes(train: pd.DataFrame, test: pd.DataFrame, group_cols: List[str]):
    train, test = train.copy(), test.copy()
    for idx, col in enumerate(group_cols):
        code_col = "store_code" if idx == 0 else "prod_code" if idx == 1 else f"group_code_{idx}"
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], ignore_index=True).astype(str)
        le.fit(combined)
        train[code_col] = le.transform(train[col].astype(str))
        test[code_col] = le.transform(test[col].astype(str))
    return train, test


def build_history_stats(train: pd.DataFrame, group_cols: List[str], target: str) -> pd.DataFrame:
    rows = []
    global_mean = _safe_mean(train[target].values)
    for keys, group in train.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group.sort_values("month")[target].astype(float).values
        nonzero = values[values > 0]
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "hist_len": len(values),
                "hist_mean": _safe_mean(values, global_mean),
                "hist_median": float(np.nanmedian(values)) if len(values) else global_mean,
                "hist_std": float(np.nanstd(values)) if len(values) else 0.0,
                "hist_min": float(np.nanmin(values)) if len(values) else global_mean,
                "hist_max": float(np.nanmax(values)) if len(values) else global_mean,
                "hist_last": float(values[-1]) if len(values) else global_mean,
                "hist_last2_mean": _safe_mean(values[-2:], global_mean),
                "hist_last3_mean": _safe_mean(values[-3:], global_mean),
                "hist_nonzero_mean": _safe_mean(nonzero, global_mean),
                "hist_zero_rate": float(np.mean(values == 0)) if len(values) else 0.0,
                "hist_trend_last_first": float(values[-1] - values[0]) if len(values) >= 2 else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_history_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_cols: List[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    stats = build_history_stats(train, group_cols, target)
    stat_cols = [c for c in stats.columns if c not in group_cols]
    train = train.merge(stats, on=group_cols, how="left")
    test = test.merge(stats, on=group_cols, how="left")

    for col in stat_cols:
        fill_value = train[col].median() if col in train.columns else 0.0
        train[col] = train[col].fillna(fill_value)
        test[col] = test[col].fillna(fill_value)
    return train, test, stat_cols


def prepare_short_term_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    data_cfg = config["data"]
    feat_cfg = config["features"]
    group_cols = data_cfg["group_cols"]
    target = data_cfg["target_col"]
    cat_features = [c for c in feat_cfg["cat_features"] if c in train_df.columns]

    train = add_time_features(train_df)
    test = add_time_features(test_df)
    train, test = encode_categoricals(train, test, cat_features)
    train, test = add_group_codes(train, test, group_cols)
    train, test, history_features = add_history_features(train, test, group_cols, target)

    holiday_cols = [c for c in train.columns if c.startswith("holiday_") and c.endswith("_flag")]
    if "total_holiday_days" in train.columns:
        holiday_cols.append("total_holiday_days")

    base_features = [
        "year",
        "month_num",
        "quarter",
        "month_sin",
        "month_cos",
        "price",
        "store_code",
        "prod_code",
    ]
    features = base_features + holiday_cols + cat_features + history_features
    features = [c for c in features if c in train.columns]

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[target]
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test[target]

    val_ratio = data_cfg.get("val_ratio", 0.15)
    split_idx = int(len(X_train) * (1 - val_ratio))
    if split_idx <= 0 or split_idx >= len(X_train):
        split_idx = max(1, len(X_train) - max(1, len(X_train) // 5))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_strict": X_train.iloc[:split_idx],
        "y_train_strict": y_train.iloc[:split_idx],
        "X_val": X_train.iloc[split_idx:],
        "y_val": y_train.iloc[split_idx:],
        "feat_names": features,
        "test_meta": test[group_cols + ["month", target]],
    }


def compute_short_baselines(
    df: pd.DataFrame,
    group_cols: List[str],
    target: str,
    min_months: int,
    test_ratio: float,
) -> Tuple[List[dict], float]:
    methods = {"Mean": [], "LastValue": [], "MovingAvg3": [], "SeasonalNaive": []}

    for _, group in df.groupby(group_cols, sort=False):
        group = group.sort_values("month")
        n = len(group)
        if n < min_months:
            continue
        test_len = max(1, int(np.ceil(n * test_ratio)))
        split_idx = n - test_len
        if split_idx < 2:
            continue
        train_values = group.iloc[:split_idx][target].astype(float).values
        test_values = group.iloc[split_idx:][target].astype(float).values
        if len(test_values) == 0:
            continue

        methods["Mean"].append(wape(test_values, np.full(len(test_values), _safe_mean(train_values))))
        methods["LastValue"].append(wape(test_values, np.full(len(test_values), train_values[-1])))
        methods["MovingAvg3"].append(wape(test_values, np.full(len(test_values), _safe_mean(train_values[-3:]))))

        if len(train_values) >= 12:
            pattern = train_values[-12:]
            pred = np.tile(pattern, len(test_values) // 12 + 1)[: len(test_values)]
        else:
            pred = np.full(len(test_values), _safe_mean(train_values))
        methods["SeasonalNaive"].append(wape(test_values, pred))

    results = []
    for name, vals in methods.items():
        if vals:
            results.append({"model": f"Baseline_{name}", "wape": float(np.mean(vals)), "mape": float(np.mean(vals))})

    best = min((r["wape"] for r in results), default=np.nan)
    return results, best


def _trial_count(config: dict, default: int = 30) -> int:
    trials = config.get("modeling", {}).get("optuna_trials", {})
    if isinstance(trials, dict):
        return int(trials.get("short", trials.get("fast", default)))
    return int(trials or default)


def _load_tuned_params_file(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_tuned_params_file(path: str, tuned: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tuned, f, ensure_ascii=False, indent=2)


def tune_short_term_params(
    df_predict: pd.DataFrame,
    config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    set_random_seed(seed)
    tuned_params_path = config.get("paths", {}).get("tuned_params", "tuned_params.json")
    data_cfg = config["data"]
    short_cfg = config.get("short_term", {})
    group_cols = data_cfg["group_cols"]
    min_months = int(short_cfg.get("min_months", 6))
    test_ratio = float(short_cfg.get("test_ratio", data_cfg.get("test_ratio", 0.2)))
    n_trials = _trial_count(config)

    short_df = df_predict[df_predict["predictability_level"].eq("short") | df_predict["predictability_level"].isna()].copy()
    if short_df.empty:
        logger.warning("[short/tuning] No short-term data found")
        return pd.DataFrame()

    train_df, test_df = split_short_term_train_test(short_df, group_cols, min_months=min_months, test_ratio=test_ratio)
    if train_df.empty or test_df.empty:
        logger.warning("[short/tuning] Insufficient data after split")
        return pd.DataFrame()

    processed = prepare_short_term_features(train_df, test_df, config)
    X_train_strict = processed["X_train_strict"]
    y_train_strict = processed["y_train_strict"]
    X_val = processed["X_val"]
    y_val = processed["y_val"]

    if len(X_val) == 0:
        logger.warning("[short/tuning] Empty validation set")
        return pd.DataFrame()

    tree_tuners = {
        "RandomForest": optimize_random_forest,
        "GBDT": optimize_gbdt,
        "LightGBM": optimize_lightgbm,
        "XGBoost": optimize_xgboost,
    }

    tuned_short = {}
    rows = []
    for name, tune_fn in tree_tuners.items():
        try:
            logger.info(f"[short/{name}] Tuning {n_trials} trials...")
            best_params, best_val_wape = tune_fn(
                X_train_strict,
                y_train_strict,
                X_val,
                y_val,
                n_trials=n_trials,
                seed=seed,
            )
            if best_params is None:
                logger.warning(f"[short/{name}] Tuning unavailable")
                continue
            tuned_short[name] = best_params
            rows.append(
                {
                    "group": "short",
                    "model": name,
                    "best_val_wape": best_val_wape,
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                    "n_trials": n_trials,
                }
            )
            logger.info(f"[short/{name}] Best val WAPE={best_val_wape:.2f}%, params={best_params}")
        except Exception as e:
            logger.warning(f"[short/{name}] tuning failed: {e}")

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        logger.warning("[short/tuning] No tuned parameters were produced")
        return results_df

    tuned = _load_tuned_params_file(tuned_params_path)
    tuned["short"] = tuned_short
    _write_tuned_params_file(tuned_params_path, tuned)
    logger.info(f"[short/tuning] Saved short params to {tuned_params_path}")
    return results_df


def _train_tree_model(
    name: str,
    train_fn,
    tune_fn,
    processed: dict,
    enable_tuning: bool,
    tuned_params_path: str,
    n_trials: int,
    seed: int,
):
    X_train, y_train = processed["X_train"], processed["y_train"]
    X_test, y_test = processed["X_test"], processed["y_test"]
    X_train_strict, y_train_strict = processed["X_train_strict"], processed["y_train_strict"]
    X_val, y_val = processed["X_val"], processed["y_val"]

    best_val_wape = None
    best_params = None
    if enable_tuning and len(X_val) > 0:
        logger.info(f"[short/{name}] Tuning on strict train/val...")
        best_params, best_val_wape = tune_fn(X_train_strict, y_train_strict, X_val, y_val, n_trials=n_trials, seed=seed)
    else:
        best_params = load_tuned_params(tuned_params_path, "short", name)
        if best_params:
            logger.info(f"[short/{name}] Using tuned params from tuned_params.json")

    if best_params:
        model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test, **best_params)
    else:
        model, y_pred, metrics = train_fn(X_train, y_train, X_test, y_test)

    metrics["model"] = name
    return model, y_pred, metrics, best_val_wape, best_params


def run_short_term_modeling(
    df_predict: pd.DataFrame,
    config: dict,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    set_random_seed(seed)
    output_dir = ensure_dir(config["paths"]["output_dir"])
    data_cfg = config["data"]
    short_cfg = config.get("short_term", {})
    group_cols = data_cfg["group_cols"]
    target = data_cfg["target_col"]
    min_months = int(short_cfg.get("min_months", 6))
    test_ratio = float(short_cfg.get("test_ratio", data_cfg.get("test_ratio", 0.2)))
    enable_tuning = bool(short_cfg.get("enable_tuning", False))

    short_df = df_predict[df_predict["predictability_level"].eq("short") | df_predict["predictability_level"].isna()].copy()
    if short_df.empty:
        logger.info("No short-term data found")
        return pd.DataFrame(), {}

    logger.info(f"[short] Rows={len(short_df):,}, combos={short_df.groupby(group_cols).ngroups:,}")
    train_df, test_df = split_short_term_train_test(short_df, group_cols, min_months=min_months, test_ratio=test_ratio)
    if train_df.empty or test_df.empty:
        logger.warning("[short] Insufficient data after split")
        return pd.DataFrame(), {}

    processed = prepare_short_term_features(train_df, test_df, config)
    baseline_results, best_baseline = compute_short_baselines(short_df, group_cols, target, min_months, test_ratio)
    logger.info(f"[short] Best baseline WAPE={best_baseline:.2f}%")

    all_results = list(baseline_results)
    all_test_preds = {}
    trained_models = {}
    val_wapes = {}
    tuned_short = {}
    n_trials = _trial_count(config)
    tuned_params_path = config.get("paths", {}).get("tuned_params")

    tree_trainers = {
        "RandomForest": (train_random_forest, optimize_random_forest),
        "GBDT": (train_gbdt, optimize_gbdt),
        "LightGBM": (train_lightgbm, optimize_lightgbm),
        "XGBoost": (train_xgboost, optimize_xgboost),
    }

    for name, (train_fn, tune_fn) in tree_trainers.items():
        try:
            model, y_pred, metrics, best_val_wape, best_params = _train_tree_model(
                name, train_fn, tune_fn, processed, enable_tuning, tuned_params_path, n_trials, seed
            )
            all_test_preds[name] = y_pred
            trained_models[name] = model
            all_results.append(metrics)
            if best_val_wape is not None:
                val_wapes[name] = best_val_wape
            if enable_tuning and best_params:
                tuned_short[name] = best_params
            logger.info(f"[short/{name}] Test WAPE={metrics['wape']:.2f}%")
        except Exception as e:
            logger.warning(f"[short/{name}] failed: {e}")

    if enable_tuning and tuned_short:
        tuned = _load_tuned_params_file(tuned_params_path)
        tuned["short"] = tuned_short
        _write_tuned_params_file(tuned_params_path, tuned)
        logger.info(f"[short/tuning] Saved short params to {tuned_params_path}")

    try:
        model, y_pred, metrics = train_ridge(
            processed["X_train"], processed["y_train"], processed["X_test"], processed["y_test"]
        )
        metrics["model"] = "Ridge"
        all_test_preds["Ridge"] = y_pred
        trained_models["Ridge"] = model
        all_results.append(metrics)
        logger.info(f"[short/Ridge] Test WAPE={metrics['wape']:.2f}%")
    except Exception as e:
        logger.warning(f"[short/Ridge] failed: {e}")

    if short_cfg.get("enable_dl", False):
        try:
            from src.models.dl_models import run_dl_experiment

            train_lengths = train_df.groupby(group_cols).size()
            test_lengths = test_df.groupby(group_cols).size()
            max_seq = min(train_lengths.max(), test_lengths.max() - 1)
            if max_seq >= 2:
                seq_length = int(min(config["modeling"]["dl"].get("seq_length", 12), max_seq))
                for dl_name, dl_type in [("LSTM", "lstm"), ("Transformer", "transformer")]:
                    _, _, wape_dl, dl_model = run_dl_experiment(
                        train_df.copy(),
                        test_df.copy(),
                        model_type=dl_type,
                        seq_length=seq_length,
                        epochs=short_cfg.get("dl_epochs", config["modeling"]["dl"].get("epochs", 100)),
                        batch_size=short_cfg.get("dl_batch_size", 32),
                        patience=short_cfg.get("dl_patience", config["modeling"]["dl"].get("patience", 15)),
                        lr=config["modeling"]["dl"].get("lr", 0.001),
                        weight_decay=config["modeling"]["dl"].get("weight_decay", 1e-5),
                    )
                    trained_models[dl_name] = dl_model
                    all_results.append({"model": dl_name, "wape": wape_dl, "mape": wape_dl})
                    logger.info(f"[short/{dl_name}] Test WAPE={wape_dl:.2f}%")
            else:
                logger.info("[short/DL] skipped: test spans are too short for sequence models")
        except Exception as e:
            logger.warning(f"[short/DL] failed: {e}")

    if len(all_test_preds) >= 2:
        y_test = processed["y_test"].values
        fusion = run_fusion(all_test_preds, y_test, val_wapes=val_wapes if val_wapes else None)
        for method, value in fusion.items():
            all_results.append({"model": f"Ensemble_{method.capitalize()}", "wape": value, "mape": value})

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        return results_df, trained_models

    if not np.isnan(best_baseline):
        results_df["improvement_vs_baseline"] = (best_baseline - results_df["wape"]) / best_baseline * 100
    results_df = results_df.sort_values("wape").reset_index(drop=True)

    results_path = os.path.join(output_dir, "results_short.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved short-term results: {results_path}")
    logger.info(f"[short] Best model: {results_df.iloc[0]['model']} (WAPE={results_df.iloc[0]['wape']:.2f}%)")
    logger.info(f"\nSHORT Group Results:\n{results_df.round(2).to_string(index=False)}")

    plot_model_comparison(results_df, best_baseline, output_dir, title_suffix="_short")
    for name, model in trained_models.items():
        try:
            plot_feature_importance(model, processed["feat_names"], "short", name, output_dir)
        except Exception as e:
            logger.warning(f"Feature importance plot failed for short/{name}: {e}")

    return results_df, trained_models
