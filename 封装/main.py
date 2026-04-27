"""
Capstone Sales Forecast - Main Entry Point
整合数据预处理、分群、特征工程、建模、评估、融合、短时序处理
"""
import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd

from src.clustering import run_clustering
from src.data_processing import (
    add_holiday_features,
    enrich_with_raw_features,
    run_data_pipeline,
)
from src.evaluation import compute_baseline
from src.features import prepare_features
from src.models.dl_models import run_dl_experiment
from src.models.ensemble import run_fusion
from src.models.traditional import train_prophet_sample, train_ridge
from src.models.tree_models import train_gbdt, train_lightgbm, train_random_forest, train_xgboost
from src.short_term import prepare_short_term_data, run_short_term_pipeline
from src.tuning import optimize_gbdt, optimize_lightgbm, optimize_random_forest, optimize_xgboost
from src.utils import ensure_dir, load_config, set_random_seed, setup_logging
from src.visualization import plot_feature_importance, plot_model_comparison

logger = logging.getLogger(__name__)


def split_train_test_by_group(df: pd.DataFrame, group_cols: list, test_ratio: float = 0.2):
    train_list, test_list = [], []
    for _, group in df.groupby(group_cols):
        group = group.sort_values("month")
        split_idx = int(len(group) * (1 - test_ratio))
        if split_idx == 0 or split_idx >= len(group):
            continue
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
    return pd.concat(train_list, ignore_index=True), pd.concat(test_list, ignore_index=True)


def run_group_pipeline(grp: str, train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, seed: int = 42):
    set_random_seed(seed)
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["modeling"]
    output_dir = config["paths"]["output_dir"]
    enable_tuning = model_cfg.get("enable_tuning", True)

    logger.info(f"========== Group: {grp} ==========")

    # Feature engineering
    processed = prepare_features(
        train_df, test_df,
        group_cols=data_cfg["group_cols"],
        target=data_cfg["target_col"],
        lags=feat_cfg["lags"],
        cat_features=feat_cfg["cat_features"],
        val_ratio=data_cfg["val_ratio"],
    )

    X_train = processed["X_train"]
    y_train = processed["y_train"]
    X_test = processed["X_test"]
    y_test = processed["y_test"]
    X_train_strict = processed["X_train_strict"]
    y_train_strict = processed["y_train_strict"]
    X_val = processed["X_val"]
    y_val = processed["y_val"]
    feat_names = processed["feat_names"]
    test_meta = processed["test_meta"]

    logger.info(f"Features: {len(feat_names)}")

    # Baseline
    full_group = pd.concat([train_df, test_df]).sort_values("month")
    baseline = compute_baseline(full_group, data_cfg["group_cols"], data_cfg["target_col"], data_cfg["test_ratio"])

    all_results = []
    all_test_preds = {}
    trained_models = {}
    val_wapes = {}

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
            trained_models[name] = model
            metrics["model"] = name
            all_results.append(metrics)
            logger.info(f"[{name}] Test WAPE={metrics['wape']:.2f}%")

            plot_feature_importance(model, feat_names, grp, name, output_dir)
        except Exception as e:
            logger.warning(f"[{name}] failed: {e}")

    # Non-tree models
    try:
        model, y_pred, metrics = train_ridge(X_train, y_train, X_test, y_test)
        all_test_preds["Ridge"] = y_pred
        trained_models["Ridge"] = model
        metrics["model"] = "Ridge"
        all_results.append(metrics)
        logger.info(f"[Ridge] Test WAPE={metrics['wape']:.2f}%")
    except Exception as e:
        logger.warning(f"Ridge failed: {e}")

    try:
        _, _, prophet_wape, _ = train_prophet_sample(
            train_df, test_df, data_cfg["group_cols"],
            sample_size=model_cfg.get("prophet_sample_size", 50), random_state=seed,
        )
        if prophet_wape is not None:
            all_results.append({"model": "Prophet", "wape": prophet_wape, "mape": prophet_wape})
            logger.info(f"[Prophet] Test WAPE={prophet_wape:.2f}%")
    except Exception as e:
        logger.warning(f"Prophet failed: {e}")

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
            all_results.append({"model": f"Ensemble_{method.capitalize()}", "wape": val, "mape": val})

    results_df = pd.DataFrame(all_results)
    if not np.isnan(baseline.get("best", np.nan)):
        results_df["improvement_vs_baseline"] = (baseline["best"] - results_df["wape"]) / baseline["best"] * 100
    results_df = results_df.sort_values("wape")

    # Save results
    results_path = os.path.join(ensure_dir(output_dir), f"results_{grp}.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results: {results_path}")

    plot_model_comparison(results_df, baseline.get("best", np.nan), output_dir, title_suffix=f"_{grp}")

    return results_df, trained_models, processed


def main():
    parser = argparse.ArgumentParser(description="Capstone Sales Forecast Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--stage", default="all", choices=["data", "cluster", "train", "short", "all"], help="Pipeline stage")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna tuning (override config)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    set_random_seed(config["project"]["random_seed"])

    # Command-line flag overrides config
    if args.skip_tuning:
        config["modeling"]["enable_tuning"] = False

    paths = config["paths"]
    data_cfg = config["data"]
    ensure_dir(paths["output_dir"])
    ensure_dir(paths["log_dir"])

    # Stage 1: Data processing
    if args.stage in ("data", "all"):
        logger.info("=" * 60)
        logger.info("Stage 1: Data Processing")
        logger.info("=" * 60)
        df_processed = run_data_pipeline(
            paths["input_parquet"],
            paths["output_dir"],
            missing_threshold=data_cfg["missing_rate_threshold"],
        )
        df_processed.to_parquet(paths["monthly_filled"], index=False)
        logger.info(f"Saved processed data: {paths['monthly_filled']}")

        # Enrich with raw features
        df_enriched = enrich_with_raw_features(paths["monthly_filled"], paths["input_parquet"])
        df_enriched.to_parquet(paths["data_with_features"], index=False)
        logger.info(f"Saved enriched data: {paths['data_with_features']}")

        # Add holidays
        df_holiday = add_holiday_features(df_enriched)
        df_holiday.to_parquet(paths["data_with_holidays"], index=False)
        logger.info(f"Saved holiday data: {paths['data_with_holidays']}")
    else:
        df_holiday = pd.read_parquet(paths["data_with_holidays"])

    # Stage 2: Clustering
    if args.stage in ("cluster", "all"):
        logger.info("=" * 60)
        logger.info("Stage 2: Clustering")
        logger.info("=" * 60)
        clusters = run_clustering(
            df_holiday,
            long_term_threshold=config["clustering"]["long_term_threshold"],
            score_weights=config["clustering"]["score_weights"],
            low_quantile=config["clustering"]["low_quantile"],
            high_quantile=config["clustering"]["high_quantile"],
        )
        clusters.to_csv(paths["cluster_result"], index=False)
        logger.info(f"Saved clusters: {paths['cluster_result']}")

        # Merge predictability labels
        labels = clusters[["ref_branch_code", "material_nature_sum_desc", "predictability_level"]].drop_duplicates()
        df_predict = df_holiday.merge(labels, on=["ref_branch_code", "material_nature_sum_desc"], how="left")
        df_predict.to_parquet(paths["data_predict"], index=False)
        logger.info(f"Saved predict data: {paths['data_predict']}")

        # Split into files
        for level in ["high", "medium", "low"]:
            sub = df_predict[df_predict["predictability_level"] == level]
            if not sub.empty:
                sub.to_parquet(os.path.join(paths["output_dir"], f"{level}.parquet"), index=False)
    else:
        df_predict = pd.read_parquet(paths["data_predict"])

    # Stage 3: Training
    if args.stage in ("train", "all"):
        logger.info("=" * 60)
        logger.info("Stage 3: Model Training")
        logger.info("=" * 60)

        all_group_models = {}
        all_group_processed = {}

        for grp in ["high", "medium", "low"]:
            grp_data = df_predict[df_predict["predictability_level"] == grp]
            if grp_data.empty:
                logger.warning(f"No data for group {grp}")
                continue

            train_df, test_df = split_train_test_by_group(
                grp_data, data_cfg["group_cols"], test_ratio=data_cfg["test_ratio"],
            )
            if len(train_df) == 0 or len(test_df) == 0:
                logger.warning(f"Insufficient data for group {grp}")
                continue

            results_df, trained_models, processed = run_group_pipeline(grp, train_df, test_df, config)
            all_group_models[grp] = trained_models
            all_group_processed[grp] = processed

            logger.info(f"\n{grp.upper()} Group Results:")
            logger.info(results_df.round(2).to_string(index=False))

        # Save all models
        model_path = os.path.join(paths["output_dir"], "all_models.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(all_group_models, f)
        logger.info(f"Saved all models: {model_path}")

    # Stage 4: Short-term
    if args.stage in ("short", "all"):
        logger.info("=" * 60)
        logger.info("Stage 4: Short-term Modeling")
        logger.info("=" * 60)

        short_data = prepare_short_term_data(
            df_predict,
            cat_features=config["features"]["cat_features"],
            test_ratio=data_cfg["test_ratio"],
            val_ratio=data_cfg["val_ratio"],
            seed=config["project"]["random_seed"],
        )
        short_results = run_short_term_pipeline(
            short_data,
            use_tuning=config["modeling"].get("enable_tuning", True),
            seed=config["project"]["random_seed"],
        )

        short_path = os.path.join(paths["output_dir"], "results_short.csv")
        short_results.to_csv(short_path, index=False)
        logger.info(f"Saved short-term results: {short_path}")
        logger.info(f"\nShort-term Results:\n{short_results.round(2).to_string(index=False)}")

        plot_model_comparison(short_results, short_data["baseline"].get("best", np.nan), paths["output_dir"], title_suffix="_short")

    logger.info("=" * 60)
    logger.info("Pipeline completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
