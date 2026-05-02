"""
Capstone Sales Forecast - Main Entry Point.

5-stage pipeline: data -> cluster -> preparation -> modeling -> analysis.
"""
import argparse
import logging
import os
import pickle

import pandas as pd

from src.clustering import run_clustering
from src.data_processing import run_data_pipeline
from src.group_preparation import run_group_preparation
from src.modeling import run_modeling
from src.result_analysis import run_result_analysis
from src.short_term_modeling import run_short_term_modeling, tune_short_term_params
from src.utils import ensure_dir, load_config, set_random_seed, setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Capstone Sales Forecast Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--stage", default="all", choices=["data", "cluster", "train", "all"], help="Pipeline stage")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip long-term Optuna tuning")
    parser.add_argument(
        "--tune-short-params",
        action="store_true",
        help="Tune short-term tree models and write params into tuned_params.json, then exit",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    set_random_seed(config["project"]["random_seed"])

    if args.skip_tuning:
        config["modeling"]["enable_tuning"] = False

    paths = config["paths"]
    data_cfg = config["data"]
    ensure_dir(paths["output_dir"])
    ensure_dir(paths["log_dir"])

    if args.stage in ("data", "all"):
        logger.info("=" * 60)
        logger.info("Stage 1: Data Processing")
        logger.info("=" * 60)
        df_processed = run_data_pipeline(
            paths["input_parquet"],
            paths["input_parquet"],
            missing_threshold=data_cfg["missing_rate_threshold"],
        )
        df_processed.to_parquet(paths["data_predict"], index=False)
        logger.info(f"Saved processed data: {paths['data_predict']}")
    else:
        df_processed = pd.read_parquet(paths["data_predict"])

    if args.stage in ("cluster", "all"):
        logger.info("=" * 60)
        logger.info("Stage 2: Clustering")
        logger.info("=" * 60)
        clusters = run_clustering(
            df_processed,
            long_term_threshold=config["clustering"]["long_term_threshold"],
            score_weights=config["clustering"]["score_weights"],
            low_quantile=config["clustering"]["low_quantile"],
            high_quantile=config["clustering"]["high_quantile"],
        )
        clusters.to_csv(paths["cluster_result"], index=False)
        logger.info(f"Saved clusters: {paths['cluster_result']}")

        labels = clusters[["ref_branch_code", "material_nature_sum_desc", "predictability_level"]].drop_duplicates()
        df_predict = df_processed.merge(labels, on=["ref_branch_code", "material_nature_sum_desc"], how="left")
        df_predict.to_parquet(paths["data_predict"], index=False)
        logger.info(f"Saved predict data: {paths['data_predict']}")
    else:
        df_predict = pd.read_parquet(paths["data_predict"])

    if args.tune_short_params:
        logger.info("=" * 60)
        logger.info("Short-Term Parameter Tuning")
        logger.info("=" * 60)
        tune_short_term_params(df_predict, config, seed=config["project"]["random_seed"])
        logger.info("Short-term tuning completed")
        return

    if args.stage in ("train", "all"):
        logger.info("=" * 60)
        logger.info("Stage 3-5: Group Preparation -> Modeling -> Result Analysis")
        logger.info("=" * 60)

        all_group_models = {}

        for grp in ["high", "medium", "low"]:
            grp_data = df_predict[df_predict["predictability_level"] == grp]
            if grp_data.empty:
                logger.warning(f"No data for group {grp}")
                continue

            processed, train_df, test_df = run_group_preparation(grp_data, config)
            if processed is None:
                logger.warning(f"Group {grp} preparation failed")
                continue

            all_results, trained_models, all_test_preds, baseline, val_wapes = run_modeling(
                processed, train_df, test_df, config, grp, seed=config["project"]["random_seed"]
            )

            run_result_analysis(all_results, trained_models, baseline, config, grp, processed)

            all_group_models[grp] = trained_models
            logger.info(f"Group {grp} completed")

        if config.get("short_term", {}).get("enable_modeling", True):
            logger.info("=" * 60)
            logger.info("Stage 4-5: Short-Term Modeling")
            logger.info("=" * 60)
            short_results, short_models = run_short_term_modeling(
                df_predict, config, seed=config["project"]["random_seed"]
            )
            if short_models:
                all_group_models["short"] = short_models
                logger.info("Short-term modeling completed")
            elif short_results.empty:
                logger.warning("Short-term modeling skipped or produced no results")

        model_path = os.path.join(paths["output_dir"], "all_models.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(all_group_models, f)
        logger.info(f"Saved all models: {model_path}")

    logger.info("=" * 60)
    logger.info("Pipeline completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
