import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils import ensure_dir
from src.visualization import plot_feature_importance, plot_model_comparison

logger = logging.getLogger(__name__)


def run_result_analysis(
    all_results: List[dict],
    trained_models: dict,
    baseline: Dict[str, float],
    config: dict,
    grp: str,
    processed: dict,
) -> pd.DataFrame:
    output_dir = config["paths"]["output_dir"]
    feat_names = processed.get("feat_names", [])

    results_df = pd.DataFrame(all_results)
    if not np.isnan(baseline.get("best", np.nan)):
        results_df["improvement_vs_baseline"] = (baseline["best"] - results_df["wape"]) / baseline["best"] * 100
    results_df = results_df.sort_values("wape")

    # Save results CSV
    results_path = os.path.join(ensure_dir(output_dir), f"results_{grp}.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results: {results_path}")

    # Save best model info
    if not results_df.empty:
        best_row = results_df.iloc[0]
        logger.info(f"[{grp}] Best model: {best_row['model']} (WAPE={best_row['wape']:.2f}%)")

    # Plot model comparison
    plot_model_comparison(results_df, baseline.get("best", np.nan), output_dir, title_suffix=f"_{grp}")

    # Plot feature importance for tree models
    for name, model in trained_models.items():
        try:
            plot_feature_importance(model, feat_names, grp, name, output_dir)
        except Exception as e:
            logger.warning(f"Feature importance plot failed for {name}: {e}")

    logger.info(f"\n{grp.upper()} Group Results:")
    logger.info(results_df.round(2).to_string(index=False))

    return results_df
