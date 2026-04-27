import logging
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


def compute_span(group: pd.DataFrame) -> int:
    months = group["month"].sort_values()
    if len(months) < 2:
        return 1
    start, end = months.iloc[0], months.iloc[-1]
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def compute_cv_stl_features(sales: np.ndarray) -> Tuple[float, float, float]:
    sales = np.asarray(sales, dtype=float)
    if np.std(sales) < 1e-8:
        return 0.0 if np.mean(sales) > 0 else 100.0, 0.0, 100.0

    mean_s = np.mean(sales)
    cv = np.std(sales) / mean_s if mean_s > 1e-6 else 100.0

    try:
        stl = STL(sales, period=12, robust=True).fit()
        var_seasonal = np.var(stl.seasonal)
        var_resid = np.var(stl.resid)
        seasonal_strength = 1 - var_resid / (var_seasonal + var_resid + 1e-8)
        cv_resid = np.std(stl.resid) / (mean_s + 1e-8)
    except Exception:
        seasonal_strength = 0.0
        cv_resid = 100.0

    return cv, seasonal_strength, cv_resid


def run_clustering(
    df: pd.DataFrame,
    long_term_threshold: int = 24,
    score_weights: dict = None,
    low_quantile: float = 0.5,
    high_quantile: float = 0.8,
) -> pd.DataFrame:
    if score_weights is None:
        score_weights = {"cv": 0.4, "seasonal_strength": 0.4, "residual_cv": 0.2}

    logger.info("Evaluating time spans...")
    spans = df.groupby(["ref_branch_code", "material_nature_sum_desc"]).apply(compute_span)
    long_term_mask = spans >= long_term_threshold

    long_term_combos = spans[long_term_mask].index.tolist()
    short_term_combos = spans[~long_term_mask].index.tolist()

    logger.info(f"Total combos: {len(spans)}, Long: {len(long_term_combos)}, Short: {len(short_term_combos)}")

    df_long = df.set_index(["ref_branch_code", "material_nature_sum_desc"]).loc[long_term_combos].reset_index()

    logger.info("Extracting STL features for long-term series...")
    features = []
    for (branch, material), group in df_long.groupby(["ref_branch_code", "material_nature_sum_desc"]):
        group = group.sort_values("month")
        sales = group["monthly_sales"].values
        cv, fs, cv_resid = compute_cv_stl_features(sales)
        features.append({
            "ref_branch_code": branch,
            "material_nature_sum_desc": material,
            "CV": cv,
            "seasonal_strength": fs,
            "residual_cv": cv_resid,
            "length": len(sales),
        })

    df_feat = pd.DataFrame(features)

    # Clip outliers
    df_feat["CV"] = df_feat["CV"].clip(upper=df_feat["CV"].quantile(0.99))
    df_feat["residual_cv"] = df_feat["residual_cv"].clip(upper=df_feat["residual_cv"].quantile(0.99))

    # Percentile ranks
    df_feat["CV_pct"] = df_feat["CV"].rank(pct=True)
    df_feat["Fs_pct"] = df_feat["seasonal_strength"].rank(pct=True)
    df_feat["ResCV_pct"] = df_feat["residual_cv"].rank(pct=True)

    w_cv = score_weights.get("cv", 0.4)
    w_fs = score_weights.get("seasonal_strength", 0.4)
    w_res = score_weights.get("residual_cv", 0.2)

    df_feat["score"] = (
        w_cv * (1 - df_feat["CV_pct"]) +
        w_fs * df_feat["Fs_pct"] +
        w_res * (1 - df_feat["ResCV_pct"])
    ) * 100

    low_thresh = df_feat["score"].quantile(low_quantile)
    high_thresh = df_feat["score"].quantile(high_quantile)

    def assign_group(score):
        if score >= high_thresh:
            return "high"
        elif score >= low_thresh:
            return "medium"
        else:
            return "low"

    df_feat["predictability_level"] = df_feat["score"].apply(assign_group)

    logger.info(f"Cluster distribution:\n{df_feat['predictability_level'].value_counts()}")
    logger.info(f"Thresholds: low/medium={low_thresh:.2f}, medium/high={high_thresh:.2f}")

    return df_feat
