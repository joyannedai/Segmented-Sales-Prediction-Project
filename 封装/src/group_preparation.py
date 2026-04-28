import logging
from typing import Tuple

import pandas as pd

from src.features import prepare_features

logger = logging.getLogger(__name__)


def split_train_test_by_group(df: pd.DataFrame, group_cols: list, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_list, test_list = [], []
    for _, group in df.groupby(group_cols):
        group = group.sort_values("month")
        split_idx = int(len(group) * (1 - test_ratio))
        if split_idx == 0 or split_idx >= len(group):
            continue
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
    return pd.concat(train_list, ignore_index=True), pd.concat(test_list, ignore_index=True)


def run_group_preparation(grp_data: pd.DataFrame, config: dict):
    data_cfg = config["data"]
    feat_cfg = config["features"]

    train_df, test_df = split_train_test_by_group(
        grp_data, data_cfg["group_cols"], test_ratio=data_cfg["test_ratio"],
    )
    if len(train_df) == 0 or len(test_df) == 0:
        logger.warning("Insufficient data after split")
        return None, train_df, test_df

    processed = prepare_features(
        train_df, test_df,
        group_cols=data_cfg["group_cols"],
        target=data_cfg["target_col"],
        lags=feat_cfg["lags"],
        cat_features=feat_cfg["cat_features"],
        val_ratio=data_cfg["val_ratio"],
    )
    logger.info(f"Features prepared: {len(processed['feat_names'])}")
    return processed, train_df, test_df
