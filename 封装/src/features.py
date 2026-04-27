from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_time_features(df: pd.DataFrame, time_col: str = "month") -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[time_col].dt.year
    df["month_num"] = df[time_col].dt.month
    df["quarter"] = df[time_col].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df


def create_lag_features(df: pd.DataFrame, group_cols: List[str], target: str, lags: List[int]) -> pd.DataFrame:
    df = df.sort_values(["month"]).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target].shift(lag)
    return df


def create_trend_features(df: pd.DataFrame, group_cols: List[str], target: str) -> pd.DataFrame:
    df = df.sort_values(["month"]).reset_index(drop=True)
    lag_1 = df.groupby(group_cols)[target].shift(1)
    lag_2 = df.groupby(group_cols)[target].shift(2)

    conditions = [
        lag_1.isna() | lag_2.isna(),
        lag_1 > lag_2,
        lag_1 < lag_2,
        lag_1 == lag_2,
    ]
    choices = [0, 1, -1, 0]
    df["discrete_trend"] = np.select(conditions, choices, default=0)
    df["time_idx"] = df.groupby(group_cols).cumcount() + 1
    df["rolling_mean_3"] = df.groupby(group_cols)[target].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, cat_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in cat_features:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str).unique()
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    return train, test


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: List[str],
    target: str,
    lags: List[int],
    cat_features: List[str],
    val_ratio: float = 0.15,
) -> dict:
    train = create_time_features(train_df)
    test = create_time_features(test_df)

    train = create_lag_features(train, group_cols, target, lags)
    test = create_lag_features(test, group_cols, target, lags)

    train = create_trend_features(train, group_cols, target)
    test = create_trend_features(test, group_cols, target)

    le_store, le_prod = LabelEncoder(), LabelEncoder()
    all_stores = pd.concat([train["ref_branch_code"], test["ref_branch_code"]]).unique()
    all_prods = pd.concat([train["material_nature_sum_desc"], test["material_nature_sum_desc"]]).unique()
    le_store.fit(all_stores)
    le_prod.fit(all_prods)
    train["store_code"] = le_store.transform(train["ref_branch_code"])
    test["store_code"] = le_store.transform(test["ref_branch_code"])
    train["prod_code"] = le_prod.transform(train["material_nature_sum_desc"])
    test["prod_code"] = le_prod.transform(test["material_nature_sum_desc"])

    train, test = encode_categoricals(train, test, cat_features)

    holiday_flag_cols = [col for col in train.columns if col.startswith("holiday_") and col.endswith("_flag")]

    base_features = [
        "year", "month_num", "quarter", "month_sin", "month_cos",
        "price", "total_holiday_days",
        "discrete_trend", "time_idx", "rolling_mean_3",
    ] + holiday_flag_cols + cat_features + [f"lag_{lag}" for lag in lags] + ["store_code", "prod_code"]

    if "trend" in train.columns and "trend" not in base_features:
        base_features.append("trend")

    available_features = [f for f in base_features if f in train.columns]

    X_train = train[available_features]
    y_train = train[target]
    X_test = test[available_features]
    y_test = test[target]

    nona_idx_train = X_train.dropna().index
    nona_idx_test = X_test.dropna().index
    X_train = X_train.loc[nona_idx_train]
    y_train = y_train.loc[nona_idx_train]
    X_test = X_test.loc[nona_idx_test]
    y_test = y_test.loc[nona_idx_test]

    split_idx = int(len(X_train) * (1 - val_ratio))
    X_train_strict, y_train_strict = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
    X_val, y_val = X_train.iloc[split_idx:], y_train.iloc[split_idx:]

    test_meta = test.loc[nona_idx_test, ["ref_branch_code", "material_nature_sum_desc", "month", target]]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feat_names": available_features,
        "X_train_strict": X_train_strict,
        "y_train_strict": y_train_strict,
        "X_val": X_val,
        "y_val": y_val,
        "test_meta": test_meta,
    }
