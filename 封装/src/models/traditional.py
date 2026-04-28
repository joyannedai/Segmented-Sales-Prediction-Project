import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.evaluation import evaluate

logger = logging.getLogger(__name__)


def train_ridge(X_train, y_train, X_test, y_test, random_state=42):
    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test))
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics


def train_prophet_sample(train_df, test_df, group_cols, sample_size=50, random_state=42):
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed")
        return None, None, None, None

    all_groups = train_df[group_cols].drop_duplicates()
    if len(all_groups) > sample_size:
        sample_groups = all_groups.sample(n=sample_size, random_state=random_state)
    else:
        sample_groups = all_groups

    y_true_list, y_pred_list = [], []
    model = None
    for _, (store, prod) in sample_groups.iterrows():
        train_group = train_df[(train_df[group_cols[0]] == store) & (train_df[group_cols[1]] == prod)]
        test_group = test_df[(test_df[group_cols[0]] == store) & (test_df[group_cols[1]] == prod)]
        if len(test_group) == 0 or len(train_group) < 6:
            continue
        df_train = train_group[["month", "monthly_sales"]].rename(columns={"month": "ds", "monthly_sales": "y"})
        df_test = test_group[["month"]].rename(columns={"month": "ds"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_train)
        forecast = model.predict(df_test)
        y_true_list.extend(test_group["monthly_sales"].values)
        y_pred_list.extend(forecast["yhat"].values)

    if len(y_true_list) == 0:
        return None, None, None, None

    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)
    metrics = evaluate(y_true_arr, y_pred_arr)
    return y_pred_arr, metrics["mape"], metrics["wape"], model
