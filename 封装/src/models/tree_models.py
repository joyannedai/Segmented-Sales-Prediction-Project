import logging

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.models.base import evaluate

logger = logging.getLogger(__name__)


def train_random_forest(X_train, y_train, X_test, y_test, random_state=42, **kwargs):
    defaults = {"n_estimators": 200, "max_depth": 15, "random_state": random_state, "n_jobs": -1}
    defaults.update(kwargs)
    model = RandomForestRegressor(**defaults)
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test))
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics


def train_gbdt(X_train, y_train, X_test, y_test, random_state=42, **kwargs):
    defaults = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "random_state": random_state}
    defaults.update(kwargs)
    model = GradientBoostingRegressor(**defaults)
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test))
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics


def train_lightgbm(X_train, y_train, X_test, y_test, random_state=42, **kwargs):
    defaults = {
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": -1, "random_state": random_state, "n_jobs": -1, "verbose": -1,
    }
    defaults.update(kwargs)
    model = lgb.LGBMRegressor(**defaults)
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test))
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics


def train_xgboost(X_train, y_train, X_test, y_test, random_state=42, **kwargs):
    defaults = {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": random_state, "n_jobs": -1, "verbosity": 0,
    }
    defaults.update(kwargs)
    model = xgb.XGBRegressor(**defaults)
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test))
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics
