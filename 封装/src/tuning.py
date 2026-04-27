import logging
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.models.base import wape

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed, skipping hyperparameter tuning")


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=50, seed=42):
    if not OPTUNA_AVAILABLE:
        return None, None
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
        model = lgb.LGBMRegressor(objective="regression", metric="mape", random_state=seed, n_jobs=-1, verbose=-1, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="mape", callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return wape(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50, seed=42):
    if not OPTUNA_AVAILABLE:
        return None, None
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mape", random_state=seed, n_jobs=-1, verbosity=0, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return wape(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optimize_random_forest(X_train, y_train, X_val, y_val, n_trials=30, seed=42):
    if not OPTUNA_AVAILABLE:
        return None, None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        return wape(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optimize_gbdt(X_train, y_train, X_val, y_val, n_trials=30, seed=42):
    if not OPTUNA_AVAILABLE:
        return None, None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingRegressor(random_state=seed, **params)
        model.fit(X_train, y_train)
        return wape(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value
