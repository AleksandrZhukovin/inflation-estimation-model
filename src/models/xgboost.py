"""
XGBoost model

Implement model class based on BaseModel with XGBoost
specific logic: early-stopping holdout split to determine
optimal tree count, then refit on full training data.
Also includes hyperparameter tuning using Optuna.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.models.base import BaseModel
from src.types import XGBoostParams
from src.utils.metrics import get_metric_fn


N_TRIALS = 150


class XGBoostModel(BaseModel):
    def __init__(
        self,
        params: XGBoostParams,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 100,
        n_estimators_fixed: int = 1000,
        random_state: int = 42,
    ):
        self.params = params
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators_fixed = n_estimators_fixed
        self.random_state = random_state
        self.model = None

    @property
    def name(self) -> str:
        return "xgboost_model"

    @property
    def label(self) -> str:
        return "XGBoost Model"

    def _build_regressor(self, n_estimators, early_stopping_rounds=None):
        return XGBRegressor(
            learning_rate=self.params["eta"],
            max_depth=int(self.params["max_depth"]),
            min_child_weight=int(self.params.get("min_child_weight", 1)),
            gamma=self.params["gamma"],
            reg_alpha=self.params.get("reg_alpha", 0),
            reg_lambda=self.params["reg_lambda"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0,
        )

    def fit(self, X_train, y_train):
        if X_train is None or len(X_train) == 0:
            raise ValueError("Pass train data to fit the model.")

        n = len(X_train)
        holdout_size = min(72, max(int(n * 0.12), 36))

        X_fit = X_train.iloc[:-holdout_size]
        y_fit = y_train.iloc[:-holdout_size]
        X_val = X_train.iloc[-holdout_size:]
        y_val = y_train.iloc[-holdout_size:]

        model_es = self._build_regressor(self.n_estimators, self.early_stopping_rounds)
        model_es.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

        raw_best_n = model_es.best_iteration + 1
        best_n = self.n_estimators_fixed if raw_best_n <= 10 else raw_best_n

        self.model = self._build_regressor(best_n)
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    @staticmethod
    def _cv_fit_evaluate(
        params,
        X,
        y,
        tscv,
        cv_n_estimators,
        cv_early_stopping_rounds,
        random_state,
        metric_fn,
    ):
        fold_scores = []

        for train_idx, val_idx in tscv.split(X):
            n_fold = len(train_idx)
            es_start = int(n_fold * 0.8)
            fit_idx = train_idx[:es_start]
            es_idx = train_idx[es_start:]

            model = XGBRegressor(
                learning_rate=params["eta"],
                max_depth=int(params["max_depth"]),
                min_child_weight=int(params.get("min_child_weight", 1)),
                gamma=params["gamma"],
                reg_alpha=params.get("reg_alpha", 0),
                reg_lambda=params["reg_lambda"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                n_estimators=cv_n_estimators,
                early_stopping_rounds=cv_early_stopping_rounds,
                random_state=random_state,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(
                X.iloc[fit_idx],
                y.iloc[fit_idx],
                eval_set=[(X.iloc[es_idx], y.iloc[es_idx])],
                verbose=False,
            )

            pred = model.predict(X.iloc[val_idx])
            actual = y.iloc[val_idx].values
            fold_scores.append(metric_fn(actual, pred))

        return float(np.mean(fold_scores)), float(np.std(fold_scores))

    @classmethod
    def tune(cls, X, y, cfg, n_trials=N_TRIALS, eval_metric="rmse"):
        """
        Tune XGBoost hyperparameters using Optuna and TimeSeriesSplit.
        Return best parameters and a DataFrame of tuning results.
        """
        metric_fn = get_metric_fn(eval_metric)

        tscv = TimeSeriesSplit(n_splits=cfg.xgboost.cv_n_splits)
        cv_n_est = cfg.xgboost.cv_n_estimators
        cv_es = cfg.xgboost.cv_early_stopping_rounds
        seed = cfg.project.random_seed

        def objective(trial):
            params = {
                "eta": trial.suggest_float("eta", 0.01, 0.30, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            mean_score, _ = cls._cv_fit_evaluate(
                params, X, y, tscv, cv_n_est, cv_es, seed, metric_fn
            )
            return mean_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        records = []
        for t in study.trials:
            if t.value is not None:
                records.append(
                    {
                        **t.params,
                        "mean_score": round(t.value, 4),
                        "std_score": float("nan"),
                    }
                )
        results_df = (
            pd.DataFrame(records).sort_values("mean_score").reset_index(drop=True)
        )

        return best_params, results_df
