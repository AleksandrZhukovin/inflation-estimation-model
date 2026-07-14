"""
Model pipeline executor.

Orchestrates the full model pipeline: data loading,
ARIMA fitting, feature engineering, XGBoost tuning/training,
hybrid composition, and prediction.
"""

import pandas as pd
from pathlib import Path

from src.data import load_dataset, validate_schema, split_train_test
from src.features import build_feature_matrix
from src.models.arima import ARIMAModel
from src.models.xgboost import XGBoostModel
from src.models.hybrid import HybridModel
from src.tuning import tune_xgboost_optuna, N_TRIALS
from src.features import build_test_features


class ModelExecutor:
    def __init__(self, cfg):
        self.cfg = cfg

        self.train_df = None
        self.test_df = None

        self.arima_models = {}
        self.xgboost_model = None
        self.hybrid_models = {}

        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.best_params = None

    def load_data(self):
        df = load_dataset(self.cfg.data.dataset_path)
        validate_schema(df)
        self.train_df, self.test_df = split_train_test(
            df,
            self.cfg.data.train_end,
            self.cfg.data.test_start,
            test_countries=self.cfg.data.test_countries,
            test_end=getattr(self.cfg.data, "test_end", None),
        )

    def fit_arima(self):
        if self.train_df is None:
            raise RuntimeError("Call load_data() first.")

        for country in self.cfg.data.countries:
            series, dates = self._get_country_series(country)
            model = ARIMAModel(order=None)
            model.fit(series)
            self.arima_models[country] = {
                "model": model,
                "series": series,
                "dates": dates,
            }

    def build_features(self):
        if not self.arima_models:
            raise RuntimeError("Call fit_arima() first.")

        residuals_df = self._compute_residuals()
        self.X_train, self.y_train, self.feature_names = build_feature_matrix(
            self.train_df, residuals_df, self.cfg
        )

    def tune_xgboost(self, n_trials=N_TRIALS):
        if self.X_train is None:
            raise RuntimeError("Call build_features() first.")

        self.best_params, results_df = tune_xgboost_optuna(
            self.X_train, self.y_train, self.cfg, n_trials=n_trials
        )
        return results_df

    def fit_xgboost(self, params=None):
        if self.X_train is None:
            raise RuntimeError("Call build_features() first.")

        if params is not None:
            self.best_params = params

        if self.best_params is None:
            raise RuntimeError(
                "No params available. Call tune_xgboost() first "
                "or pass params explicitly."
            )

        self.xgboost_model = XGBoostModel(
            params=self.best_params,
            n_estimators=self.cfg.xgboost.n_estimators,
            early_stopping_rounds=self.cfg.xgboost.early_stopping_rounds,
            n_estimators_fixed=self.cfg.xgboost.n_estimators_fixed,
            random_state=self.cfg.project.random_seed,
        )
        self.xgboost_model.fit(self.X_train, self.y_train)

    def build_hybrid(self):
        if not self.arima_models:
            raise RuntimeError("Call fit_arima() first.")
        if self.xgboost_model is None:
            raise RuntimeError("Call fit_xgboost() first.")

        for country in self.cfg.data.test_countries:
            arima_entry = self.arima_models[country]
            self.hybrid_models[country] = HybridModel(
                arima_model=arima_entry["model"],
                xgboost_model=self.xgboost_model,
            )

    def predict(self):
        if not self.hybrid_models:
            raise RuntimeError("Call build_hybrid() first.")
        if self.test_df is None:
            raise RuntimeError("No test data available.")

        X_test = build_test_features(
            self.test_df,
            self.cfg,
            self.feature_names,
            last_train_df=self.train_df,
        )

        rows = []
        xgb_offset = 0

        for country in self.cfg.data.test_countries:
            country_mask = self.test_df[self.cfg.data.country_column] == country
            country_test = (
                self.test_df[country_mask]
                .sort_values(self.cfg.data.date_column)
                .reset_index(drop=True)
            )
            n_periods = len(country_test)

            X_country = X_test.iloc[xgb_offset : xgb_offset + n_periods]
            xgb_offset += n_periods

            hybrid = self.hybrid_models[country]
            forecast = hybrid.predict(n_periods, X_country)

            for i, (_, row) in enumerate(country_test.iterrows()):
                rows.append(
                    {
                        "Country": country,
                        "Date": row[self.cfg.data.date_column],
                        "Actual": float(row[self.cfg.data.target_column]),
                        "Hybrid": float(forecast[i]),
                    }
                )

        return (
            pd.DataFrame(rows).sort_values(["Country", "Date"]).reset_index(drop=True)
        )

    def save_models(self, path=None):
        models_dir = Path(path or self.cfg.outputs.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        for country, entry in self.arima_models.items():
            entry["model"].save(models_dir / f"arima_{country.lower()}.pkl")

        if self.xgboost_model is not None:
            self.xgboost_model.save(models_dir / "xgboost_final.pkl")

    def run(self):
        self.load_data()
        self.fit_arima()
        self.build_features()
        self.tune_xgboost()
        self.fit_xgboost()
        self.build_hybrid()
        predictions_df = self.predict()
        self.save_models()
        return predictions_df

    def _get_country_series(self, country):
        cfg = self.cfg
        subset = self.train_df[self.train_df[cfg.data.country_column] == country].copy()

        train_start = (
            cfg.data.ua_train_start if country == "UA" else cfg.data.lt_lv_train_start
        )
        subset = (
            subset[subset[cfg.data.date_column] >= pd.Timestamp(train_start)]
            .sort_values(cfg.data.date_column)
            .reset_index(drop=True)
        )

        valid = subset[cfg.data.target_column].notna()
        subset = subset[valid].reset_index(drop=True)

        series = subset[cfg.data.target_column].values.astype(float)
        dates = pd.DatetimeIndex(subset[cfg.data.date_column])
        return series, dates

    def _compute_residuals(self):
        frames = []
        for country, entry in self.arima_models.items():
            model = entry["model"]
            series = entry["series"]
            dates = entry["dates"]

            fitted = model.get_in_sample_predictions()
            n = min(len(series), len(fitted))
            residuals = series[-n:] - fitted[-n:]

            frame = pd.DataFrame(
                {
                    "Date": dates[-n:],
                    "Country": country,
                    "arima_residual": residuals,
                }
            )
            frames.append(
                frame.dropna(subset=["arima_residual"]).reset_index(drop=True)
            )

        return pd.concat(frames, ignore_index=True).sort_values(["Country", "Date"])
