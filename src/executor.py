"""
Model pipeline executor.

Orchestrates the full model pipeline: data loading,
ARIMA fitting, feature engineering, XGBoost tuning/training,
hybrid composition, and prediction. Also handles benchmarks,
multi-horizon evaluation, and SHAP analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.data import DataLoader
from src.utils.features import build_feature_matrix, build_test_features
from src.models.arima import ARIMAModel
from src.models.xgboost import XGBoostModel
from src.models.hybrid import HybridModel
from src.models.random_walk import RandomWalkModel
from src.models.seasonal_naive import SeasonalNaiveModel
from src.utils.metrics import get_metric_fn
from src.utils.tests import dm_test
from src.eda import EDAAnalyzer


class ModelExecutor:
    def __init__(self, cfg, eval_metric="rmse", visualizer=None):
        self.cfg = cfg
        self.eval_metric = eval_metric
        self.metric_fn = get_metric_fn(eval_metric)
        self.visualizer = visualizer

        self.train_df = None
        self.test_df = None

        self.arima_models = {}
        self.xgboost_model = None
        self.hybrid_models = {}

        self.benchmarks = {"RW": {}, "SN": {}, "XGB_pure": None}

        self.X_train = None
        self.y_train = None
        self.feature_names = None

        self.X_train_pure = None
        self.y_train_pure = None
        self.feature_names_pure = None

        self.best_params = None
        self.predictions_df = None

        self.entity_col = getattr(
            self.cfg.data,
            "entity_column",
            getattr(self.cfg.data, "country_column", "Country"),
        )
        self.entities = getattr(
            self.cfg.data, "entities", getattr(self.cfg.data, "countries", [])
        )
        self.test_entities = getattr(
            self.cfg.data, "test_entities", getattr(self.cfg.data, "test_countries", [])
        )

    def load_data(self):
        loader = DataLoader(self.cfg)

        df = loader.load_dataset()
        loader.validate_schema(df)
        self.full_df = df

        self.train_df, self.test_df = loader.split_train_test(df)

    def fit_arima(self):
        if self.train_df is None:
            raise RuntimeError("Call load_data() first.")

        for entity in self.entities:
            series, dates = self._get_entity_series(entity)
            model = ARIMAModel(order=None)
            model.fit(series)
            self.arima_models[entity] = {
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

    def tune_xgboost(self, n_trials=150):
        if self.X_train is None:
            raise RuntimeError("Call build_features() first.")

        self.best_params, results_df = XGBoostModel.tune(
            self.X_train,
            self.y_train,
            self.cfg,
            n_trials=n_trials,
            eval_metric=self.eval_metric,
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

        for entity in self.test_entities:
            arima_entry = self.arima_models[entity]
            self.hybrid_models[entity] = HybridModel(
                arima_model=arima_entry["model"],
                xgboost_model=self.xgboost_model,
            )

    def fit_benchmarks(self):
        if self.train_df is None:
            raise RuntimeError("Call load_data() first.")

        # Random Walk & Seasonal Naive
        for entity in self.test_entities:
            series, _ = self._get_entity_series(entity)

            rw_model = RandomWalkModel()
            rw_model.fit(series)
            self.benchmarks["RW"][entity] = rw_model

            sn_model = SeasonalNaiveModel(seasonality=12)
            sn_model.fit(series)
            self.benchmarks["SN"][entity] = sn_model

        # Pure XGBoost Benchmark (predicting target directly)
        if self.best_params is None:
            raise RuntimeError(
                "Cannot fit pure XGBoost without best_params (run tune_xgboost first)."
            )

        # Build features for pure XGBoost (no residuals)
        # We can fake the residuals_df by using the original target
        fake_res = self.train_df[
            [self.cfg.data.date_column, self.entity_col, self.cfg.data.target_column]
        ].copy()
        fake_res = fake_res.rename(
            columns={self.cfg.data.target_column: "arima_residual"}
        )
        fake_res = fake_res.dropna()

        self.X_train_pure, self.y_train_pure, self.feature_names_pure = (
            build_feature_matrix(self.train_df, fake_res, self.cfg)
        )

        self.benchmarks["XGB_pure"] = XGBoostModel(
            params=self.best_params,
            n_estimators=self.cfg.xgboost.n_estimators,
            early_stopping_rounds=self.cfg.xgboost.early_stopping_rounds,
            n_estimators_fixed=self.cfg.xgboost.n_estimators_fixed,
            random_state=self.cfg.project.random_seed,
        )
        self.benchmarks["XGB_pure"].fit(self.X_train_pure, self.y_train_pure)

    def predict_all(self):
        if not self.hybrid_models or not self.benchmarks["RW"]:
            raise RuntimeError("Call build_hybrid() and fit_benchmarks() first.")
        if self.test_df is None:
            raise RuntimeError("No test data available.")

        # Features for hybrid
        X_test = build_test_features(
            self.test_df,
            self.cfg,
            self.feature_names,
            last_train_df=self.train_df,
        )

        # Features for pure XGBoost
        X_test_pure = build_test_features(
            self.test_df,
            self.cfg,
            self.feature_names_pure,
            last_train_df=self.train_df,
        )

        rows = []
        xgb_offset = 0

        for entity in self.test_entities:
            entity_mask = self.test_df[self.entity_col] == entity
            entity_test = (
                self.test_df[entity_mask]
                .sort_values(self.cfg.data.date_column)
                .reset_index(drop=True)
            )
            n_periods = len(entity_test)

            X_entity = X_test.iloc[xgb_offset : xgb_offset + n_periods]
            X_entity_pure = X_test_pure.iloc[xgb_offset : xgb_offset + n_periods]
            xgb_offset += n_periods

            # Forecasts
            fc_hybrid = self.hybrid_models[entity].predict(n_periods, X_entity)
            fc_arima = self.arima_models[entity]["model"].predict(n_periods)
            fc_rw = self.benchmarks["RW"][entity].predict(n_periods)
            fc_sn = self.benchmarks["SN"][entity].predict(n_periods)
            fc_xgb_pure = self.benchmarks["XGB_pure"].predict(X_entity_pure)

            for i, (_, row) in enumerate(entity_test.iterrows()):
                rows.append(
                    {
                        self.entity_col: entity,
                        "Date": row[self.cfg.data.date_column],
                        "Actual": float(row[self.cfg.data.target_column]),
                        self.hybrid_models[entity].label: float(fc_hybrid[i]),
                        self.arima_models[entity]["model"].label: float(fc_arima[i]),
                        self.benchmarks["XGB_pure"].label: float(fc_xgb_pure[i]),
                        self.benchmarks["RW"][entity].label: float(fc_rw[i]),
                        self.benchmarks["SN"][entity].label: float(fc_sn[i]),
                    }
                )

        self.predictions_df = (
            pd.DataFrame(rows)
            .sort_values([self.entity_col, "Date"])
            .reset_index(drop=True)
        )
        return self.predictions_df

    def evaluate_point_metrics(self):
        if self.predictions_df is None:
            raise RuntimeError("Call predict_all() first.")

        comp_rows = []
        dm_rows = []

        # Exclude structural columns to find all model labels
        model_labels = [
            c
            for c in self.predictions_df.columns
            if c not in [self.entity_col, "Date", "Actual", "Horizon"]
        ]

        for entity in self.test_entities:
            sub = self.predictions_df[
                self.predictions_df[self.entity_col] == entity
            ].sort_values("Date")
            actual = sub["Actual"].values

            hybrid_label = next((m for m in model_labels if "Hybrid" in m), None)
            if hybrid_label and hybrid_label in sub.columns:
                hybrid_pred = sub[hybrid_label].values
            else:
                hybrid_pred = None

            dm_row = {"Country": entity}

            for model in model_labels:
                pred = sub[model].values
                comp_rows.append(
                    {
                        "Country": entity,
                        "Model": model,
                        "Metric": round(self.metric_fn(actual, pred), 4),
                    }
                )

                if hybrid_pred is not None and model != hybrid_label:
                    stat, pval = dm_test(actual, hybrid_pred, pred, h=1)
                    dm_row[f"DM vs {model} (stat)"] = (
                        round(stat, 4) if not np.isnan(stat) else None
                    )
                    dm_row[f"DM vs {model} (p)"] = (
                        round(pval, 4) if not np.isnan(pval) else None
                    )

            dm_rows.append(dm_row)

        model_comparison = pd.DataFrame(comp_rows)
        dm_pvalues = pd.DataFrame(dm_rows)

        tables_dir = Path(self.cfg.outputs.tables_dir) / "evaluation"
        tables_dir.mkdir(parents=True, exist_ok=True)
        model_comparison.to_csv(
            tables_dir / f"model_comparison_{self.eval_metric}.csv", index=False
        )
        dm_pvalues.to_csv(tables_dir / "dm_test_pvalues.csv", index=False)

        if self.visualizer:
            self.visualizer.plot_metrics_by_entity_model(
                model_comparison, self.eval_metric
            )
            self.visualizer.plot_predictions_comparison(
                self.predictions_df, self.test_entities, self.entity_col
            )

        return model_comparison, dm_pvalues

    def evaluate_multi_horizon(self):
        if self.predictions_df is None:
            raise RuntimeError("Call predict_all() first.")

        rows = []

        # Exclude structural columns to find all model labels
        model_labels = [
            c
            for c in self.predictions_df.columns
            if c not in [self.entity_col, "Date", "Actual", "Horizon"]
        ]

        for entity in self.test_entities:
            sub = (
                self.predictions_df[self.predictions_df[self.entity_col] == entity]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            actual = sub["Actual"].values
            n_test = len(sub)

            for h in self.cfg.horizons:
                if h > n_test:
                    continue
                for model_label in model_labels:
                    pred_h = sub[model_label].values[:h]
                    actual_h = actual[:h]
                    score = self.metric_fn(actual_h, pred_h)
                    rows.append(
                        {
                            self.entity_col: entity,
                            "Horizon": h,
                            "Model": model_label,
                            "Metric": round(score, 4),
                        }
                    )

        horizon_df = (
            pd.DataFrame(rows)
            .sort_values([self.entity_col, "Horizon", "Model"])
            .reset_index(drop=True)
        )

        # Save table and plot
        tables_dir = Path(self.cfg.outputs.tables_dir) / "horizon"
        tables_dir.mkdir(parents=True, exist_ok=True)

        horizon_df.to_csv(tables_dir / f"horizon_{self.eval_metric}.csv", index=False)
        if self.visualizer:
            self.visualizer.plot_metric_vs_horizon(
                horizon_df, self.test_entities, self.entity_col, self.eval_metric
            )

        return horizon_df

    def analyze_shap(self):
        if self.xgboost_model is None or self.test_df is None:
            raise RuntimeError("Call fit_xgboost() and load_data() first.")

        tables_dir = Path(self.cfg.outputs.tables_dir) / "shap"
        predictions_dir = Path(self.cfg.outputs.predictions_dir)
        for d in (tables_dir, predictions_dir):
            d.mkdir(parents=True, exist_ok=True)

        X_test = build_test_features(
            self.test_df,
            self.cfg,
            self.feature_names,
            last_train_df=self.train_df,
        )

        # The internal model is stored in self.xgboost_model.model
        explainer = shap.TreeExplainer(self.xgboost_model.model)
        explanation = explainer(X_test)
        shap_values = explanation.values

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        ranking_df = pd.DataFrame(
            {"Feature": self.feature_names, "MeanAbsSHAP": mean_abs}
        )
        ranking_df = ranking_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(
            drop=True
        )
        ranking_df.to_csv(tables_dir / "global_feature_ranking.csv", index=False)

        cols = {
            f"{f}_shap": shap_values[:, i] for i, f in enumerate(self.feature_names)
        }
        shap_df = pd.DataFrame(cols, index=X_test.index).reset_index(drop=True)
        shap_df.to_csv(predictions_dir / "shap_values_test.csv", index=False)

        if self.visualizer:
            self.visualizer.plot_shap_summary(ranking_df)

        return ranking_df, shap_df

    def save_models(self, path=None):
        models_dir = Path(path or self.cfg.outputs.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        for entity, entry in self.arima_models.items():
            entry["model"].save(models_dir / f"arima_{entity.lower()}.pkl")

        if self.xgboost_model is not None:
            self.xgboost_model.save(models_dir / "xgboost_final.pkl")

        if self.benchmarks.get("XGB_pure") is not None:
            self.benchmarks["XGB_pure"].save(models_dir / "xgboost_pure.pkl")

        for entity, model in self.hybrid_models.items():
            model.save(models_dir / f"hybrid_{entity.lower()}.pkl")

        for entity, model in self.benchmarks.get("RW", {}).items():
            model.save(models_dir / f"rw_{entity.lower()}.pkl")

        for entity, model in self.benchmarks.get("SN", {}).items():
            model.save(models_dir / f"sn_{entity.lower()}.pkl")

    def run(self, run_analysis=True):
        self.load_data()
        self.fit_arima()
        self.build_features()
        self.tune_xgboost()
        self.fit_xgboost()
        self.build_hybrid()
        self.fit_benchmarks()

        predictions_df = self.predict_all()

        if run_analysis:
            eda = EDAAnalyzer(self.cfg, visualizer=self.visualizer)
            eda.run_all(self.full_df, self.train_df)

            self.evaluate_point_metrics()
            self.evaluate_multi_horizon()
            self.analyze_shap()

        self.save_models()
        return predictions_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_entity_series(self, entity):
        subset = self.train_df[self.train_df[self.entity_col] == entity].copy()

        subset = subset.sort_values(self.cfg.data.date_column).reset_index(drop=True)

        valid = subset[self.cfg.data.target_column].notna()
        subset = subset[valid].reset_index(drop=True)

        series = subset[self.cfg.data.target_column].values.astype(float)
        dates = pd.DatetimeIndex(subset[self.cfg.data.date_column])
        return series, dates

    def _compute_residuals(self):
        frames = []
        for entity, entry in self.arima_models.items():
            model = entry["model"]
            series = entry["series"]
            dates = entry["dates"]

            residuals = model.get_residuals(series)
            n = len(residuals)

            frame = pd.DataFrame(
                {
                    "Date": dates[-n:],
                    self.entity_col: entity,
                    "arima_residual": residuals,
                }
            )
            frames.append(
                frame.dropna(subset=["arima_residual"]).reset_index(drop=True)
            )

        return pd.concat(frames, ignore_index=True).sort_values(
            [self.entity_col, "Date"]
        )
