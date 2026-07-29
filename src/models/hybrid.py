"""
Hybrid ARIMA+XGBoost model.

Compose pre-fitted ARIMA model (base forecast) with
a pre-fitted XGBoost model (residual correction).
Final forecast is calculated as ARIMA prediction + XGBoost correction.
"""

import numpy as np

from src.models.arima import ARIMAModel
from src.models.base import BaseModel
from src.models.xgboost import XGBoostModel


class HybridModel(BaseModel):
    def __init__(
        self,
        arima_model: ARIMAModel,
        xgboost_model: XGBoostModel,
    ):
        self.arima_model = arima_model
        self.xgboost_model = xgboost_model

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def label(self) -> str:
        return "Hybrid ARIMA-XGBoost"

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "HybridModel is a composition of pre-fitted models. "
            "Fit ARIMAModel and XGBoostModel separately."
        )

    def predict(self, n_periods: int, X_features):
        arima_forecast = self.arima_model.predict(n_periods)
        xgb_correction = self.xgboost_model.predict(X_features)
        return np.asarray(arima_forecast) + np.asarray(xgb_correction)
