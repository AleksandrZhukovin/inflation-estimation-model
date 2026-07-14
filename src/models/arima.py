"""
ARIMA model

Implments model class based on BaseModel with ARIMA
specific logic.
"""

import numpy as np
import pmdarima as pm

from src.models.base import BaseModel


class ARIMAModel(BaseModel):
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, train_data):
        if train_data is None or len(train_data) == 0:
            raise ValueError("Pass train data to fit the model.")

        if self.order is not None:
            self.model = pm.arima.ARIMA(self.order)
            self.model.fit(train_data)
        else:
            self.model = pm.auto_arima(train_data, seasonal=False, d=1)

    def predict(self, n_periods):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        forecast = self.model.predict(n_periods=n_periods)
        return forecast

    def get_in_sample_predictions(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return np.asarray(self.model.predict_in_sample(), dtype=float)
