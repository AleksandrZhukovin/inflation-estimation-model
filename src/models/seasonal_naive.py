"""
Seasonal Naive model.

Implement model class based on BaseModel with Seasonal Naive
specific logic: predict by repeating the last observed season.
"""

import numpy as np

from src.models.base import BaseModel


class SeasonalNaiveModel(BaseModel):
    def __init__(self, seasonality: int = 12):
        self.seasonality = seasonality
        self.last_season = None

    @property
    def name(self) -> str:
        return "seasonal_naive"

    @property
    def label(self) -> str:
        return "Seasonal Naive"

    def fit(self, train_data):
        if train_data is None or len(train_data) == 0:
            raise ValueError("Pass train data to fit the model.")

        train_data = np.asarray(train_data)
        if len(train_data) < self.seasonality:
            raise ValueError(
                f"Training data length ({len(train_data)}) must be at least "
                f"the seasonality ({self.seasonality})."
            )

        # Store the last season
        self.last_season = train_data[-self.seasonality :].astype(float)

    def predict(self, n_periods: int):
        if self.last_season is None:
            raise ValueError("Model has not been fitted yet.")

        # Repeat the last season to cover n_periods
        repeats = int(np.ceil(n_periods / self.seasonality))
        tiled = np.tile(self.last_season, repeats)
        return tiled[:n_periods]
