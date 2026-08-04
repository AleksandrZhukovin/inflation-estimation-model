"""
Random Walk model.

Implement model class based on BaseModel with Random Walk
specific logic: predicts the last observed value for all future periods.
"""

import numpy as np

from src.models.base import BaseModel


class RandomWalkModel(BaseModel):
    def __init__(self):
        self.last_value = None

    @property
    def name(self) -> str:
        return "random_walk"

    @property
    def label(self) -> str:
        return "Random Walk"

    def fit(self, train_data):
        if train_data is None or len(train_data) == 0:
            raise ValueError("Pass train data to fit the model.")

        # Store the very last value
        self.last_value = float(np.asarray(train_data)[-1])

    def predict(self, n_periods: int):
        if self.last_value is None:
            raise ValueError("Model has not been fitted yet.")

        return np.full(n_periods, self.last_value)
