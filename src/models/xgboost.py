"""
XGBoost model

Implements model class based on BaseModel with XGBoost
specific logic: early-stopping holdout split to determine
optimal tree count, then refit on full training data.
"""

from xgboost import XGBRegressor

from src.models.base import BaseModel
from src.types import XGBoostParams


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
