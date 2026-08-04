"""
Pydantic schemas for experiment API request/response payloads.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    dataset_path: str = "data/diploma_dataset_1.csv"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: Optional[str] = None
    countries: list[str] = ["UA", "LT", "LV"]
    test_countries: list[str] = ["UA"]


class ArimaConfig(BaseModel):
    d: int = 1
    max_p: int = 5
    max_q: int = 5
    seasonal: bool = False


class XGBoostConfig(BaseModel):
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    n_estimators_fixed: int = 1000
    n_tuning_trials: int = 150


class WalkForwardConfig(BaseModel):
    initial_train_years: int = 10
    test_window_years: int = 1


class FiguresConfig(BaseModel):
    dpi: int = 300
    format: str = "png"
    style: str = "seaborn-v0_8-whitegrid"


class ExperimentRequest(BaseModel):
    """Top-level request payload for POST /api/v1/experiments/run."""

    eval_metric: str = Field(default="rmse", pattern="^(rmse|mae|mape)$")
    mode: str = Field(default="walk_forward", pattern="^(single|walk_forward)$")
    data: DataConfig = DataConfig()
    arima: ArimaConfig = ArimaConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    figures: FiguresConfig = FiguresConfig()
    save_plots: bool = True

    def to_config_namespace(self):
        """
        Convert Pydantic model to the SimpleNamespace tree that the
        existing src/ layer expects.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            project=SimpleNamespace(random_seed=42),
            data=SimpleNamespace(
                dataset_path=self.data.dataset_path,
                date_column="Date",
                target_column="CPI_Index",
                country_column="Country",
                countries=self.data.countries,
                flag_columns=[],
                train_end=self.data.train_end,
                test_start=self.data.test_start,
                test_end=self.data.test_end,
                test_countries=self.data.test_countries,
                ua_train_start="2007-01-01",
                lt_lv_train_start="2007-01-01",
                patches=[
                    SimpleNamespace(
                        column="Global_GPR_Index",
                        target_entity="UA",
                        source_entity="LT",
                    )
                ],
            ),
            arima=SimpleNamespace(
                d=self.arima.d,
                max_p=self.arima.max_p,
                max_q=self.arima.max_q,
                seasonal=self.arima.seasonal,
                information_criterion="aic",
                stepwise=True,
                ljung_box_lags=10,
            ),
            xgb_grid=SimpleNamespace(
                eta=[0.05, 0.1, 0.15],
                max_depth=[4, 5, 6, 8],
                min_child_weight=[1, 5],
                gamma=[0],
                reg_alpha=[0, 0.1],
                reg_lambda=[0.5, 1.0, 2.0],
                subsample=[0.8, 1.0],
                colsample_bytree=[0.8, 1.0],
            ),
            xgboost=SimpleNamespace(
                n_estimators=self.xgboost.n_estimators,
                early_stopping_rounds=self.xgboost.early_stopping_rounds,
                n_estimators_fixed=self.xgboost.n_estimators_fixed,
                eval_split_ratio=0.1,
                cv_n_splits=5,
                cv_n_estimators=500,
                cv_early_stopping_rounds=100,
            ),
            horizons=[1, 3, 6, 12],
            walk_forward=SimpleNamespace(
                initial_train_years=self.walk_forward.initial_train_years,
                test_window_years=self.walk_forward.test_window_years,
            ),
            outputs=SimpleNamespace(
                models_dir="outputs/models",
                predictions_dir="outputs/predictions",
                tables_dir="outputs/tables",
                figures_dir="outputs/figures",
            ),
            figures=SimpleNamespace(
                dpi=self.figures.dpi,
                format=self.figures.format,
                style=self.figures.style,
            ),
        )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class MetricRow(BaseModel):
    country: str
    model: str
    value: float


class ExperimentResponse(BaseModel):
    """Response payload returned after experiment completion."""

    job_id: str
    status: str
    eval_metric: str
    metrics: list[MetricRow] = []
    dm_test_summary: list[dict] = []
    plot_urls: list[str] = []
    csv_urls: list[str] = []
    summary_markdown: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    result: Optional[ExperimentResponse] = None
