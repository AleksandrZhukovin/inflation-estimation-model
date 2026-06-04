# Inflation Estimation Model

## Dataset
Link: https://docs.google.com/spreadsheets/d/16a271hZg2W89L9PSKg2VTPfsj9-z33OvdYHuWeG9HbE/edit?usp=sharing

## Project structure

```
.
├── main.py                  # Pipeline entry point
├── config.yaml              # All configuration
├── pyproject.toml           # Dependencies
│
├── src/
│   ├── config.py            # Config loader (YAML → SimpleNamespace)
│   ├── data.py              # Dataset loading, validation, train/test split
│   ├── eda.py               # Exploratory analysis (ADF, STL, plots)
│   ├── arima.py             # Auto-ARIMA fitting per country
│   ├── features.py          # Feature matrix + ARIMA residuals
│   ├── tuning.py            # Hyperparameter search (Optuna TPE, 150 trials)
│   ├── xgboost_model.py     # XGBoost training with early stopping
│   ├── hybrid.py            # Hybrid forecast = ARIMA + XGBoost residual
│   ├── benchmarks.py        # Pure-ARIMA and pure-XGBoost baselines
│   ├── evaluation.py        # RMSE comparison + Diebold-Mariano test
│   ├── multi_horizon.py     # RMSE at horizons h = 1, 3, 6, 12
│   └── shap_analysis.py     # SHAP feature importance
│
├── data/
│   └── diploma_dataset_1.csv  # Panel dataset (UA/LT/LV, monthly)
│
└── outputs/
    ├── models/              # Serialized models (.pkl) and best_params.json
    ├── predictions/         # CSV forecasts and SHAP values
    ├── tables/              # Per-phase result tables (CSV)
    ├── figures/             # Plots (PNG)
    └── reports/             # FINAL_REPORT.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -e .

# Install dev tools (ruff, pytest)
pip install -e ".[dev]"
```

## Run

```bash
python main.py
```

## Configuration

Key fields in `config.yaml`:

| Field | Default | Description |
|---|---|---|
| `data.dataset_path` | `data/diploma_dataset_1.csv` | Input dataset |
| `data.train_end` | `2019-12-31` | End of pre-test training window |
| `data.test_start` / `test_end` | `2020-01-01` / `2021-12-31` | Test window |
| `xgboost.n_estimators` | `2000` | Tree budget for final training |
| `xgboost.cv_n_splits` | `5` | Folds for TimeSeriesSplit CV |
| `project.random_seed` | `42` | Global RNG seed |
