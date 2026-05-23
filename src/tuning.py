import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

GRID_PARAMS = [
    "eta",
    "max_depth",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "subsample",
    "colsample_bytree",
]

N_TRIALS = 150


def _make_combinations(cfg):
    grid = cfg.xgb_grid
    values = [
        grid.eta,
        grid.max_depth,
        grid.min_child_weight,
        grid.gamma,
        grid.reg_alpha,
        grid.reg_lambda,
        grid.subsample,
        grid.colsample_bytree,
    ]
    return [dict(zip(GRID_PARAMS, combo)) for combo in product(*values)]


def _fit_evaluate(
    params, X, y, tscv, cv_n_estimators, cv_early_stopping_rounds, random_state
):
    fold_rmses = []

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
        fold_rmses.append(float(np.sqrt(np.mean((actual - pred) ** 2))))

    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))


def tune_xgboost(X, y, cfg):
    combinations = _make_combinations(cfg)
    n_splits = cfg.xgboost.cv_n_splits
    cv_n_est = cfg.xgboost.cv_n_estimators
    cv_es = cfg.xgboost.cv_early_stopping_rounds

    tscv = TimeSeriesSplit(n_splits=n_splits)

    records = []
    best_rmse = float("inf")
    best_params = {}

    for params in combinations:
        mean_rmse, std_rmse = _fit_evaluate(
            params, X, y, tscv, cv_n_est, cv_es, cfg.project.random_seed
        )

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params.copy()

        records.append(
            {**params, "mean_rmse": round(mean_rmse, 4), "std_rmse": round(std_rmse, 4)}
        )

    results_df = pd.DataFrame(records)
    return best_params, results_df


def tune_xgboost_optuna(X, y, cfg, n_trials=150):
    import optuna

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
        mean_rmse, _ = _fit_evaluate(params, X, y, tscv, cv_n_est, cv_es, seed)
        return mean_rmse

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
                {**t.params, "mean_rmse": round(t.value, 4), "std_rmse": float("nan")}
            )
    results_df = pd.DataFrame(records).sort_values("mean_rmse").reset_index(drop=True)

    return best_params, results_df


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema
    from src.features import build_feature_matrix, compute_arima_residuals

    models_dir = Path(cfg.outputs.models_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "tuning"

    for d in (models_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, _ = split_train_test(
        df,
        cfg.data.train_end,
        cfg.data.test_start,
    )

    arima_results = {
        country: load_arima_model(models_dir / f"arima_{country.lower()}.pkl")
        for country in cfg.data.countries
    }
    residuals_df = compute_arima_residuals(arima_results)
    X, y, feature_names = build_feature_matrix(train_df, residuals_df, cfg)

    best_params, results_df = tune_xgboost_optuna(X, y, cfg, n_trials=N_TRIALS)
    best_rmse = float(results_df["mean_rmse"].min())

    params_path = models_dir / "best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    results_df.to_csv(tables_dir / "hyperparameter_search_full.csv", index=False)
    results_df.nsmallest(10, "mean_rmse").reset_index(drop=True).to_csv(
        tables_dir / "hyperparameter_search_top10.csv", index=False
    )

    print(
        f"\n[Tuning] Done.\n"
        f"  Trials      : {len(results_df)}\n"
        f"  Best RMSE   : {best_rmse:.4f}\n"
        f"  Best params : {best_params}\n"
        f"  Models  → {params_path}\n"
        f"  Tables  → {tables_dir}\n"
    )
