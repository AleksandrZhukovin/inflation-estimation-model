import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, params, cfg):
    n = len(X_train)
    holdout_size = min(72, max(int(n * 0.12), 36))

    X_fit = X_train.iloc[:-holdout_size]
    y_fit = y_train.iloc[:-holdout_size]
    X_val = X_train.iloc[-holdout_size:]
    y_val = y_train.iloc[-holdout_size:]

    model_es = XGBRegressor(
        learning_rate=params["eta"],
        max_depth=int(params["max_depth"]),
        min_child_weight=int(params.get("min_child_weight", 1)),
        gamma=params["gamma"],
        reg_alpha=params.get("reg_alpha", 0),
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        n_estimators=cfg.xgboost.n_estimators,
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        random_state=cfg.project.random_seed,
        tree_method="hist",
        verbosity=0,
    )
    model_es.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    raw_best_n = model_es.best_iteration + 1
    es_val_rmse = model_es.evals_result()["validation_0"]["rmse"][
        model_es.best_iteration
    ]

    if raw_best_n <= 10:
        best_n = cfg.xgboost.n_estimators_fixed
    else:
        best_n = raw_best_n

    model_final = XGBRegressor(
        learning_rate=params["eta"],
        max_depth=int(params["max_depth"]),
        min_child_weight=int(params.get("min_child_weight", 1)),
        gamma=params["gamma"],
        reg_alpha=params.get("reg_alpha", 0),
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        n_estimators=best_n,
        random_state=cfg.project.random_seed,
        tree_method="hist",
        verbosity=0,
    )
    model_final.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    model_final._best_n_from_es = best_n
    model_final._es_val_rmse = es_val_rmse

    return model_final


def predict_xgboost(model, X):
    return model.predict(X)


def save_xgboost_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_xgboost_model(path):
    path = Path(path)
    return joblib.load(path)


# run


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema
    from src.features import build_feature_matrix, compute_arima_residuals

    models_dir = Path(cfg.outputs.models_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "xgboost"

    for d in (models_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    params_path = models_dir / "best_params.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

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
    X_train, y_train, _ = build_feature_matrix(train_df, residuals_df, cfg)

    model = train_xgboost(X_train, y_train, params, cfg)

    model_path = models_dir / "xgboost_final.pkl"
    save_xgboost_model(model, model_path)

    evals = model.evals_result()
    train_rmse_val = evals["validation_0"]["rmse"][-1]
    best_n = getattr(model, "_best_n_from_es", model.n_estimators)
    es_val_rmse = getattr(model, "_es_val_rmse", float("nan"))

    summary_df = pd.DataFrame(
        [
            {
                "best_n_estimators": best_n,
                "n_estimators_budget": cfg.xgboost.n_estimators,
                "train_rows": len(X_train),
                "train_rmse": round(train_rmse_val, 4),
                "es_holdout_rmse": round(es_val_rmse, 4),
            }
        ]
    )
    summary_df.to_csv(tables_dir / "training_summary.csv", index=False)

    print(
        f"\n[XGBoost] Done.\n"
        f"  Optimal trees    : {best_n} (budget: {cfg.xgboost.n_estimators})\n"
        f"  RMSE (ES holdout): {es_val_rmse:.4f}\n"
        f"  RMSE (train full): {train_rmse_val:.4f}\n"
        f"  Model   → {model_path}\n"
        f"  Tables  → {tables_dir}\n"
    )
