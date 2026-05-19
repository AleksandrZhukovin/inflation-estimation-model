import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

COUNTRY_LABELS = {"UA": "Україна", "LT": "Литва", "LV": "Латвія"}


# pure-ARIMA


def arima_forecast(arima_results, test_df, cfg):
    from src.arima import predict_arima

    result = {}
    for country in cfg.data.test_countries:
        n_test = int((test_df[cfg.data.country_column] == country).sum())
        fc = predict_arima(arima_results[country], n_test)
        result[country] = fc
    return result


# pure-XGBoost benchmark


def _build_xgb_pure_train(train_df, cfg):
    drop_always = {
        cfg.data.date_column,
        cfg.data.target_column,
        *cfg.data.flag_columns,
    }

    rows = []
    for country in cfg.data.countries:
        subset = train_df[train_df[cfg.data.country_column] == country].copy()
        train_start = (
            cfg.data.ua_train_start if country == "UA" else cfg.data.lt_lv_train_start
        )
        subset = subset[subset[cfg.data.date_column] >= pd.Timestamp(train_start)]
        rows.append(subset)

    merged = pd.concat(rows, ignore_index=True).sort_values(
        [cfg.data.date_column, cfg.data.country_column]
    )

    merged = merged.dropna(subset=[cfg.data.target_column]).reset_index(drop=True)

    y = merged[cfg.data.target_column].reset_index(drop=True)

    feature_df = merged.drop(
        columns=[c for c in drop_always if c in merged.columns],
    )
    country_dummies = pd.get_dummies(
        feature_df[cfg.data.country_column], prefix="Country", dtype=float
    )
    feature_df = feature_df.drop(columns=[cfg.data.country_column])
    X = pd.concat([feature_df, country_dummies], axis=1).reset_index(drop=True)

    return X, y, list(X.columns)


def train_xgb_pure(train_df, best_params, cfg):
    X_train, y_train, feature_names = _build_xgb_pure_train(train_df, cfg)
    n = len(X_train)
    split_idx = int(n * (1.0 - cfg.xgboost.eval_split_ratio))

    X_fit, y_fit = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
    X_eval, y_eval = X_train.iloc[split_idx:], y_train.iloc[split_idx:]

    model = XGBRegressor(
        learning_rate=best_params["eta"],
        max_depth=int(best_params["max_depth"]),
        gamma=best_params["gamma"],
        reg_lambda=best_params["reg_lambda"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        n_estimators=cfg.xgboost.n_estimators,
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        random_state=cfg.project.random_seed,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_eval, y_eval)],
        verbose=False,
    )
    return model, feature_names


def xgb_pure_forecast(model, feature_names, test_df, cfg):
    from src.hybrid import _build_test_features

    X_test = _build_test_features(test_df, cfg, feature_names)
    all_preds = model.predict(X_test)

    result = {}
    offset = 0
    for country in cfg.data.test_countries:
        n = int((test_df[cfg.data.country_column] == country).sum())
        result[country] = all_preds[offset : offset + n]
        offset += n
    return result


def save_xgb_pure(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_xgb_pure(path):
    return joblib.load(path)


# assemble benchmark DataFrame


def build_benchmark_df(test_df, arima_fc, xgb_pure_fc, cfg):
    rows = []
    for country in cfg.data.test_countries:
        country_test = (
            test_df[test_df[cfg.data.country_column] == country]
            .sort_values(cfg.data.date_column)
            .reset_index(drop=True)
        )
        for i, row in country_test.iterrows():
            rows.append(
                {
                    "Country": country,
                    "Date": row[cfg.data.date_column],
                    "Actual": float(row[cfg.data.target_column]),
                    "ARIMA_only": float(arima_fc[country][i]),
                    "XGB_pure": float(xgb_pure_fc[country][i]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Country", "Date"]).reset_index(drop=True)


def _rmse(actual, pred):
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


# run


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema

    models_dir = Path(cfg.outputs.models_dir)
    predictions_dir = Path(cfg.outputs.predictions_dir)

    for d in (models_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)

    params_path = models_dir / "best_params.json"
    with open(params_path, encoding="utf-8") as f:
        best_params = json.load(f)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, test_df = split_train_test(
        df,
        cfg.data.train_end,
        cfg.data.test_start,
        test_countries=cfg.data.test_countries,
        test_end=cfg.data.test_end,
        post_test_start=getattr(cfg.data, "ua_post_test_start", None),
    )

    arima_results = {
        country: load_arima_model(models_dir / f"arima_{country.lower()}.pkl")
        for country in cfg.data.countries
    }

    arima_fc = arima_forecast(arima_results, test_df, cfg)

    xgb_pure_model, feature_names = train_xgb_pure(train_df, best_params, cfg)
    save_xgb_pure(xgb_pure_model, models_dir / "xgboost_pure.pkl")
    xgb_pure_fc = xgb_pure_forecast(xgb_pure_model, feature_names, test_df, cfg)

    benchmark_df = build_benchmark_df(test_df, arima_fc, xgb_pure_fc, cfg)
    pred_path = predictions_dir / "benchmark_predictions.csv"
    benchmark_df.to_csv(pred_path, index=False)

    print("\n[Benchmarks] Done.\n")
    for country in cfg.data.test_countries:
        sub = benchmark_df[benchmark_df["Country"] == country]
        actual = sub["Actual"].values
        print(
            f"  {COUNTRY_LABELS.get(country, country):10s} | "
            f"ARIMA={_rmse(actual, sub['ARIMA_only'].values):.4f}  "
            f"XGB_pure={_rmse(actual, sub['XGB_pure'].values):.4f}"
        )
    print(f"\n  Models  → {models_dir / 'xgboost_pure.pkl'}")
    print(f"  Predictions → {pred_path}\n")
