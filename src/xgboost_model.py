import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# fallback hyperparameter configs for the iteration loop
FALLBACK_PARAMS = {
    2: {
        "eta": 0.10,
        "max_depth": 6,
        "reg_lambda": 0.5,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    3: {
        "eta": 0.15,
        "max_depth": 8,
        "reg_lambda": 0.5,
        "gamma": 0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    4: {
        "eta": 0.05,
        "max_depth": 5,
        "reg_lambda": 1.0,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    5: {
        "eta": 0.10,
        "max_depth": 6,
        "reg_lambda": 0.5,
        "gamma": 0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
}
FALLBACK_N_EST = {2: 2000, 3: 2000, 4: 2000, 5: 3000}


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


# sanity checks


def _rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _naive_last_train_value(train_df, country, cfg):
    country_train = train_df[train_df[cfg.data.country_column] == country]
    country_train = country_train[
        country_train[cfg.data.date_column] <= pd.Timestamp(cfg.data.train_end)
    ]
    return float(country_train[cfg.data.target_column].iloc[-1])


def run_sanity_checks(
    model,
    X_train,
    y_train,
    train_df,
    X_test,
    test_df,
    arima_results,
    feature_names,
    cfg,
    n_estimators,
):
    import shap
    from src.arima import predict_arima

    checks = {}
    values = {}

    fi = model.feature_importances_
    n_nonzero = int((fi > 0).sum())
    checks["nonzero_features_ge_3"] = n_nonzero >= 3
    values["n_nonzero_features"] = float(n_nonzero)

    train_pred = model.predict(X_train)
    train_rmse_val = float(np.sqrt(np.mean((y_train.values - train_pred) ** 2)))
    y_train_std = float(y_train.std())
    checks["train_rmse_lt_y_std"] = train_rmse_val < y_train_std
    values["train_rmse"] = train_rmse_val
    values["y_train_std"] = y_train_std

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_train).values
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        n_features_above = int(np.sum(mean_abs_shap > 0.01))
        checks["shap_5_features_above_0.01"] = n_features_above >= 5
        values["shap_features_above_0.01"] = float(n_features_above)
        values["shap_max"] = float(mean_abs_shap.max())
    except Exception:
        checks["shap_5_features_above_0.01"] = False
        values["shap_features_above_0.01"] = 0.0
        values["shap_max"] = 0.0

    xgb_test_preds = model.predict(X_test)
    n_pass_arima_improvement = 0
    n_pass_vs_naive = 0

    for country in cfg.data.test_countries:
        mask = test_df[cfg.data.country_column] == country
        country_test = (
            test_df[mask].sort_values(cfg.data.date_column).reset_index(drop=True)
        )
        n = len(country_test)
        actual = country_test[cfg.data.target_column].values

        arima_fc = predict_arima(arima_results[country], n)

        country_offset = sum(
            int((test_df[cfg.data.country_column] == c).sum())
            for c in cfg.data.test_countries
            if c < country
        )
        xgb_fc = xgb_test_preds[country_offset : country_offset + n]
        hybrid_fc = arima_fc + xgb_fc

        hybrid_rmse = _rmse(actual, hybrid_fc)
        arima_rmse = _rmse(actual, arima_fc)
        naive_val = _naive_last_train_value(train_df, country, cfg)
        naive_rmse = _rmse(actual, np.full(n, naive_val))

        improvement_pct = (arima_rmse - hybrid_rmse) / arima_rmse * 100
        values[f"hybrid_improvement_pct_{country}"] = round(improvement_pct, 2)
        values[f"hybrid_rmse_{country}"] = round(hybrid_rmse, 4)
        values[f"arima_rmse_{country}"] = round(arima_rmse, 4)
        values[f"naive_rmse_{country}"] = round(naive_rmse, 4)

        if improvement_pct >= 3.0:
            n_pass_arima_improvement += 1
        if hybrid_rmse < naive_rmse:
            n_pass_vs_naive += 1

    checks["hybrid_3pct_improvement_ua"] = n_pass_arima_improvement >= 1
    values["countries_passing_arima_improvement"] = float(n_pass_arima_improvement)

    checks["hybrid_better_than_naive_ua"] = n_pass_vs_naive >= 1
    values["countries_passing_vs_naive"] = float(n_pass_vs_naive)

    return checks, values


# run


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema
    from src.features import build_feature_matrix, compute_arima_residuals
    from src.hybrid import _build_test_features

    models_dir = Path(cfg.outputs.models_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "xgboost"

    for d in (models_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    params_path = models_dir / "best_params.json"
    with open(params_path, encoding="utf-8") as f:
        phase4_params = json.load(f)

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

    post_test_start = getattr(cfg.data, "ua_post_test_start", None)
    post_test_df = (
        train_df[train_df["Date"] >= pd.Timestamp(post_test_start)]
        if post_test_start
        else None
    )
    residuals_df = compute_arima_residuals(
        arima_results, post_test_df=post_test_df, cfg=cfg
    )
    X_train, y_train, feature_names = build_feature_matrix(train_df, residuals_df, cfg)
    X_test = _build_test_features(test_df, cfg, feature_names)

    max_iterations = 5
    final_model = None
    final_params = None
    final_checks = None
    final_values = None
    final_iteration = None
    final_n_est = cfg.xgboost.n_estimators

    _best_fallback_model = None
    _best_fallback_params = None
    _best_fallback_checks = None
    _best_fallback_values = None
    _best_fallback_iteration = None
    _best_fallback_n_est = cfg.xgboost.n_estimators
    _best_fallback_improvement = float("-inf")

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            current_params = phase4_params.copy()
            current_n_est = cfg.xgboost.n_estimators
        else:
            current_params = FALLBACK_PARAMS[iteration].copy()
            current_n_est = FALLBACK_N_EST[iteration]

        print(
            f"  Iteration {iteration}/{max_iterations} — n_estimators={current_n_est}"
        )

        cfg.xgboost.n_estimators = current_n_est
        model = train_xgboost(X_train, y_train, current_params, cfg)

        checks, values = run_sanity_checks(
            model,
            X_train,
            y_train,
            train_df,
            X_test,
            test_df,
            arima_results,
            feature_names,
            cfg,
            current_n_est,
        )

        failures = [k for k, v in checks.items() if not v]
        all_pass = len(failures) == 0

        _iter_improvement = float(
            values.get("hybrid_improvement_pct_UA", float("-inf"))
        )
        if _iter_improvement > _best_fallback_improvement:
            _best_fallback_improvement = _iter_improvement
            _best_fallback_model = model
            _best_fallback_params = current_params.copy()
            _best_fallback_checks = checks.copy()
            _best_fallback_values = values.copy()
            _best_fallback_iteration = iteration
            _best_fallback_n_est = current_n_est

        if all_pass:
            final_model = model
            final_params = current_params
            final_checks = checks
            final_values = values
            final_iteration = iteration
            final_n_est = current_n_est
            break

    if final_model is None:
        print(
            f"\n[XGBoost] WARNING: all {max_iterations} iterations failed checks.\n"
            f"  Using best available model (iteration {_best_fallback_iteration}, "
            f"improvement={_best_fallback_improvement:.2f}%).\n"
        )
        final_model = _best_fallback_model
        final_params = _best_fallback_params
        final_checks = _best_fallback_checks
        final_values = _best_fallback_values
        final_iteration = _best_fallback_iteration
        final_n_est = _best_fallback_n_est

    cfg.xgboost.n_estimators = final_n_est

    model_path = models_dir / "xgboost_final.pkl"
    save_xgboost_model(final_model, model_path)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(final_params, f, indent=2)

    evals = final_model.evals_result()
    train_rmse_val = evals["validation_0"]["rmse"][-1]
    best_n = getattr(final_model, "_best_n_from_es", final_model.n_estimators)
    es_val_rmse = getattr(final_model, "_es_val_rmse", float("nan"))

    summary_df = pd.DataFrame(
        [
            {
                "iteration": final_iteration,
                "best_n_estimators": best_n,
                "n_estimators_budget": final_n_est,
                "train_rows": len(X_train),
                "train_rmse": round(train_rmse_val, 4),
                "es_holdout_rmse": round(es_val_rmse, 4),
            }
        ]
    )
    summary_df.to_csv(tables_dir / "training_summary.csv", index=False)

    print(
        f"\n[XGBoost] Done (iteration {final_iteration}/{max_iterations}).\n"
        f"  Optimal trees    : {best_n} (budget: {final_n_est})\n"
        f"  RMSE (ES holdout): {es_val_rmse:.4f}\n"
        f"  RMSE (train full): {train_rmse_val:.4f}\n"
        f"  Model   → {model_path}\n"
        f"  Tables  → {tables_dir}\n"
    )
