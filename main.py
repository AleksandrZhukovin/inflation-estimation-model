import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.config import load_config
from src.eda import run as run_eda
from src.multi_horizon import run as run_multi_horizon
from src.shap_analysis import run as run_shap


def _wf_generate_windows(cfg, df):
    ua_dates = df[df["Country"] == "UA"]["Date"]
    ua_first_year = pd.Timestamp(cfg.data.ua_train_start).year
    ua_last_year = ua_dates.max().year
    init = cfg.walk_forward.initial_train_years
    step = cfg.walk_forward.test_window_years
    windows = []
    test_year = ua_first_year + init
    while test_year <= ua_last_year:
        windows.append((test_year - step, test_year))
        test_year += step
    return windows



def _wf_run_window(train_end_year, test_year, df, cfg):
    from src.arima import fit_arima_for_country
    from src.benchmarks import (
        arima_forecast,
        build_benchmark_df,
        random_walk_forecast,
        seasonal_naive_forecast,
        train_xgb_pure,
        xgb_pure_forecast,
    )
    from src.data import split_train_test
    from src.evaluation import rmse
    from src.features import build_feature_matrix, compute_arima_residuals
    from src.hybrid import predict_hybrid
    from src.tuning import N_TRIALS, tune_xgboost_optuna
    from src.xgboost_model import train_xgboost

    train_df, test_df = split_train_test(
        df,
        f"{train_end_year}-12-31",
        f"{test_year}-01-01",
        test_countries=cfg.data.test_countries,
        test_end=f"{test_year}-12-31",
    )
    if len(test_df) == 0:
        return None

    arima_results = {
        c: fit_arima_for_country(c, train_df, cfg) for c in cfg.data.countries
    }
    residuals_df = compute_arima_residuals(arima_results)
    X_train, y_train, feature_names = build_feature_matrix(train_df, residuals_df, cfg)

    params, _ = tune_xgboost_optuna(X_train, y_train, cfg, n_trials=N_TRIALS)
    model = train_xgboost(X_train, y_train, params, cfg)

    hybrid_df = predict_hybrid(
        test_df, arima_results, model, feature_names, cfg, train_df=train_df
    )
    arima_fc = arima_forecast(arima_results, test_df, cfg)
    xgb_model_p, feat_p = train_xgb_pure(train_df, params, cfg)
    xgb_fc = xgb_pure_forecast(xgb_model_p, feat_p, test_df, cfg, train_df=train_df)
    rw_fc = random_walk_forecast(train_df, test_df, cfg)
    sn_fc = seasonal_naive_forecast(train_df, test_df, cfg)
    bench_df = build_benchmark_df(test_df, arima_fc, xgb_fc, rw_fc, sn_fc, cfg)

    country = cfg.data.test_countries[0]
    actual = test_df[test_df[cfg.data.country_column] == country][
        cfg.data.target_column
    ].values
    hybrid_pred = hybrid_df[hybrid_df["Country"] == country]["Hybrid"].values

    metrics = {
        "train_end_year": train_end_year,
        "test_year": test_year,
        "n_test": len(actual),
        "rmse_arima": round(rmse(actual, arima_fc[country]), 4),
        "rmse_hybrid": round(rmse(actual, hybrid_pred), 4),
        "rmse_xgb_pure": round(rmse(actual, xgb_fc[country]), 4),
        "rmse_rw": round(rmse(actual, rw_fc[country]), 4),
        "rmse_sn": round(rmse(actual, sn_fc[country]), 4),
    }
    return metrics, arima_results, model, xgb_model_p, params, hybrid_df, bench_df


def run_walk_forward(cfg):
    from src.arima import save_arima_model
    from src.benchmarks import save_xgb_pure
    from src.data import load_dataset, validate_schema
    from src.xgboost_model import save_xgboost_model

    models_dir = Path(cfg.outputs.models_dir)
    predictions_dir = Path(cfg.outputs.predictions_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "walk_forward"
    for d in (models_dir, predictions_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)

    windows = _wf_generate_windows(cfg, df)
    ua_start = pd.Timestamp(cfg.data.ua_train_start).year
    records = []
    last = None

    for i, (train_end_year, test_year) in enumerate(windows):
        print(
            f"  Window {i + 1}/{len(windows)}: {ua_start}–{train_end_year} → test {test_year} ..."
        )
        result = _wf_run_window(train_end_year, test_year, df, cfg)
        if result is None:
            continue
        metrics, arima_results, model, xgb_pure, params, hybrid_df, bench_df = result
        records.append(metrics)
        last = (
            train_end_year,
            test_year,
            arima_results,
            model,
            xgb_pure,
            params,
            hybrid_df,
            bench_df,
        )

    if last is None:
        raise RuntimeError("Walk-forward: no windows produced valid results")

    (
        train_end_year,
        test_year,
        arima_results,
        model,
        xgb_pure,
        params,
        hybrid_df,
        bench_df,
    ) = last

    for country, res in arima_results.items():
        save_arima_model(res, models_dir / f"arima_{country.lower()}.pkl")
    save_xgboost_model(model, models_dir / "xgboost_final.pkl")
    save_xgb_pure(xgb_pure, models_dir / "xgboost_pure.pkl")
    with open(models_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    hybrid_df.to_csv(predictions_dir / "hybrid_test_predictions.csv", index=False)
    bench_df.to_csv(predictions_dir / "benchmark_predictions.csv", index=False)

    cfg.data.train_end = f"{train_end_year}-12-31"
    cfg.data.test_start = f"{test_year}-01-01"
    cfg.data.test_end = f"{test_year}-12-31"

    per_window_df = pd.DataFrame(records)
    per_window_df.to_csv(tables_dir / "per_window_metrics.csv", index=False)

    summary_rows = []
    for col in ["rmse_arima", "rmse_hybrid", "rmse_xgb_pure", "rmse_rw", "rmse_sn"]:
        summary_rows.append(
            {
                "model": col,
                "mean_rmse": round(float(np.mean(per_window_df[col])), 4),
                "std_rmse": round(float(np.std(per_window_df[col])), 4),
            }
        )
    pd.DataFrame(summary_rows).to_csv(tables_dir / "summary_metrics.csv", index=False)

    header = f"{'Win':>4}  {'Train':>10}  {'Test':>4}  {'ARIMA':>8}  {'Hybrid':>8}  {'XGB_pure':>8}  {'RW':>8}  {'SN':>8}"
    print(f"\n[Walk-forward CV] {len(records)} windows\n")
    print(header)
    print("─" * len(header))
    for j, row in per_window_df.iterrows():
        print(
            f"{j + 1:>4}  {ua_start}–{int(row.train_end_year):>4}  "
            f"{int(row.test_year):>4}  "
            f"{row.rmse_arima:>8.4f}  {row.rmse_hybrid:>8.4f}  {row.rmse_xgb_pure:>8.4f}  "
            f"{row.rmse_rw:>8.4f}  {row.rmse_sn:>8.4f}"
        )
    print("─" * len(header))
    for row in summary_rows:
        tag = row["model"].replace("rmse_", "").upper()
        print(
            f"  {'Mean ' + tag:>16}        {row['mean_rmse']:>8.4f}  ±{row['std_rmse']:.4f}"
        )
    print(f"\n  Tables → {tables_dir}\n  Models → {models_dir}\n")


PHASES = [
    ("EDA", run_eda),
    ("Walk-forward CV", run_walk_forward),
    ("Multi-horizon", run_multi_horizon),
    ("SHAP analysis", run_shap),
]


def main():
    cfg = load_config()
    random.seed(cfg.project.random_seed)
    np.random.seed(cfg.project.random_seed)

    t_total = time.perf_counter()
    for name, fn in PHASES:
        t0 = time.perf_counter()
        fn(cfg)
        print(f"[OK] {name} — {time.perf_counter() - t0:.1f}s")

    print(f"\nDone — {time.perf_counter() - t_total:.1f}s total")


if __name__ == "__main__":
    main()
