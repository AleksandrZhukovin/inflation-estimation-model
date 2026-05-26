from pathlib import Path

import numpy as np
import pandas as pd


def compute_arima_residuals(arima_results):
    frames = []
    for country, result in arima_results.items():
        n = min(len(result.train_dates), len(result.residuals))
        frames.append(
            pd.DataFrame(
                {
                    "Date": result.train_dates[-n:],
                    "Country": country,
                    "arima_residual": result.residuals[-n:],
                }
            )
            .dropna(subset=["arima_residual"])
            .reset_index(drop=True)
        )
    return pd.concat(frames, ignore_index=True).sort_values(["Country", "Date"])


def build_feature_matrix(train_df, residuals_df, cfg):
    drop_always = {
        cfg.data.date_column,
        cfg.data.target_column,
        *cfg.data.flag_columns,
    }

    merged = train_df.merge(
        residuals_df[["Date", "Country", "arima_residual"]],
        on=["Date", "Country"],
        how="inner",
    )

    assert len(merged) > 0, (
        "Feature matrix merge returned 0 rows — check that train_df dates match residuals dates"
    )

    rows = []
    for country in cfg.data.countries:
        subset = merged[merged[cfg.data.country_column] == country]
        train_start = (
            cfg.data.ua_train_start if country == "UA" else cfg.data.lt_lv_train_start
        )
        subset = subset[subset[cfg.data.date_column] >= pd.Timestamp(train_start)]
        rows.append(subset)
    merged = pd.concat(rows, ignore_index=True).sort_values(
        [cfg.data.date_column, cfg.data.country_column]
    )

    macro_cols = [
        c
        for c in merged.columns
        if c not in drop_always and c not in (cfg.data.country_column, "arima_residual")
    ]
    merged = merged.copy()
    merged[macro_cols] = merged.groupby(cfg.data.country_column)[macro_cols].shift(1)
    merged = merged.dropna(subset=macro_cols).reset_index(drop=True)

    _dates = pd.to_datetime(merged[cfg.data.date_column])
    merged["Month_sin"] = np.sin(2 * np.pi * _dates.dt.month / 12)
    merged["Month_cos"] = np.cos(2 * np.pi * _dates.dt.month / 12)
    merged["Year_trend"] = (_dates.dt.year - 2004) / 20.0

    y = merged["arima_residual"].reset_index(drop=True)

    feature_df = merged.drop(
        columns=[c for c in drop_always if c in merged.columns] + ["arima_residual"],
    )

    country_dummies = pd.get_dummies(
        feature_df[cfg.data.country_column], prefix="Country", dtype=float
    )
    feature_df = feature_df.drop(columns=[cfg.data.country_column])
    X = pd.concat([feature_df, country_dummies], axis=1).reset_index(drop=True)

    feature_names = list(X.columns)

    return X, y, feature_names


def build_feature_summary(X, feature_names):
    rows = []
    for col in feature_names:
        s = X[col]
        rows.append(
            {
                "Feature": col,
                "dtype": str(s.dtype),
                "pct_missing": round(s.isna().mean() * 100, 2),
                "min": round(s.min(), 4) if s.notna().any() else None,
                "max": round(s.max(), 4) if s.notna().any() else None,
                "mean": round(s.mean(), 4) if s.notna().any() else None,
                "std": round(s.std(), 4) if s.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema

    predictions_dir = Path(cfg.outputs.predictions_dir)
    tables_dir = Path(cfg.outputs.tables_dir) / "features"

    for d in (predictions_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, _ = split_train_test(
        df,
        cfg.data.train_end,
        cfg.data.test_start,
    )

    models_dir = Path(cfg.outputs.models_dir)
    arima_results = {
        country: load_arima_model(models_dir / f"arima_{country.lower()}.pkl")
        for country in cfg.data.countries
    }

    residuals_df = compute_arima_residuals(arima_results)
    residuals_df.to_csv(predictions_dir / "arima_residuals_train.csv", index=False)

    X, y, feature_names = build_feature_matrix(train_df, residuals_df, cfg)

    summary_df = build_feature_summary(X, feature_names)
    summary_df.to_csv(tables_dir / "feature_summary.csv", index=False)

    resid_by_country = (
        residuals_df.groupby("Country")["arima_residual"].count().to_dict()
    )
    nan_total = int(X.isnull().sum().sum())

    print(
        f"\n[Features] Done.\n"
        f"  Feature matrix  : {len(X)} rows × {len(feature_names)} features\n"
        f"  ARIMA residuals : {resid_by_country}\n"
        f"  NaN in matrix   : {nan_total}\n"
        f"  Tables  → {tables_dir}\n"
    )
