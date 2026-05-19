from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COUNTRY_LABELS = {"UA": "Україна", "LT": "Литва", "LV": "Латвія"}


def _build_test_features(test_df, cfg, feature_names):
    drop_always = {
        cfg.data.date_column,
        cfg.data.target_column,
        *cfg.data.flag_columns,
    }

    working = test_df.copy()

    _dates = pd.to_datetime(working[cfg.data.date_column])
    working["Month_sin"] = np.sin(2 * np.pi * _dates.dt.month / 12)
    working["Month_cos"] = np.cos(2 * np.pi * _dates.dt.month / 12)
    working["Year_trend"] = (_dates.dt.year - 2004) / 20.0

    country_dummies = pd.get_dummies(
        working[cfg.data.country_column], prefix="Country", dtype=float
    )
    working = working.drop(columns=[cfg.data.country_column])
    working = pd.concat([working, country_dummies], axis=1)
    working = working.drop(columns=[c for c in drop_always if c in working.columns])

    for col in feature_names:
        if col not in working.columns:
            working[col] = 0.0

    return working[feature_names].reset_index(drop=True)


def predict_hybrid(test_df, arima_results, xgb_model, feature_names, cfg):
    from src.arima import predict_arima

    X_test = _build_test_features(test_df, cfg, feature_names)
    xgb_preds_all = xgb_model.predict(X_test)

    rows = []
    xgb_offset = 0

    for country in cfg.data.test_countries:
        country_mask = test_df[cfg.data.country_column] == country
        country_test = (
            test_df[country_mask]
            .sort_values(cfg.data.date_column)
            .reset_index(drop=True)
        )
        n_periods = len(country_test)

        arima_fc = predict_arima(arima_results[country], n_periods)

        xgb_fc = xgb_preds_all[xgb_offset : xgb_offset + n_periods]
        xgb_offset += n_periods

        hybrid_fc = arima_fc + xgb_fc

        for i, row in country_test.iterrows():
            rows.append(
                {
                    "Country": country,
                    "Date": row[cfg.data.date_column],
                    "Actual": float(row[cfg.data.target_column]),
                    "ARIMA_only": float(arima_fc[i]),
                    "XGB_residual": float(xgb_fc[i]),
                    "Hybrid": float(hybrid_fc[i]),
                }
            )

    result_df = (
        pd.DataFrame(rows).sort_values(["Country", "Date"]).reset_index(drop=True)
    )
    return result_df


# figures


def _save_fig(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def plot_predictions(predictions_df, country, cfg, save_path):
    subset = predictions_df[predictions_df["Country"] == country].sort_values("Date")
    label = COUNTRY_LABELS.get(country, country)

    plt.style.use(cfg.figures.style)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        subset["Date"], subset["Actual"], label="Actual", linewidth=2, color="black"
    )
    ax.plot(
        subset["Date"],
        subset["ARIMA_only"],
        label="ARIMA only",
        linewidth=1.5,
        linestyle="--",
        color="#1f77b4",
    )
    ax.plot(
        subset["Date"],
        subset["Hybrid"],
        label="Hybrid ARIMA+XGBoost",
        linewidth=1.5,
        linestyle="-.",
        color="#d62728",
    )

    ax.set_title(f"Forecast vs Actual — {label}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("CPI YoY (%)")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save_path, cfg.figures.dpi)


# run


def run(cfg):
    from src.arima import load_arima_model
    from src.data import load_dataset, split_train_test, validate_schema
    from src.features import build_feature_matrix, compute_arima_residuals
    from src.xgboost_model import load_xgboost_model

    models_dir = Path(cfg.outputs.models_dir)
    predictions_dir = Path(cfg.outputs.predictions_dir)
    figures_dir = Path(cfg.outputs.figures_dir) / "hybrid"

    for d in (models_dir, predictions_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, test_df = split_train_test(
        df,
        cfg.data.train_end,
        cfg.data.test_start,
        test_countries=cfg.data.test_countries,
        test_end=cfg.data.test_end,
    )
    assert train_df["Date"].max() < test_df["Date"].min(), "Train/test leakage detected"

    arima_results = {
        country: load_arima_model(models_dir / f"arima_{country.lower()}.pkl")
        for country in cfg.data.countries
    }
    xgb_model = load_xgboost_model(models_dir / "xgboost_final.pkl")

    residuals_df = compute_arima_residuals(arima_results)
    _, _, feature_names = build_feature_matrix(train_df, residuals_df, cfg)

    predictions_df = predict_hybrid(
        test_df, arima_results, xgb_model, feature_names, cfg
    )

    pred_path = predictions_dir / "hybrid_test_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    for country in cfg.data.test_countries:
        plot_predictions(
            predictions_df,
            country,
            cfg,
            figures_dir / f"predictions_vs_actual_{country.lower()}.png",
        )

    for country in cfg.data.test_countries:
        sub = predictions_df[predictions_df["Country"] == country]
        actual = sub["Actual"].values
        print(
            f"  {COUNTRY_LABELS.get(country, country):10s} | "
            f"RMSE ARIMA={_rmse(actual, sub['ARIMA_only'].values):.4f}  "
            f"RMSE Hybrid={_rmse(actual, sub['Hybrid'].values):.4f}"
        )

    print(
        f"\n[Hybrid] Done.\n"
        f"  Predictions → {pred_path}\n"
        f"  Figures     → {figures_dir}\n"
    )
