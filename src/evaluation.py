from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["ARIMA_only", "XGB_pure", "Hybrid", "RW", "SN"]
MODEL_UA = {
    "ARIMA_only": "Чистий ARIMA",
    "XGB_pure": "Чистий XGBoost",
    "Hybrid": "Гібрид ARIMA+XGBoost",
    "RW": "Випадкове блукання",
    "SN": "Сезонний наїв",
}
COUNTRY_LABELS = {"UA": "Україна", "LT": "Литва", "LV": "Латвія"}
MODEL_COLORS = {
    "Чистий ARIMA": "#1f77b4",
    "Чистий XGBoost": "#ff7f0e",
    "Гібрид ARIMA+XGBoost": "#d62728",
    "Випадкове блукання": "#2ca02c",
    "Сезонний наїв": "#9467bd",
}


# point metrics


def rmse(actual, pred):
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


# Diebold-Mariano test


def dm_test(actual, pred1, pred2, h=1):
    mask = (~np.isnan(pred1)) & (~np.isnan(pred2))
    if mask.sum() < 4:
        return float("nan"), float("nan")

    e1 = actual[mask] - pred1[mask]
    e2 = actual[mask] - pred2[mask]
    d = e1**2 - e2**2

    T = len(d)
    d_bar = np.mean(d)

    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2.0 * gamma_k
    lrv = gamma0 + gamma_sum

    if lrv <= 0:
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(lrv / T)

    k = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if k <= 0:
        k = 1.0
    dm_star = dm_stat * np.sqrt(k)

    p_value = float(2 * stats.t.sf(np.abs(dm_star), df=T - 1))

    return float(dm_star), p_value


# full comparison table


def evaluate_all_models(hybrid_df, benchmark_df, cfg):
    all_cols = hybrid_df[["Country", "Date", "Actual", "Hybrid"]].merge(
        benchmark_df[["Country", "Date", "ARIMA_only", "XGB_pure", "RW", "SN"]],
        on=["Country", "Date"],
        how="inner",
    )

    comp_rows = []
    dm_rows = []

    for country in cfg.data.test_countries:
        sub = all_cols[all_cols["Country"] == country].sort_values("Date")
        actual = sub["Actual"].values
        label = country

        for model in MODELS:
            if model not in sub.columns:
                continue
            pred = sub[model].values
            comp_rows.append(
                {
                    "Країна": label,
                    "Модель": MODEL_UA.get(model, model),
                    "RMSE": round(rmse(actual, pred), 4),
                }
            )

        hybrid_pred = sub["Hybrid"].values
        dm_row = {"Країна": label}
        for benchmark in ["ARIMA_only", "XGB_pure"]:
            if benchmark not in sub.columns:
                continue
            bench_pred = sub[benchmark].values
            stat, pval = dm_test(actual, hybrid_pred, bench_pred, h=1)
            dm_row[f"DM vs {benchmark} (stat)"] = (
                round(stat, 4) if not np.isnan(stat) else None
            )
            dm_row[f"DM vs {benchmark} (p)"] = (
                round(pval, 4) if not np.isnan(pval) else None
            )
        dm_rows.append(dm_row)

    model_comparison = pd.DataFrame(comp_rows)
    dm_pvalues = pd.DataFrame(dm_rows)

    return model_comparison, dm_pvalues


# figures


def _save_fig(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_by_country_model(comparison_df, cfg, save_path):
    plt.style.use(cfg.figures.style)

    countries = comparison_df["Країна"].unique()
    models = comparison_df["Модель"].unique()
    n_countries = len(countries)
    n_models = len(models)
    x = np.arange(n_countries)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        vals = []
        for c in countries:
            row = comparison_df[
                (comparison_df["Країна"] == c) & (comparison_df["Модель"] == model)
            ]
            vals.append(float(row["RMSE"].values[0]) if len(row) else 0.0)

        color = MODEL_COLORS.get(model, "#7f7f7f")
        ax.bar(
            x + i * width,
            vals,
            width,
            label=model,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    country_labels_ua = [COUNTRY_LABELS.get(c, c) for c in countries]
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(country_labels_ua)
    ax.set_title("RMSE by country and model", fontsize=12)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Country")
    ax.legend(title="Model", loc="upper right")
    fig.tight_layout()
    _save_fig(fig, save_path, cfg.figures.dpi)


def plot_predictions_comparison(hybrid_df, benchmark_df, country, cfg, save_path):
    h_sub = hybrid_df[hybrid_df["Country"] == country].sort_values("Date")
    b_sub = benchmark_df[benchmark_df["Country"] == country].sort_values("Date")
    merged = h_sub.merge(b_sub[["Date", "XGB_pure"]], on="Date", how="inner")

    label = COUNTRY_LABELS.get(country, country)
    plt.style.use(cfg.figures.style)
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        merged["Date"], merged["Actual"], label="Actual", linewidth=2.2, color="black"
    )
    ax.plot(
        merged["Date"],
        merged["ARIMA_only"],
        label="ARIMA only",
        linewidth=1.4,
        linestyle="--",
        color="#1f77b4",
    )
    ax.plot(
        merged["Date"],
        merged["XGB_pure"],
        label="XGBoost pure",
        linewidth=1.2,
        linestyle="-.",
        color="#ff7f0e",
    )
    ax.plot(
        merged["Date"],
        merged["Hybrid"],
        label="Hybrid ARIMA+XGBoost",
        linewidth=1.8,
        linestyle="-.",
        color="#d62728",
    )

    ax.set_title(f"Forecast comparison — {label}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("CPI YoY (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, save_path, cfg.figures.dpi)


# run


def run(cfg):
    tables_dir = Path(cfg.outputs.tables_dir) / "evaluation"
    figures_dir = Path(cfg.outputs.figures_dir) / "evaluation"
    predictions_dir = Path(cfg.outputs.predictions_dir)

    for d in (tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    hybrid_df = pd.read_csv(
        predictions_dir / "hybrid_test_predictions.csv", parse_dates=["Date"]
    )
    benchmark_df = pd.read_csv(
        predictions_dir / "benchmark_predictions.csv", parse_dates=["Date"]
    )

    comparison_df, dm_df = evaluate_all_models(hybrid_df, benchmark_df, cfg)

    comparison_df.to_csv(tables_dir / "model_comparison.csv", index=False)
    dm_df.to_csv(tables_dir / "dm_test_pvalues.csv", index=False)

    plot_rmse_by_country_model(
        comparison_df, cfg, figures_dir / "rmse_by_country_model.png"
    )

    for country in cfg.data.test_countries:
        plot_predictions_comparison(
            hybrid_df,
            benchmark_df,
            country,
            cfg,
            figures_dir / f"predictions_comparison_{country.lower()}.png",
        )

    print("\n[Evaluation] Done.\n")
    print(comparison_df.to_string(index=False))
    print(f"\n  Tables  → {tables_dir}")
    print(f"  Figures → {figures_dir}\n")
