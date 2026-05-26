from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {
    "Hybrid": "Гібрид ARIMA+XGBoost",
    "ARIMA_only": "Чистий ARIMA",
    "XGB_pure": "Чистий XGBoost",
    "RW": "Випадкове блукання",
    "SN": "Сезонний наїв",
}
COUNTRY_LABELS = {"UA": "Україна", "LT": "Литва", "LV": "Латвія"}
MODEL_COLORS = {
    "Гібрид ARIMA+XGBoost": "#d62728",
    "Чистий ARIMA": "#1f77b4",
    "Чистий XGBoost": "#ff7f0e",
    "Випадкове блукання": "#2ca02c",
    "Сезонний наїв": "#9467bd",
}
MODEL_MARKERS = {
    "Гібрид ARIMA+XGBoost": "o",
    "Чистий ARIMA": "s",
    "Чистий XGBoost": "D",
    "Випадкове блукання": "^",
    "Сезонний наїв": "v",
}


def _rmse(actual, pred):
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


def evaluate_horizons(hybrid_df, benchmark_df, cfg):
    all_cols = hybrid_df[["Country", "Date", "Actual", "Hybrid", "ARIMA_only"]].merge(
        benchmark_df[["Country", "Date", "XGB_pure", "RW", "SN"]],
        on=["Country", "Date"],
        how="inner",
    )

    rows = []
    for country in cfg.data.test_countries:
        sub = (
            all_cols[all_cols["Country"] == country]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        actual = sub["Actual"].values
        n_test = len(sub)

        for h in cfg.horizons:
            if h > n_test:
                continue
            for model_key, model_label in MODELS.items():
                if model_key not in sub.columns:
                    continue
                pred_h = sub[model_key].values[:h]
                actual_h = actual[:h]
                score = _rmse(actual_h, pred_h)
                rows.append(
                    {
                        "Country": country,
                        "Horizon": h,
                        "Model": model_label,
                        "RMSE": round(score, 4),
                    }
                )

    result = (
        pd.DataFrame(rows)
        .sort_values(["Country", "Horizon", "Model"])
        .reset_index(drop=True)
    )
    return result


# figures


def _save_fig(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_horizon(horizon_df, cfg, save_path):
    countries = cfg.data.test_countries
    plt.style.use(cfg.figures.style)
    fig, axes = plt.subplots(
        1, len(countries), figsize=(5 * len(countries), 5), sharey=False
    )

    if len(countries) == 1:
        axes = [axes]

    for ax, country in zip(axes, countries):
        sub = horizon_df[horizon_df["Country"] == country]
        for model in sub["Model"].unique():
            model_sub = sub[sub["Model"] == model].sort_values("Horizon")
            ax.plot(
                model_sub["Horizon"],
                model_sub["RMSE"],
                label=model,
                color=MODEL_COLORS.get(model, "#7f7f7f"),
                marker=MODEL_MARKERS.get(model, "o"),
                linewidth=1.5,
                markersize=6,
            )

        ax.set_title(COUNTRY_LABELS.get(country, country), fontsize=11)
        ax.set_xlabel("Horizon (months)")
        ax.set_ylabel("RMSE")
        ax.set_xticks(cfg.horizons)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )
    fig.suptitle("RMSE by forecast horizon", fontsize=13, y=1.06)
    fig.tight_layout()
    _save_fig(fig, save_path, cfg.figures.dpi)


# run


def run(cfg):
    tables_dir = Path(cfg.outputs.tables_dir) / "horizon"
    figures_dir = Path(cfg.outputs.figures_dir) / "horizon"
    predictions_dir = Path(cfg.outputs.predictions_dir)

    for d in (tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    hybrid_df = pd.read_csv(
        predictions_dir / "hybrid_test_predictions.csv", parse_dates=["Date"]
    )
    benchmark_df = pd.read_csv(
        predictions_dir / "benchmark_predictions.csv", parse_dates=["Date"]
    )

    horizon_df = evaluate_horizons(hybrid_df, benchmark_df, cfg)
    horizon_df.to_csv(tables_dir / "horizon_metrics.csv", index=False)

    plot_rmse_vs_horizon(horizon_df, cfg, figures_dir / "rmse_vs_horizon.png")

    print("\n[Multi-horizon] Done.\n")
    pivot = horizon_df.pivot_table(
        index=["Country", "Model"], columns="Horizon", values="RMSE"
    )
    pivot.index = [(f"{COUNTRY_LABELS.get(c, c)}", m) for c, m in pivot.index]
    print(pivot.to_string())
    print(f"\n  Tables  → {tables_dir}")
    print(f"  Figures → {figures_dir}\n")
