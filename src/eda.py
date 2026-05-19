from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data import get_country_subset, load_dataset, split_train_test, validate_schema

COUNTRY_LABELS = {"UA": "Україна", "LT": "Литва", "LV": "Латвія"}
COUNTRY_COLORS = {"UA": "#1f77b4", "LT": "#ff7f0e", "LV": "#2ca02c"}


# helpers


def _save_fig(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _country_train_slice(train_df, country, cfg):
    subset = get_country_subset(train_df, country)
    start = cfg.data.ua_train_start if country == "UA" else cfg.data.lt_lv_train_start
    return subset[subset[cfg.data.date_column] >= pd.Timestamp(start)].reset_index(
        drop=True
    )


# tables


def compute_descriptive_stats(train_df, cfg):
    frames = []
    for country in cfg.data.countries:
        subset = _country_train_slice(train_df, country, cfg)
        drop_cols = [
            cfg.data.date_column,
            cfg.data.country_column,
            *cfg.data.flag_columns,
        ]
        numeric = subset.drop(columns=[c for c in drop_cols if c in subset.columns])
        desc = numeric.describe().T.round(4)
        desc.insert(0, "Country", country)
        desc.index.name = "Feature"
        frames.append(desc)
    return pd.concat(frames)


# figures


def plot_cpi_timeseries(df, cfg, save_path):
    plt.style.use(cfg.figures.style)
    fig, ax = plt.subplots(figsize=(14, 5))

    for country in cfg.data.countries:
        subset = get_country_subset(df, country).sort_values(cfg.data.date_column)
        ax.plot(
            subset[cfg.data.date_column],
            subset[cfg.data.target_column],
            label=COUNTRY_LABELS[country],
            color=COUNTRY_COLORS[country],
            linewidth=1.6,
        )

    ax.axvline(
        pd.Timestamp(cfg.data.test_start),
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Початок тестового набору ({cfg.data.test_start[:7]})",
    )
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Дата")
    ax.set_ylabel("ІСЦ (р/р, %)")
    ax.set_title("Індекс споживчих цін — зміна рік до року (%)", fontsize=13)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, save_path, cfg.figures.dpi)


# run


def run(cfg):
    tables_dir = Path(cfg.outputs.tables_dir) / "eda"
    figures_dir = Path(cfg.outputs.figures_dir) / "eda"

    for d in (tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data.dataset_path)
    validate_schema(df)
    train_df, test_df = split_train_test(df, cfg.data.train_end, cfg.data.test_start)

    desc_df = compute_descriptive_stats(train_df, cfg)
    desc_df.to_csv(tables_dir / "descriptive_stats.csv")

    plot_cpi_timeseries(df, cfg, figures_dir / "cpi_timeseries.png")

    print(
        f"\n[EDA] Done.\n"
        f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}\n"
        f"  Tables  → {tables_dir}\n"
        f"  Figures → {figures_dir}\n"
    )
