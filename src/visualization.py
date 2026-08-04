"""
Visualization plots

Implement plotting of required experiment results.
"""

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


matplotlib.use("Agg")


class Visualizer:
    def __init__(
        self,
        cfg,
        figures_dir="outputs/figures",
        save_plots=True,
        show_plots=False,
        style="ggplot",
        dpi=300,
    ):
        self.cfg = cfg
        self.figures_dir = Path(figures_dir)
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.style = style
        self.dpi = dpi

    def _handle_fig(self, fig, filename: str):
        if self.save_plots:
            save_path = self.figures_dir / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if self.show_plots:
            plt.show(block=False)

        plt.close(fig)

    def plot_metric_vs_horizon(
        self,
        horizon_df,
        test_entities,
        entity_col,
        metric_name,
        filename="horizon/metric_vs_horizon.png",
    ):
        if not self.save_plots and not self.show_plots:
            return

        plt.style.use(self.style)
        fig, axes = plt.subplots(
            1, len(test_entities), figsize=(5 * len(test_entities), 5), sharey=False
        )

        if len(test_entities) == 1:
            axes = [axes]

        for ax, entity in zip(axes, test_entities):
            sub = horizon_df[horizon_df[entity_col] == entity]
            for model in sub["Model"].unique():
                model_sub = sub[sub["Model"] == model].sort_values("Horizon")

                ax.plot(
                    model_sub["Horizon"],
                    model_sub["Metric"],
                    label=model,
                    linewidth=1.5,
                    markersize=6,
                    marker="o",  # Provide a default marker to ensure points are visible
                )

            ax.set_title(str(entity), fontsize=11)
            ax.set_xlabel("Horizon (months)")
            ax.set_ylabel(metric_name.upper())
            ax.set_xticks(sorted(horizon_df["Horizon"].unique()))

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(labels),
            title="Model",
        )
        fig.tight_layout()
        self._handle_fig(fig, filename)

    def plot_metrics_by_entity_model(
        self,
        comparison_df,
        metric_name,
        filename="evaluation/metrics_by_entity_model.png",
    ):
        if not self.save_plots and not self.show_plots:
            return

        plt.style.use(self.style)
        countries = comparison_df["Country"].unique()
        models = comparison_df["Model"].unique()
        n_countries = len(countries)
        n_models = len(models)
        x = np.arange(n_countries)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(11, 6))

        for i, model in enumerate(models):
            vals = []
            for c in countries:
                row = comparison_df[
                    (comparison_df["Country"] == c) & (comparison_df["Model"] == model)
                ]
                vals.append(float(row["Metric"].values[0]) if len(row) else 0.0)

            ax.bar(
                x + i * width,
                vals,
                width,
                label=model,
                edgecolor="white",
                linewidth=0.5,
            )

        country_names = [str(c) for c in countries]
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(country_names)
        ax.set_title(f"{metric_name.upper()} by entity and model", fontsize=12)
        ax.set_ylabel(metric_name.upper())
        ax.set_xlabel("Entity")
        ax.legend(title="Model", loc="upper right")
        fig.tight_layout()
        self._handle_fig(fig, filename)

    def plot_predictions_comparison(
        self, predictions_df, test_entities, entity_col="Country"
    ):
        if not self.save_plots and not self.show_plots:
            return

        plt.style.use(self.style)
        for entity in test_entities:
            merged = predictions_df[predictions_df[entity_col] == entity].sort_values(
                "Date"
            )

            label = str(entity)
            fig, ax = plt.subplots(figsize=(13, 5))

            ax.plot(
                merged["Date"],
                merged["Actual"],
                label="Actual",
                linewidth=2.2,
                color="black",
            )

            # Find all model columns dynamically
            model_labels = [
                c
                for c in merged.columns
                if c not in [entity_col, "Date", "Actual", "Horizon"]
            ]

            # Cycle through line styles/widths to differentiate if needed, though color cycle helps
            line_styles = ["--", "-.", ":", "-"]
            line_widths = [1.4, 1.2, 1.8, 1.0, 1.6]

            for idx, m_label in enumerate(model_labels):
                ls = line_styles[idx % len(line_styles)]
                lw = line_widths[idx % len(line_widths)]
                ax.plot(
                    merged["Date"],
                    merged[m_label],
                    label=m_label,
                    linewidth=lw,
                    linestyle=ls,
                )

            ax.set_title(f"Forecast comparison — {label}", fontsize=12)
            ax.set_xlabel("Date")
            ax.set_ylabel("Target")
            ax.legend(loc="best")
            fig.tight_layout()
            self._handle_fig(
                fig, f"evaluation/predictions_comparison_{str(entity).lower()}.png"
            )

    def plot_shap_summary(
        self, ranking_df, top_n=15, filename="shap/shap_feature_importance.png"
    ):
        if not self.save_plots and not self.show_plots:
            return

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_df = ranking_df.head(top_n).sort_values("MeanAbsSHAP", ascending=True)

        # Color can remain fixed here since it's a single category metric
        ax.barh(plot_df["Feature"], plot_df["MeanAbsSHAP"], color="#1f77b4")
        ax.set_title(f"Top {top_n} Features by Mean |SHAP|")
        ax.set_xlabel("Mean |SHAP| value")

        fig.tight_layout()
        self._handle_fig(fig, filename)

    def plot_cpi_timeseries(
        self,
        df,
        entities,
        entity_col,
        target_col,
        date_col,
        test_start,
        filename="eda/cpi_timeseries.png",
    ):
        if not self.save_plots and not self.show_plots:
            return

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(14, 5))

        for entity in entities:
            subset = df[df[entity_col] == entity].sort_values(date_col)

            ax.plot(
                subset[date_col],
                subset[target_col],
                label=str(entity),
                linewidth=1.6,
            )

        ax.axvline(
            pd.Timestamp(test_start),
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"Test Start ({str(test_start)[:7]})",
        )
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Date")
        ax.set_ylabel("Target (YoY, %)")
        ax.set_title("Target Variable Time Series", fontsize=13)
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        self._handle_fig(fig, filename)
