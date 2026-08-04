"""
EDA analyzer

Compute descriptive statistics and plot CPI time series for each entity.
"""

from pathlib import Path
import pandas as pd


class EDAAnalyzer:
    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.visualizer = visualizer

        self.tables_dir = Path(self.cfg.outputs.tables_dir) / "eda"

        self.entity_col = getattr(
            self.cfg.data,
            "entity_column",
            getattr(self.cfg.data, "country_column", "Country"),
        )
        self.entities = getattr(
            self.cfg.data, "entities", getattr(self.cfg.data, "countries", [])
        )

    def _country_train_slice(self, train_df, entity):
        subset = train_df[train_df[self.entity_col] == entity]

        # Backward compatibility for start dates
        start_dates = getattr(self.cfg.data, "entity_start_dates", None)
        if start_dates is None and hasattr(self.cfg.data, "ua_train_start"):
            start_dates = {
                "UA": self.cfg.data.ua_train_start,
                "LT": self.cfg.data.lt_lv_train_start,
                "LV": getattr(self.cfg.data, "lt_lv_train_start", None),
            }

        start = start_dates.get(entity) if start_dates else None

        if start:
            subset = subset[subset[self.cfg.data.date_column] >= pd.Timestamp(start)]

        return subset.reset_index(drop=True)

    def compute_descriptive_stats(self, train_df, save=True):
        frames = []
        for entity in self.entities:
            subset = self._country_train_slice(train_df, entity)

            drop_cols = [
                self.cfg.data.date_column,
                self.entity_col,
                *getattr(self.cfg.data, "flag_columns", []),
            ]
            numeric = subset.drop(columns=[c for c in drop_cols if c in subset.columns])
            desc = numeric.describe().T.round(4)
            desc.insert(0, "Entity", str(entity))
            desc.index.name = "Feature"
            frames.append(desc)

        desc_df = pd.concat(frames)

        if save:
            self.tables_dir.mkdir(parents=True, exist_ok=True)
            desc_df.to_csv(self.tables_dir / "descriptive_stats.csv")

        return desc_df

    def run_all(self, full_df, train_df):
        self.compute_descriptive_stats(train_df)
        if self.visualizer:
            self.visualizer.plot_cpi_timeseries(
                df=full_df,
                entities=self.entities,
                entity_col=self.entity_col,
                target_col=self.cfg.data.target_column,
                date_col=self.cfg.data.date_column,
                test_start=self.cfg.data.test_start,
            )
