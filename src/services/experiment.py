"""
Experiment orchestration service.

Stateless runner that accepts a typed config, executes the full
model pipeline (single window or walk-forward CV), and returns
structured results with all artifacts written to a job-scoped
output directory.
"""

import copy
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from src.data import DataLoader
from src.executor import ModelExecutor
from src.utils.metrics import get_metric_fn
from src.visualization import Visualizer


@dataclass
class ExperimentResult:
    """Immutable result container returned by ExperimentRunner."""

    job_id: str
    output_dir: Path

    # DataFrames — always populated after a successful run
    predictions: Optional[pd.DataFrame] = None
    model_comparison: Optional[pd.DataFrame] = None
    dm_pvalues: Optional[pd.DataFrame] = None
    horizon_metrics: Optional[pd.DataFrame] = None
    shap_ranking: Optional[pd.DataFrame] = None

    # Walk-forward specific (None for single-window runs)
    per_window_metrics: Optional[pd.DataFrame] = None
    summary_metrics: Optional[pd.DataFrame] = None

    # Relative paths to generated plot images (relative to output_dir)
    plot_paths: list[str] = field(default_factory=list)


class ExperimentRunner:
    """
    Stateless experiment orchestrator.

    Decoupled from CLI and web layers — receives a SimpleNamespace
    config tree, returns structured ExperimentResult, writes all
    artifacts into outputs/{job_id}/.
    """

    def __init__(self, base_output_dir: str = "outputs"):
        self.base_output_dir = Path(base_output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single_window(
        self,
        cfg,
        job_id: Optional[str] = None,
        eval_metric: str = "rmse",
    ) -> ExperimentResult:
        """
        Run one train/test split: fit all models, predict, evaluate.
        """
        job_id, job_dir = self._make_job_dir(job_id)
        cfg = self._scope_outputs(cfg, job_dir)
        visualizer = self._make_visualizer(cfg)

        executor = ModelExecutor(cfg, eval_metric=eval_metric, visualizer=visualizer)
        predictions_df = executor.run(run_analysis=True)

        result = ExperimentResult(
            job_id=job_id,
            output_dir=job_dir,
            predictions=predictions_df,
        )

        # Attach evaluation results that executor already computed
        if executor.predictions_df is not None:
            result.model_comparison, result.dm_pvalues = (
                executor.evaluate_point_metrics()
            )
            result.horizon_metrics = executor.evaluate_multi_horizon()
            result.shap_ranking, _ = executor.analyze_shap()

        result.plot_paths = self._collect_plot_paths(job_dir)
        return result

    def run_walk_forward(
        self,
        cfg,
        job_id: Optional[str] = None,
        eval_metric: str = "rmse",
    ) -> ExperimentResult:
        """
        Run the full walk-forward cross-validation loop.
        The last window includes full analysis (EDA, SHAP, plots).
        """
        job_id, job_dir = self._make_job_dir(job_id)
        cfg = self._scope_outputs(cfg, job_dir)
        visualizer = self._make_visualizer(cfg)
        metric_fn = get_metric_fn(eval_metric)

        entity_col = getattr(
            cfg.data,
            "entity_column",
            getattr(cfg.data, "country_column", "Country"),
        )
        test_entities = getattr(
            cfg.data,
            "test_entities",
            getattr(cfg.data, "test_countries", []),
        )
        primary_entity = test_entities[0] if test_entities else None

        # Load data once for window generation
        loader = DataLoader(cfg)
        df = loader.load_dataset()
        loader.validate_schema(df)

        windows = self._generate_windows(cfg, df)
        records = []
        last_executor = None

        for i, (train_end_year, test_year) in enumerate(windows):
            is_last = i == len(windows) - 1

            # Create a config copy for this window (no in-place mutation)
            window_cfg = self._copy_cfg_for_window(cfg, train_end_year, test_year)

            executor = ModelExecutor(
                window_cfg,
                eval_metric=eval_metric,
                visualizer=visualizer if is_last else None,
            )
            try:
                predictions_df = executor.run(run_analysis=is_last)
            except Exception:
                continue

            if is_last:
                last_executor = executor

            if primary_entity and not predictions_df.empty:
                sub = predictions_df[predictions_df[entity_col] == primary_entity]
                if not sub.empty:
                    actual = sub["Actual"].values
                    row = {
                        "train_end_year": train_end_year,
                        "test_year": test_year,
                        "n_test": len(actual),
                    }
                    model_labels = [
                        c
                        for c in sub.columns
                        if c not in [entity_col, "Date", "Actual", "Horizon"]
                    ]
                    for label in model_labels:
                        row[label] = round(metric_fn(actual, sub[label].values), 4)
                    records.append(row)

        if not records:
            raise RuntimeError("Walk-forward: no windows produced valid results")

        # Build walk-forward summary tables
        per_window_df = pd.DataFrame(records)
        model_cols = [
            c
            for c in per_window_df.columns
            if c not in ["train_end_year", "test_year", "n_test"]
        ]
        summary_rows = []
        for col in model_cols:
            summary_rows.append(
                {
                    "Model": col,
                    f"mean_{eval_metric}": round(float(np.mean(per_window_df[col])), 4),
                    f"std_{eval_metric}": round(float(np.std(per_window_df[col])), 4),
                }
            )
        summary_df = pd.DataFrame(summary_rows)

        # Persist walk-forward CSVs
        wf_dir = Path(cfg.outputs.tables_dir) / "walk_forward"
        wf_dir.mkdir(parents=True, exist_ok=True)
        per_window_df.to_csv(wf_dir / f"per_window_{eval_metric}.csv", index=False)
        summary_df.to_csv(wf_dir / f"summary_{eval_metric}.csv", index=False)

        # Assemble result
        result = ExperimentResult(
            job_id=job_id,
            output_dir=job_dir,
            per_window_metrics=per_window_df,
            summary_metrics=summary_df,
        )

        # Attach last-window evaluation artifacts
        if last_executor and last_executor.predictions_df is not None:
            result.predictions = last_executor.predictions_df

        result.plot_paths = self._collect_plot_paths(job_dir)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_job_dir(self, job_id: Optional[str]) -> tuple[str, Path]:
        """Create outputs/{job_id}/ directory, generate UUID if needed."""
        job_id = job_id or str(uuid.uuid4())[:8]
        job_dir = self.base_output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_id, job_dir

    def _scope_outputs(self, cfg, job_dir: Path):
        """
        Return a shallow copy of cfg with outputs.* paths redirected
        into the job-scoped directory.
        """
        cfg = copy.deepcopy(cfg)
        cfg.outputs = SimpleNamespace(
            tables_dir=str(job_dir / "tables"),
            figures_dir=str(job_dir / "figures"),
            predictions_dir=str(job_dir / "predictions"),
            models_dir=str(job_dir / "models"),
        )
        return cfg

    def _make_visualizer(self, cfg) -> Visualizer:
        return Visualizer(
            cfg=cfg,
            figures_dir=cfg.outputs.figures_dir,
            save_plots=True,
            show_plots=False,
            style=getattr(cfg.figures, "style", "ggplot"),
            dpi=getattr(cfg.figures, "dpi", 300),
        )

    def _copy_cfg_for_window(self, cfg, train_end_year: int, test_year: int):
        """
        Create a deep copy of cfg with train/test dates set for this
        particular walk-forward window.  Avoids in-place mutation.
        """
        window_cfg = copy.deepcopy(cfg)
        window_cfg.data.train_end = f"{train_end_year}-12-31"
        window_cfg.data.test_start = f"{test_year}-01-01"
        window_cfg.data.test_end = f"{test_year}-12-31"
        return window_cfg

    @staticmethod
    def _generate_windows(cfg, df) -> list[tuple[int, int]]:
        """
        Compute walk-forward (train_end_year, test_year) sliding windows
        from the dataset date range and config parameters.
        """
        entity_col = getattr(
            cfg.data,
            "entity_column",
            getattr(cfg.data, "country_column", "Country"),
        )
        entities = df[entity_col].unique()
        base_entity = "UA" if "UA" in entities else entities[0]
        base_dates = df[df[entity_col] == base_entity]["Date"]

        base_first_year = getattr(cfg.data, "ua_train_start", None)
        if base_first_year:
            base_first_year = pd.Timestamp(base_first_year).year
        else:
            base_first_year = base_dates.min().year

        base_last_year = base_dates.max().year

        init = cfg.walk_forward.initial_train_years
        step = cfg.walk_forward.test_window_years
        windows = []
        test_year = base_first_year + init
        while test_year <= base_last_year:
            windows.append((test_year - step, test_year))
            test_year += step
        return windows

    @staticmethod
    def _collect_plot_paths(job_dir: Path) -> list[str]:
        """Glob all .png files under job_dir and return relative paths."""
        figures_dir = job_dir / "figures"
        if not figures_dir.exists():
            return []
        return [str(p.relative_to(job_dir)) for p in figures_dir.rglob("*.png")]
