"""
Experiment API endpoints.

POST /api/v1/experiments/run   — Launch an experiment (background task)
GET  /api/v1/experiments/jobs/{job_id}/status — Poll job status
"""

import random
import uuid

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException

from server.schemas import (
    ExperimentRequest,
    ExperimentResponse,
    JobStatusResponse,
    MetricRow,
)
from server.state import JobStatus, create_job, get_job, jobs
from src.services.experiment import ExperimentRunner

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


# ------------------------------------------------------------------
# Background task
# ------------------------------------------------------------------


def _run_experiment_task(job_id: str, request: ExperimentRequest):
    """Background task that runs the full experiment pipeline."""
    jobs[job_id]["status"] = JobStatus.RUNNING
    try:
        cfg = request.to_config_namespace()
        random.seed(cfg.project.random_seed)
        np.random.seed(cfg.project.random_seed)

        runner = ExperimentRunner(base_output_dir="outputs")

        if request.mode == "walk_forward":
            result = runner.run_walk_forward(
                cfg, job_id=job_id, eval_metric=request.eval_metric
            )
        else:
            result = runner.run_single_window(
                cfg, job_id=job_id, eval_metric=request.eval_metric
            )

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)


# ------------------------------------------------------------------
# Helper: build response from ExperimentResult
# ------------------------------------------------------------------


def _build_response(job_id: str, result, eval_metric: str) -> ExperimentResponse:
    """Convert ExperimentResult dataclass → Pydantic ExperimentResponse."""
    metrics = []
    if result.model_comparison is not None:
        for _, row in result.model_comparison.iterrows():
            metrics.append(
                MetricRow(
                    country=str(row.get("Country", "")),
                    model=str(row.get("Model", "")),
                    value=float(row.get("Metric", 0)),
                )
            )

    dm_summary = []
    if result.dm_pvalues is not None:
        dm_summary = result.dm_pvalues.to_dict(orient="records")

    plot_urls = [f"/static/jobs/{job_id}/{p}" for p in result.plot_paths]

    csv_urls = []
    tables_dir = result.output_dir / "tables"
    if tables_dir.exists():
        csv_urls = [
            f"/static/jobs/{job_id}/tables/{p.relative_to(tables_dir)}"
            for p in tables_dir.rglob("*.csv")
        ]

    # Build a small markdown summary
    summary_lines = [f"## Experiment `{job_id}`\n"]
    if result.summary_metrics is not None:
        summary_lines.append("### Walk-Forward Summary\n")
        summary_lines.append(result.summary_metrics.to_markdown(index=False))
    summary_md = "\n".join(summary_lines)

    return ExperimentResponse(
        job_id=job_id,
        status="completed",
        eval_metric=eval_metric,
        metrics=metrics,
        dm_test_summary=dm_summary,
        plot_urls=plot_urls,
        csv_urls=csv_urls,
        summary_markdown=summary_md,
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/run", status_code=202)
def launch_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """
    Queue an experiment for background execution.
    Returns immediately with a job_id for polling.
    """
    job_id = str(uuid.uuid4())[:8]
    create_job(job_id)

    background_tasks.add_task(_run_experiment_task, job_id, request)

    return {
        "job_id": job_id,
        "status": "pending",
        "poll_url": f"/api/v1/experiments/jobs/{job_id}/status",
    }


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll this endpoint to check if an experiment has finished."""
    entry = get_job(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response = JobStatusResponse(
        job_id=job_id,
        status=entry["status"],
        error=entry.get("error"),
    )

    if entry["status"] == JobStatus.COMPLETED and entry["result"] is not None:
        response.result = _build_response(job_id, entry["result"], eval_metric="rmse")

    return response
