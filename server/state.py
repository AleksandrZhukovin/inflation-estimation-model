"""
In-memory job store for tracking background experiment runs.

Simple dict-based store; can be swapped for Redis/DB later
without changing the router interface.
"""

from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# job_id -> {"status": JobStatus, "result": ExperimentResult | None, "error": str | None}
jobs: dict[str, dict[str, Any]] = {}


def create_job(job_id: str) -> dict:
    """Register a new job in the store."""
    entry = {"status": JobStatus.PENDING, "result": None, "error": None}
    jobs[job_id] = entry
    return entry


def get_job(job_id: str) -> Optional[dict]:
    """Retrieve job entry, or None if not found."""
    return jobs.get(job_id)
