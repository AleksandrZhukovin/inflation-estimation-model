"""
CLI entry point for the inflation estimation model pipeline.

Thin wrapper over ExperimentRunner — all orchestration logic
lives in src/services/experiment.py.
"""

import random
import time

import numpy as np

from src.config import load_config
from src.services.experiment import ExperimentRunner


def main():
    cfg = load_config()
    random.seed(cfg.project.random_seed)
    np.random.seed(cfg.project.random_seed)

    t0 = time.perf_counter()
    runner = ExperimentRunner()
    result = runner.run_walk_forward(cfg)
    elapsed = time.perf_counter() - t0

    print(f"Done — {elapsed:.1f}s | Job: {result.job_id}")
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
