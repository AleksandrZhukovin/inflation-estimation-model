import random
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.config import load_config
from src.eda import run as run_eda
from src.arima import run as run_arima
from src.features import run as run_features
from src.tuning import run as run_tuning
from src.xgboost_model import run as run_xgboost
from src.hybrid import run as run_hybrid
from src.benchmarks import run as run_benchmarks
from src.evaluation import run as run_evaluation
from src.multi_horizon import run as run_multi_horizon
from src.shap_analysis import run as run_shap

PHASES = [
    ("EDA", run_eda),
    ("ARIMA fitting", run_arima),
    ("Feature engineering", run_features),
    ("Hyperparameter tuning", run_tuning),
    ("XGBoost training", run_xgboost),
    ("Hybrid prediction", run_hybrid),
    ("Benchmarks", run_benchmarks),
    ("Evaluation", run_evaluation),
    ("Multi-horizon", run_multi_horizon),
    ("SHAP analysis", run_shap),
]


def main():
    cfg = load_config()
    random.seed(cfg.project.random_seed)
    np.random.seed(cfg.project.random_seed)

    t_total = time.perf_counter()
    for name, fn in PHASES:
        t0 = time.perf_counter()
        fn(cfg)
        print(f"[OK] {name} — {time.perf_counter() - t0:.1f}s")

    print(f"\nDone — {time.perf_counter() - t_total:.1f}s total")


if __name__ == "__main__":
    main()
