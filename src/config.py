"""
Configuration loader

Read config.yaml file with experiment params.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(item) for item in obj]
    return obj


def load_config(config_path="config.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _to_ns(raw)
