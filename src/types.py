"""
Type definitions for models.

TypedDicts, type aliases, and other type-related
classes used across model modules.
"""

from typing import TypedDict, NotRequired


class XGBoostParams(TypedDict):
    eta: float
    max_depth: int
    min_child_weight: NotRequired[int]
    gamma: float
    reg_alpha: NotRequired[float]
    reg_lambda: float
    subsample: float
    colsample_bytree: float
