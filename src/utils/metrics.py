"""
Models metrics

RMSE, MAE, MAPE errors and function for selection appropriate metric function
"""

import numpy as np


def rmse(actual, pred):
    """
    Calculate Root Mean Squared Error, ignoring NaN values in predictions.
    """
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


def mae(actual, pred):
    """
    Calculate Mean Absolute Error, ignoring NaN values in predictions.
    """
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def mape(actual, pred):
    """
    Calculate Mean Absolute Percentage Error, ignoring NaN values in predictions.
    """
    mask = ~np.isnan(pred) & (actual != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def get_metric_fn(metric_name: str):
    """
    Returns the corresponding metric function for a given string name.
    """
    metric_name = metric_name.lower().strip()
    if metric_name == "rmse":
        return rmse
    elif metric_name == "mae":
        return mae
    elif metric_name == "mape":
        return mape
    else:
        raise ValueError(f"Unknown metric '{metric_name}'. Supported: rmse, mae, mape")
