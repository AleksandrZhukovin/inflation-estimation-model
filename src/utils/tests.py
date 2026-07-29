"""
Statistical tests

Debold-Mariano test for determing statistical
significance of the difference between two forecast models.
"""

import numpy as np
from scipy import stats


def dm_test(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano statistical test for comparing forecast accuracy.
    Returns (statistic, p-value).
    """
    mask = (~np.isnan(pred1)) & (~np.isnan(pred2))
    if mask.sum() < 4:
        return float("nan"), float("nan")

    e1 = actual[mask] - pred1[mask]
    e2 = actual[mask] - pred2[mask]
    d = e1**2 - e2**2

    T = len(d)
    d_bar = np.mean(d)

    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2.0 * gamma_k
    lrv = gamma0 + gamma_sum

    if lrv <= 0:
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(lrv / T)

    k = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if k <= 0:
        k = 1.0
    dm_star = dm_stat * np.sqrt(k)

    p_value = float(2 * stats.t.sf(np.abs(dm_star), df=T - 1))

    return float(dm_star), p_value
