"""Tests of the mean daily loss difference between two methods that allow for serial dependence.

The per-instance differences are averaged within each valid date (series d_t). A Diebold-Mariano test with
a Newey-West (Bartlett) variance and the Harvey-Leybourne-Newbold correction, and a moving-block bootstrap
interval, are computed on d_t.
"""
import numpy as np
import pandas as pd
from scipy import stats


def daily_mean_series(diff: np.ndarray, valid_time) -> pd.Series:
    """Mean of the per-instance differences on each valid date, in date order."""
    days = pd.to_datetime(pd.Series(valid_time)).dt.normalize().to_numpy()
    return pd.Series(np.asarray(diff, dtype=float)).groupby(days).mean().sort_index()


def newey_west_variance(d: np.ndarray, lag: int) -> float:
    """Long-run variance of d with Bartlett weights up to `lag`."""
    d = np.asarray(d, dtype=float)
    e = d - d.mean()
    n = len(e)
    lrv = np.dot(e, e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * np.dot(e[k:], e[:-k]) / n
    return float(lrv)


def hac_dm_test(d: np.ndarray, lag: int = 5) -> dict:
    """Diebold-Mariano test of mean(d) = 0 with a HAC variance and the
    Harvey et al. (1997) small-sample correction, compared with a t
    distribution with n - 1 degrees of freedom."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    lrv = newey_west_variance(d, lag)
    stat = d.mean() / np.sqrt(lrv / n)
    h = lag + 1
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat_hln = stat * correction
    p = 2.0 * stats.t.sf(abs(stat_hln), df=n - 1)
    return {"n_days": n, "mean_diff": float(d.mean()), "dm_stat": float(stat_hln), "dm_pvalue": float(p), "hac_lag": lag}


def moving_block_bootstrap_ci(d: np.ndarray, block: int = 7, n_boot: int = 2000, ci: float = 0.95,
                              seed: int = 0) -> tuple[float, float]:
    """Percentile CI of mean(d) from a moving-block bootstrap with blocks of
    `block` consecutive days."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = d[idx].mean()
    alpha = 1.0 - ci
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def robust_comparison(loss_a: np.ndarray, loss_b: np.ndarray, valid_time, lag: int = 5, block: int = 7,
                      seed: int = 0) -> dict:
    """HAC-DM test and moving-block CI for the mean daily loss difference a - b."""
    d = daily_mean_series(np.asarray(loss_a) - np.asarray(loss_b), valid_time).to_numpy()
    out = hac_dm_test(d, lag=lag)
    out["mbb_ci_low"], out["mbb_ci_high"] = moving_block_bootstrap_ci(d, block=block, seed=seed)
    out["mbb_block_days"] = block
    return out
