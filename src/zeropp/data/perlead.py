"""Lead-matched daily series for the zero-shot models.

For a station and lead l the series is indexed by issue day d: target = observation at d + l, covariate =
ensemble mean (optionally spread) of the forecast issued on d for lead l. At issue day t the last known
target is d* = t - floor(l/24) - 1, the horizon is t - d*, and only the last step is scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def last_known_offset(lead_hours: float) -> int:
    """Number of days between issue day t and the last known target index d*."""
    return int(np.floor(lead_hours / 24.0)) + 1


def horizon_steps(lead_hours: float) -> int:
    return last_known_offset(lead_hours)


def daily_arrays(issue_days: np.ndarray, *cols: np.ndarray):
    """Place per-issue-day values on a complete daily calendar (NaN gaps).

    issue_days: datetime64[D] array (unique). Returns (calendar, *arrays).
    """
    issue_days = np.asarray(issue_days).astype("datetime64[D]")
    cal = np.arange(issue_days.min(), issue_days.max() + 1, dtype="datetime64[D]")
    pos = (issue_days - cal[0]).astype(int)
    out = []
    for c in cols:
        a = np.full(len(cal), np.nan)
        a[pos] = c
        out.append(a)
    return (cal, *out)


def build_instances(y: np.ndarray, x: np.ndarray, lead_hours: float, window: int,
                    score_idx: np.ndarray | None = None, x2: np.ndarray | None = None):
    """Build instances for one (station, lead) from daily arrays y, x (len = calendar).

    Returns list of dicts with idx (issue-day index t), context (W,), covariate
    (W + h,) (or (2, W + h) if x2 given), horizon h, obs y[t]. Skips if fewer
    than `window` known targets or any NaN in context/covariates (including
    the scored step's own covariate and target).
    """
    k = last_known_offset(lead_hours)
    h = k
    n = len(y)
    idxs = range(n) if score_idx is None else score_idx
    out = []
    for t in idxs:
        t = int(t)
        dstar = t - k
        lo = dstar - window + 1
        if lo < 0 or t >= n:
            continue
        ctx = y[lo: dstar + 1]
        cov = x[lo: t + 1]
        if np.isnan(ctx).any() or np.isnan(cov).any() or np.isnan(y[t]):
            continue
        if x2 is not None:
            cov2 = x2[lo: t + 1]
            if np.isnan(cov2).any():
                continue
            cov = np.stack([cov, cov2], axis=0)
        out.append(dict(idx=t, context=ctx, covariate=cov, horizon=h, obs=float(y[t])))
    return out
