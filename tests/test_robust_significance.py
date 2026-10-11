import numpy as np
import pandas as pd
import pytest

from zeropp.eval.robust_significance import (
    daily_mean_series,
    hac_dm_test,
    moving_block_bootstrap_ci,
    newey_west_variance,
)


def test_daily_mean_series_averages_within_dates():
    vt = pd.to_datetime(["2017-01-02 00:00", "2017-01-02 12:00", "2017-01-01 06:00"])
    s = daily_mean_series(np.array([1.0, 3.0, 5.0]), vt)
    assert list(s.values) == [5.0, 2.0]


def test_newey_west_equals_plain_variance_at_lag_zero():
    d = np.array([1.0, -1.0, 2.0, 0.0])
    assert newey_west_variance(d, 0) == pytest.approx(np.var(d))


def test_hac_dm_detects_clear_shift_and_not_noise():
    rng = np.random.default_rng(0)
    assert hac_dm_test(rng.normal(-0.5, 1.0, 400))["dm_pvalue"] < 1e-6
    assert hac_dm_test(rng.normal(0.0, 1.0, 400))["dm_pvalue"] > 0.01


def test_hac_widens_inference_under_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=600)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, len(e)):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    assert newey_west_variance(ar, 10) > 2 * newey_west_variance(ar, 0)


def test_moving_block_ci_contains_mean():
    rng = np.random.default_rng(2)
    d = rng.normal(0.3, 1.0, 300)
    lo, hi = moving_block_bootstrap_ci(d, block=7, n_boot=500)
    assert lo < d.mean() < hi
