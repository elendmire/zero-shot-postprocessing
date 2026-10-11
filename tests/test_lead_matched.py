import numpy as np
import pytest

from zeropp.data.perlead import build_instances, daily_arrays, last_known_offset


@pytest.mark.parametrize("lead,k", [(0, 1), (6, 1), (24, 2), (30, 2), (48, 3), (120, 6)])
def test_known_cutoff_offset(lead, k):
    assert last_known_offset(lead) == k
    # brute force: y_d known at t iff d + lead/24 < t strictly
    t = 100
    known = [d for d in range(0, t + 1) if d + lead / 24.0 < t]
    assert max(known) == t - k


@pytest.mark.parametrize("lead", [0, 6, 24, 30, 120])
def test_context_ends_at_last_known_and_scored_step_is_issue_t(lead):
    n, W = 60, 10
    y = np.arange(n, dtype=float)          # y_d = d (so values reveal the index)
    x = 1000.0 + np.arange(n, dtype=float)  # x_d = 1000 + d
    t = 50
    inst = build_instances(y, x, lead, W, score_idx=[t])[0]
    k = last_known_offset(lead)
    # last context value is y_{t-k}; valid time t-k+lead/24 < t
    assert inst["context"][-1] == t - k
    assert inst["context"][-1] + lead / 24.0 < t
    assert len(inst["context"]) == W
    # covariate = past W + future h values, ending at the forecast issued at t
    assert inst["horizon"] == k
    assert len(inst["covariate"]) == W + k
    assert inst["covariate"][-1] == 1000.0 + t
    assert inst["covariate"][W - 1] == 1000.0 + (t - k)
    # scored target is the observation for issue day t
    assert inst["obs"] == y[t] == t


def test_skips_short_history_and_nans():
    n, W = 30, 10
    y = np.arange(n, dtype=float)
    x = np.arange(n, dtype=float)
    assert build_instances(y, x, 0, W, score_idx=[5]) == []
    y2 = y.copy(); y2[20] = np.nan
    # t=25, lead 0: context idx 15..24 contains the NaN
    assert build_instances(y2, x, 0, W, score_idx=[25]) == []
    x2 = x.copy(); x2[25] = np.nan
    assert build_instances(y, x2, 0, W, score_idx=[25]) == []
    assert len(build_instances(y, x, 0, W, score_idx=[25])) == 1


def test_daily_arrays_gaps():
    days = np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[D]")
    cal, a = daily_arrays(days, np.array([1.0, 3.0]))
    assert len(cal) == 3 and np.isnan(a[1]) and a[2] == 3.0
