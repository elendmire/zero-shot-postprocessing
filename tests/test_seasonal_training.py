import numpy as np
import pandas as pd
import pytest

import seasonal_training as seasonal


def test_season_of_month_covers_every_month():
    seasons = [seasonal.season_of_month(m) for m in range(1, 13)]
    assert seasons == ["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"]
    with pytest.raises(ValueError):
        seasonal.season_of_month(13)


def test_time_idx_to_issue_date_handles_integer_positions():
    template = np.array(["2017-01-02", "2017-06-01", "2018-12-31"], dtype="datetime64[ns]")
    out = seasonal.time_idx_to_issue_date(pd.Series([2, 0]), template)
    assert list(out.dt.month) == [12, 1]


def test_sample_contiguous_k_takes_most_recent_cases():
    cases = pd.DataFrame({"year_idx": [1, 1, 2, 2], "time_idx": [0, 5, 0, 5],
                          "case_date": pd.to_datetime(["2000-01-01", "2000-06-01", "2001-01-01", "2001-06-01"])})
    got = seasonal.sample_contiguous_k(cases, 3)
    assert list(zip(got["year_idx"], got["time_idx"])) == [(2, 5), (2, 0), (1, 5)]


def test_first_durable_significant_k_requires_significance_at_all_larger_k():
    sig = pd.DataFrame({
        "k_cases": [1, 2, 3, 5],
        "block_mean_diff": [-0.1, -0.1, 0.05, -0.1],
        "t_pvalue": [0.01, 0.01, 0.01, 0.01],
    })
    assert seasonal.first_durable_significant_k(sig) == 5
    sig.loc[2, "block_mean_diff"] = -0.1
    assert seasonal.first_durable_significant_k(sig) == 1
