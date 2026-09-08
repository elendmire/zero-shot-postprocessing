"""Unit tests for the one new reusable function in
scripts/07_spatial_block_significance.py (day_block_ids) -- the calendar-date
normalization helper that turns a valid_time array into day-blocked
`block_ids`, the counterpart to this project's existing station_id block_ids.

scripts/07_spatial_block_significance.py is not an importable package module
(its filename starts with a digit), so it is loaded here via
importlib.util.spec_from_file_location, the same mechanism
tests/test_data_size_sweep.py already uses for scripts/03_data_size_sweep.py.
Only top-level imports/definitions execute on load; main() only runs under
`if __name__ == "__main__"`, so importing is side-effect-free (and does not
touch any NetCDF/parquet files, which only exist on the SSH server).

station_blocked_paired_test / block_bootstrap_skill_score_ci themselves are
NOT re-tested here -- they are reused verbatim from zeropp.eval.significance,
which already has its own dedicated tests (tests/test_significance.py); this
script only changes what `block_ids` array is passed to them.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_spatial_block_significance.py"
_SPEC = importlib.util.spec_from_file_location("spatial_block_significance_module", _SCRIPT_PATH)
spatial_block_significance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(spatial_block_significance)

day_block_ids = spatial_block_significance.day_block_ids


def test_day_block_ids_same_calendar_date_different_time_of_day_share_a_block():
    valid_time = pd.to_datetime(["2019-03-01 00:00:00", "2019-03-01 12:00:00", "2019-03-01 18:00:00"])
    blocks = day_block_ids(valid_time)
    assert len(np.unique(blocks)) == 1


def test_day_block_ids_different_calendar_dates_get_different_blocks():
    valid_time = pd.to_datetime(["2019-03-01 06:00:00", "2019-03-02 06:00:00", "2019-03-03 06:00:00"])
    blocks = day_block_ids(valid_time)
    assert len(np.unique(blocks)) == 3


def test_day_block_ids_preserves_length_and_row_order():
    valid_time = pd.to_datetime([
        "2019-03-02 06:00:00", "2019-03-01 00:00:00", "2019-03-01 12:00:00", "2019-03-05 00:00:00",
    ])
    blocks = day_block_ids(valid_time)
    assert len(blocks) == len(valid_time)
    # rows 1 and 2 share a calendar date (2019-03-01); rows 0 and 3 do not match
    # each other or row 1/2 -- row order (not sorted order) must be preserved.
    assert blocks[1] == blocks[2]
    assert blocks[0] != blocks[1]
    assert blocks[0] != blocks[3]
    assert blocks[3] != blocks[1]


def test_day_block_ids_zeroes_time_of_day():
    valid_time = pd.to_datetime(["2019-03-01 17:30:00"])
    blocks = day_block_ids(valid_time)
    ts = pd.Timestamp(blocks[0])
    assert ts.hour == 0 and ts.minute == 0 and ts.second == 0


def test_day_block_ids_accepts_a_pandas_series_input():
    # matched_keys["valid_time"] (a pandas Series column), the exact input shape
    # main() passes -- not just a bare DatetimeIndex/list.
    series = pd.Series(pd.to_datetime(["2019-03-01 06:00:00", "2019-03-02 06:00:00"]), name="valid_time")
    blocks = day_block_ids(series)
    assert len(blocks) == 2
    assert blocks[0] != blocks[1]


def test_day_block_ids_output_length_matches_a_larger_multi_station_input():
    # Sanity check against the real use case: many stations sharing a small
    # number of distinct calendar dates should collapse to far fewer unique
    # blocks than rows -- this is the entire point of day-blocking (it groups
    # same-day instances across ALL stations into one block).
    rng = np.random.default_rng(0)
    dates = pd.to_datetime(["2019-03-01", "2019-03-02", "2019-03-03"])
    hours = rng.integers(0, 24, size=147)
    valid_time = [dates[i % 3] + pd.Timedelta(hours=int(h)) for i, h in enumerate(hours)]
    blocks = day_block_ids(pd.Series(valid_time))
    assert len(blocks) == 147
    assert len(np.unique(blocks)) == 3


@pytest.mark.parametrize("bad_input", [[], pd.Series([], dtype="datetime64[ns]")])
def test_day_block_ids_handles_empty_input_without_raising(bad_input):
    blocks = day_block_ids(bad_input)
    assert len(blocks) == 0
