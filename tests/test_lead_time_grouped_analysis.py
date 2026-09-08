"""Unit tests for the reusable functions in scripts/06_lead_time_grouped_analysis.py
(Task 6 E4 extension / E5a / E5b / E1 lead-time-level width distribution):
assign_lead_time_bucket, find_first_sign_flip, find_last_sign_flip_into_permanent_positive.

Loaded via importlib.util.spec_from_file_location, same mechanism
tests/test_data_size_sweep.py already uses for scripts/03_data_size_sweep.py (the
filename starts with a digit, so it cannot be imported as a normal module). Only
top-level definitions execute on load; main() only runs under
`if __name__ == "__main__"`, so importing this script also loads (but does not
run) scripts/03_data_size_sweep.py's own main(), which is side-effect-free for the
same reason.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_lead_time_grouped_analysis.py"
_SPEC = importlib.util.spec_from_file_location("lead_time_grouped_analysis_module", _SCRIPT_PATH)
lead_time_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lead_time_module)

assign_lead_time_bucket = lead_time_module.assign_lead_time_bucket
find_first_sign_flip = lead_time_module.find_first_sign_flip
find_last_sign_flip_into_permanent_positive = lead_time_module.find_last_sign_flip_into_permanent_positive
LEAD_TIME_BUCKETS = lead_time_module.LEAD_TIME_BUCKETS


def test_lead_time_buckets_match_task_6_e5b_brief_example():
    labels = [b[2] for b in LEAD_TIME_BUCKETS]
    assert labels == ["0-24h", "24-72h", "72-120h"]


def test_assign_lead_time_bucket_covers_every_real_archive_step_hours_value():
    # The real archive's step_hours grid is 0, 6, 12, ..., 120 (21 values, 6h
    # spacing) -- every one must land in exactly one bucket, with no gaps at the
    # bucket boundaries (24h and 72h in particular).
    step_hours = np.arange(0.0, 126.0, 6.0)
    labels = assign_lead_time_bucket(step_hours)
    assert not any(l is None for l in labels)
    assert dict(zip(step_hours, labels))[0.0] == "0-24h"
    assert dict(zip(step_hours, labels))[18.0] == "0-24h"
    assert dict(zip(step_hours, labels))[24.0] == "24-72h"  # boundary: goes to the higher bucket
    assert dict(zip(step_hours, labels))[66.0] == "24-72h"
    assert dict(zip(step_hours, labels))[72.0] == "72-120h"  # boundary: goes to the higher bucket
    assert dict(zip(step_hours, labels))[120.0] == "72-120h"  # last bucket is closed at its upper edge


def test_assign_lead_time_bucket_raises_on_out_of_range_value():
    with pytest.raises(ValueError, match="outside every"):
        assign_lead_time_bucket(np.array([200.0]))


def test_find_first_sign_flip_known_crossing():
    step_hours = [0.0, 6.0, 12.0, 18.0]
    diff = [-1.0, -0.5, 0.5, 1.0]  # crosses between 6 and 12
    h, reason = find_first_sign_flip(step_hours, diff)
    assert h == pytest.approx(6.0 + 0.5 / (0.5 - (-0.5)) * 6.0)  # frac=0.5 -> 9.0
    assert "6.0" in reason and "12.0" in reason


def test_find_first_sign_flip_no_crossing_returns_none():
    step_hours = [0.0, 6.0, 12.0]
    diff = [-1.0, -0.5, -0.2]  # always negative, never flips
    h, reason = find_first_sign_flip(step_hours, diff)
    assert h is None
    assert "no negative-to-positive" in reason


def test_find_first_sign_flip_reports_the_first_flip_not_a_later_one():
    # Mirrors the real data's noisy pattern: flips negative->positive->negative->
    # positive. "first" must be the EARLIEST flip, not the last.
    step_hours = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
    diff = [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    h, reason = find_first_sign_flip(step_hours, diff)
    assert h == pytest.approx(3.0)  # first flip between 0 and 6, frac=0.5 -> 3.0


def test_find_last_sign_flip_into_permanent_positive_skips_earlier_temporary_flips():
    # Same noisy pattern as above: the FIRST flip (0->6) reverts (12), so it is not
    # "durable". The durable flip is the LAST negative-to-non-negative transition
    # after which every remaining point stays non-negative -- here, between 18 and
    # 24 (24 and 30 are both positive and are the only remaining points).
    step_hours = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
    diff = [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    h, reason = find_last_sign_flip_into_permanent_positive(step_hours, diff)
    assert h == pytest.approx(21.0)  # between 18 (-1.0) and 24 (1.0), frac=0.5 -> 21.0
    assert "18.0" in reason and "24.0" in reason


def test_find_last_sign_flip_into_permanent_positive_none_if_always_negative():
    step_hours = [0.0, 6.0, 12.0]
    diff = [-1.0, -0.5, -0.2]
    h, reason = find_last_sign_flip_into_permanent_positive(step_hours, diff)
    assert h is None


def test_find_last_sign_flip_into_permanent_positive_none_if_always_positive():
    step_hours = [0.0, 6.0, 12.0]
    diff = [1.0, 0.5, 0.2]
    h, reason = find_last_sign_flip_into_permanent_positive(step_hours, diff)
    assert h is None
