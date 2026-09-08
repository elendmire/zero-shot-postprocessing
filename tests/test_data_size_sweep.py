"""Unit tests for the reusable functions in scripts/03_data_size_sweep.py that
Task 5 imports per this plan's Interfaces section (find_breakpoint, sample_contiguous,
sample_random, compute_metrics, fit_predict_local_emos).

scripts/03_data_size_sweep.py is not an importable package module (its filename starts
with a digit, so `import scripts.03_data_size_sweep` is a syntax error) — it is loaded
here via importlib.util.spec_from_file_location, the same mechanism Task 5's own code
will need to reuse these functions. Only top-level imports/definitions execute on load;
main() only runs under `if __name__ == "__main__"`, so importing is side-effect-free.

Added in fix-round-1 (revised R3a): find_breakpoint previously had zero direct tests
despite being flagged as a reusable interface Task 5 imports, and two real direction
bugs in code that CALLS find_breakpoint were only caught by eyeballing a figure. These
5 tests cover find_breakpoint's own contract directly and mechanically: a known
crossing, no crossing, a NaN row, a non-monotonic curve, and a single-element array.
"""
import importlib.util
import math
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_data_size_sweep.py"
_SPEC = importlib.util.spec_from_file_location("data_size_sweep_module", _SCRIPT_PATH)
data_size_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data_size_sweep)

find_breakpoint = data_size_sweep.find_breakpoint


def test_find_breakpoint_known_ascending_crossing():
    # value rises from 1.0 (below reference=2.0) to 3.0 (above it) between n=10 and n=20.
    n_cases = [0, 10, 20, 30]
    values = [0.5, 1.0, 3.0, 3.5]
    bp = find_breakpoint(n_cases, values, reference_value=2.0, better="higher")
    assert bp is not None
    assert 10 < bp < 20
    # linear interpolation: frac = (2.0 - 1.0) / (3.0 - 1.0) = 0.5 -> bp = 10 + 0.5*10 = 15
    assert bp == pytest.approx(15.0)


def test_find_breakpoint_no_crossing():
    # always above reference=2.0; better="lower" is never achieved anywhere in range.
    n_cases = [10, 20, 30]
    values = [5.0, 4.0, 3.5]
    bp = find_breakpoint(n_cases, values, reference_value=2.0, better="lower")
    assert bp is None


def test_find_breakpoint_skips_nan_row_and_still_finds_the_real_crossing():
    # NaN at n=0 (e.g. this sweep's n_days=0 "undefined, no data to fit" placeholder
    # row) must not crash find_breakpoint, and must not prevent it from finding the
    # real crossing among the remaining, valid pairs.
    n_cases = [0, 10, 20, 30]
    values = [float("nan"), 5.0, 3.0, 1.0]
    bp = find_breakpoint(n_cases, values, reference_value=2.0, better="lower")
    assert bp is not None
    assert 20 < bp < 30
    assert bp == pytest.approx(25.0)


def test_find_breakpoint_non_monotonic_curve_reports_first_crossing():
    # values wiggle: starts above reference (worse), dips below (crosses), then back
    # above. find_breakpoint's documented contract is to walk pairs in ascending
    # n_cases order and return the FIRST crossing found, not the last or "the"
    # crossing (there can be more than one on a non-monotonic curve).
    n_cases = [0, 10, 20, 30]
    values = [5.0, 1.0, 4.0, 1.0]
    bp = find_breakpoint(n_cases, values, reference_value=2.0, better="lower")
    assert bp is not None
    assert 0 < bp < 10
    # frac = (2.0 - 5.0) / (1.0 - 5.0) = 0.75 -> bp = 0 + 0.75*10 = 7.5
    assert bp == pytest.approx(7.5)


def test_find_breakpoint_single_element_returns_none():
    # a single (n_cases, value) pair has no adjacent pair to interpolate a crossing
    # between — must return None, not raise (e.g. IndexError on an empty zip).
    bp = find_breakpoint([10], [5.0], reference_value=2.0, better="lower")
    assert bp is None


# --- Task 6 E3: n_days_for_exact_k, the k-driven-sampling helper ---

n_days_for_exact_k = data_size_sweep.n_days_for_exact_k
n_days_to_k = data_size_sweep.n_days_to_k


def test_n_days_for_exact_k_round_trips_exactly_for_the_low_n_grid():
    # This is the crux of E3's "drive by k directly" requirement: converting k ->
    # n_days -> k must return EXACTLY the original k, with no rounding drift, for
    # every value in LOW_N_K_GRID -- a human-friendly integer n_days would not
    # reliably satisfy this at small k (that's the whole reason E3 exists).
    for k in data_size_sweep.LOW_N_K_GRID:
        assert n_days_to_k(n_days_for_exact_k(k)) == k


def test_n_days_for_exact_k_uses_measured_days_per_case():
    k = 5
    expected = k * data_size_sweep.DAYS_PER_CASE
    assert n_days_for_exact_k(k) == pytest.approx(expected)


def test_low_n_k_grid_matches_task_6_e3_brief():
    assert data_size_sweep.LOW_N_K_GRID == [1, 2, 3, 5, 7]


# --- Task 6 E1: LAMBDA_CLIM is a real, documented constant, not accidentally unset ---

def test_lambda_clim_is_a_positive_finite_constant():
    assert isinstance(data_size_sweep.LAMBDA_CLIM, float)
    assert data_size_sweep.LAMBDA_CLIM > 1.0  # must actually inflate, not shrink, the spread
    # Fix-round-1 minor item: the test's own name promises "finite" but never
    # actually asserted it -- math.isfinite rejects both inf and nan (isinstance
    # float + > 1.0 alone would pass for float("inf")).
    assert math.isfinite(data_size_sweep.LAMBDA_CLIM)


# --- Fix-round-1 finding 3: bp_to_calendar_days, the single-rounding breakpoint ->
# calendar-days conversion that replaces the double-rounding k_to_calendar_days(round(bp)) ---

bp_to_calendar_days = data_size_sweep.bp_to_calendar_days
DAYS_PER_CASE = data_size_sweep.DAYS_PER_CASE


def test_bp_to_calendar_days_rounds_once_not_twice():
    # This is the exact regression case from the review finding: bp=1.64 previously
    # went through k_to_calendar_days(round(1.64)) = k_to_calendar_days(2) = 7 days
    # (2 * 3.4833 rounded), inflating the true 1.64 * 3.4833 ~= 5.71 -> 6 days.
    bp = 1.64
    assert bp_to_calendar_days(bp) == round(bp * DAYS_PER_CASE)
    assert bp_to_calendar_days(bp) != round(round(bp) * DAYS_PER_CASE)


def test_bp_to_calendar_days_none_passes_through():
    assert bp_to_calendar_days(None) is None
