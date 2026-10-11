"""Unit tests for the helper functions of scripts/training_size.py (find_breakpoint, sampling, metrics)."""
import math

import pytest

import training_size

find_breakpoint = training_size.find_breakpoint


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


# n_days_for_exact_k

n_days_for_exact_k = training_size.n_days_for_exact_k
n_days_to_k = training_size.n_days_to_k


def test_n_days_for_exact_k_round_trips_exactly_for_the_low_n_grid():
    # k -> n_days -> k returns the original k for every k of the small grid
    for k in training_size.LOW_N_K_GRID:
        assert n_days_to_k(n_days_for_exact_k(k)) == k


def test_n_days_for_exact_k_uses_measured_days_per_case():
    k = 5
    expected = k * training_size.DAYS_PER_CASE
    assert n_days_for_exact_k(k) == pytest.approx(expected)


def test_low_n_k_grid_matches_specified_grid():
    assert training_size.LOW_N_K_GRID == [1, 2, 3, 5, 7]


# LAMBDA_CLIM

def test_lambda_clim_is_a_positive_finite_constant():
    assert isinstance(training_size.LAMBDA_CLIM, float)
    assert training_size.LAMBDA_CLIM > 1.0  # inflates the spread
    assert math.isfinite(training_size.LAMBDA_CLIM)


# bp_to_calendar_days

bp_to_calendar_days = training_size.bp_to_calendar_days
DAYS_PER_CASE = training_size.DAYS_PER_CASE


def test_bp_to_calendar_days_rounds_once_not_twice():
    # rounded once: 1.64 cases -> round(1.64 * DAYS_PER_CASE) days, not round(round(1.64) * DAYS_PER_CASE)
    bp = 1.64
    assert bp_to_calendar_days(bp) == round(bp * DAYS_PER_CASE)
    assert bp_to_calendar_days(bp) != round(round(bp) * DAYS_PER_CASE)


def test_bp_to_calendar_days_none_passes_through():
    assert bp_to_calendar_days(None) is None
