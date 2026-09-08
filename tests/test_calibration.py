import numpy as np
import pytest

from zeropp.eval.calibration import (
    empirical_coverage,
    pit_histogram,
    pit_values,
    reliability_index,
)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _flat_quantile_preds(n_samples, n_leads):
    base = np.array(QUANTILE_LEVELS) * 10  # quantiles at 1,2,...,9
    return np.tile(base, (n_samples, n_leads, 1))


def test_pit_values_at_median_quantile_is_near_half():
    quantile_preds = _flat_quantile_preds(1, 1)
    y_true = np.array([[5.0]])  # matches the tau=0.5 quantile value exactly
    pit = pit_values(y_true, quantile_preds, QUANTILE_LEVELS)
    assert pit.shape == (1, 1)
    assert pit[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_pit_values_clip_below_min_quantile_to_zero():
    quantile_preds = _flat_quantile_preds(1, 1)
    y_true = np.array([[-100.0]])
    pit = pit_values(y_true, quantile_preds, QUANTILE_LEVELS)
    assert pit[0, 0] == pytest.approx(0.1, abs=1e-6)


def test_pit_histogram_sums_to_one():
    pit = np.array([0.05, 0.15, 0.5, 0.95])
    hist = pit_histogram(pit, n_bins=10)
    assert hist.shape == (10,)
    assert hist.sum() == pytest.approx(1.0)


def test_empirical_coverage_full_when_all_inside_band():
    quantile_preds = _flat_quantile_preds(3, 1)
    y_true = np.array([[2.0], [5.0], [8.0]])
    coverage = empirical_coverage(y_true, quantile_preds, QUANTILE_LEVELS)
    assert coverage == pytest.approx(1.0)


def test_empirical_coverage_partial_when_some_outside_band():
    quantile_preds = _flat_quantile_preds(2, 1)
    y_true = np.array([[-5.0], [5.0]])  # first below q10=1.0, second inside
    coverage = empirical_coverage(y_true, quantile_preds, QUANTILE_LEVELS)
    assert coverage == pytest.approx(0.5)


def test_reliability_index_zero_for_uniform_pit():
    pit = np.linspace(0.0, 1.0, 1000, endpoint=False)
    idx = reliability_index(pit, n_bins=10)
    assert idx == pytest.approx(0.0, abs=1e-3)


def test_reliability_index_positive_for_skewed_pit():
    pit = np.full(100, 0.5)
    idx = reliability_index(pit, n_bins=10)
    assert idx > 0.0


def test_pit_values_raises_on_quantile_crossing():
    quantile_preds = np.array([[[1, 2, 3, 4, 9, 6, 7, 8, 9]]], dtype=float)  # index 4 (9.0) > index 5 (6.0): crossing
    y_true = np.array([[5.0]])
    with pytest.raises(ValueError, match="crossing"):
        pit_values(y_true, quantile_preds, QUANTILE_LEVELS)


def test_empirical_coverage_raises_on_inverted_band():
    quantile_preds = _flat_quantile_preds(1, 1).copy()
    quantile_preds[..., 0] = 20.0  # q10 now way above q90=9.0, inverted band
    y_true = np.array([[5.0]])
    with pytest.raises(ValueError, match="q_lower"):
        empirical_coverage(y_true, quantile_preds, QUANTILE_LEVELS)


def test_pit_histogram_raises_on_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        pit_histogram(np.array([]))
