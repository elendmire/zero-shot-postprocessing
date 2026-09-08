import numpy as np
import pytest

from zeropp.eval.scores import (
    crps_from_quantiles,
    mae_from_quantiles,
    pinball_loss,
    twcrps_from_quantiles,
)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_pinball_loss_zero_at_perfect_forecast():
    y_true = np.array([5.0, 5.0])
    q_pred = np.array([5.0, 5.0])
    loss = pinball_loss(y_true, q_pred, tau=0.5)
    assert np.allclose(loss, 0.0)


def test_pinball_loss_asymmetric_penalty():
    y_true = np.array([10.0])
    q_pred = np.array([8.0])
    low_tau_loss = pinball_loss(y_true, q_pred, tau=0.1)
    high_tau_loss = pinball_loss(y_true, q_pred, tau=0.9)
    assert high_tau_loss > low_tau_loss


def test_crps_zero_for_degenerate_perfect_quantiles():
    y_true = np.full((3, 2), 5.0)
    quantile_preds = np.full((3, 2, len(QUANTILE_LEVELS)), 5.0)
    crps = crps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
    assert crps.shape == (3, 2)
    assert np.allclose(crps, 0.0)


def test_crps_positive_for_wrong_forecast():
    y_true = np.full((2, 1), 10.0)
    quantile_preds = np.tile(
        np.array([QUANTILE_LEVELS]) * 0 + 5.0, (2, 1, 1)
    ).reshape(2, 1, len(QUANTILE_LEVELS))
    crps = crps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
    assert np.all(crps > 0)


def test_mae_uses_median_quantile():
    y_true = np.array([[7.0]])
    quantile_preds = np.arange(1, 10, dtype=float).reshape(1, 1, 9)  # median (tau=0.5) is index 4 -> value 5.0
    mae = mae_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
    assert mae.shape == (1, 1)
    assert mae[0, 0] == pytest.approx(2.0)  # |7 - 5|


def test_twcrps_only_uses_tail_levels():
    y_true = np.full((1, 1), 5.0)
    quantile_preds = np.full((1, 1, 9), 5.0)
    twcrps = twcrps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS, tail_levels=(0.8, 0.9))
    assert twcrps.shape == (1, 1)
    assert np.allclose(twcrps, 0.0)


def test_twcrps_raises_on_level_not_in_quantile_levels():
    y_true = np.full((1, 1), 5.0)
    quantile_preds = np.full((1, 1, 9), 5.0)
    with pytest.raises(ValueError, match="tail_levels"):
        twcrps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS, tail_levels=(0.95,))


def test_crps_raises_on_quantile_shape_mismatch():
    y_true = np.full((1, 1), 5.0)
    quantile_preds = np.full((1, 1, 12), 5.0)
    with pytest.raises(ValueError, match="quantile_preds"):
        crps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
