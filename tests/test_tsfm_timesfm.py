from unittest.mock import MagicMock, patch

import numpy as np

from zeropp.models.base import Postprocessor
from zeropp.models.tsfm_timesfm import TimesFM3

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_timesfm3_is_a_postprocessor():
    assert issubclass(TimesFM3, Postprocessor)


def test_timesfm3_fit_is_a_noop_returning_self():
    model = TimesFM3(quantile_levels=QUANTILE_LEVELS)
    assert model.fit(train=None) is model


def test_timesfm3_predict_quantiles_calls_predict_with_covariates():
    fake_output = MagicMock()
    fake_output.forecast = np.zeros(4)
    fake_output.quantiles = np.zeros((4, 9))

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_output

    with patch("zeropp.models.tsfm_timesfm.timesfm.TimesFM3Forecaster.from_pretrained", return_value=fake_model):
        model = TimesFM3(quantile_levels=QUANTILE_LEVELS, weights_path="/fake/path")
        X = {
            "context": [np.ones(10)],
            "past_future_ens_mean": [np.ones(14)],
            "past_future_ens_spread": [np.ones(14)],
            "horizon": 4,
        }
        preds = model.predict_quantiles(X)

    assert preds.shape == (1, 4, 9)
    call_kwargs = fake_model.predict.call_args.kwargs
    assert "past_future_covariates" in call_kwargs
    assert call_kwargs["horizon"] == 4
    assert call_kwargs["return_quantiles"] is True


def test_timesfm3_predict_quantiles_handles_multiple_stations():
    fake_output = MagicMock()
    fake_output.forecast = np.zeros(3)
    fake_output.quantiles = np.zeros((3, 9))

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_output

    with patch("zeropp.models.tsfm_timesfm.timesfm.TimesFM3Forecaster.from_pretrained", return_value=fake_model):
        model = TimesFM3(quantile_levels=QUANTILE_LEVELS, weights_path="/fake/path")
        X = {
            "context": [np.ones(10), np.ones(10)],
            "past_future_ens_mean": [np.ones(13), np.ones(13)],
            "past_future_ens_spread": [np.ones(13), np.ones(13)],
            "horizon": 3,
        }
        preds = model.predict_quantiles(X)

    assert preds.shape == (2, 3, 9)
    assert fake_model.predict.call_count == 2


def test_timesfm3_predict_quantiles_stacks_mean_and_spread_when_both_provided():
    # B4 (paper review, 2026-09-07): the model previously discarded
    # past_future_ens_spread entirely. It now attempts to pass BOTH
    # covariates by stacking them into a (2, T) array (channels on axis 0,
    # time on axis -1 -- the shape convention verified directly from the
    # real installed timesfm3 package's model.py docstring, B9) on the
    # single past_future_covariates parameter. This test locks down what
    # this project's own code does with the array shape; whether the real
    # server-side model produces sensible output from it is
    # docs/pending_ssh_runs.md item 6's smoke test, not this test.
    fake_output = MagicMock()
    fake_output.forecast = np.zeros(4)
    fake_output.quantiles = np.zeros((4, 9))

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_output

    with patch("zeropp.models.tsfm_timesfm.timesfm.TimesFM3Forecaster.from_pretrained", return_value=fake_model):
        model = TimesFM3(quantile_levels=QUANTILE_LEVELS, weights_path="/fake/path")
        mean = np.arange(14, dtype=float)
        spread = np.arange(14, dtype=float) + 100.0
        X = {
            "context": [np.ones(10)],
            "past_future_ens_mean": [mean],
            "past_future_ens_spread": [spread],
            "horizon": 4,
        }
        model.predict_quantiles(X)

    call_kwargs = fake_model.predict.call_args.kwargs
    stacked = call_kwargs["past_future_covariates"]
    assert stacked.shape == (2, 14)
    np.testing.assert_array_equal(stacked[0], mean)
    np.testing.assert_array_equal(stacked[1], spread)


def test_timesfm3_predict_quantiles_uses_mean_only_when_spread_absent():
    fake_output = MagicMock()
    fake_output.forecast = np.zeros(4)
    fake_output.quantiles = np.zeros((4, 9))

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_output

    with patch("zeropp.models.tsfm_timesfm.timesfm.TimesFM3Forecaster.from_pretrained", return_value=fake_model):
        model = TimesFM3(quantile_levels=QUANTILE_LEVELS, weights_path="/fake/path")
        mean = np.ones(14)
        X = {
            "context": [np.ones(10)],
            "past_future_ens_mean": [mean],
            "horizon": 4,
        }
        model.predict_quantiles(X)

    call_kwargs = fake_model.predict.call_args.kwargs
    passed = call_kwargs["past_future_covariates"]
    np.testing.assert_array_equal(np.asarray(passed), mean)
