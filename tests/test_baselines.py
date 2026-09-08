import numpy as np
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.climatology import Climatology
from zeropp.models.raw import Persistence, RawEnsemble

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_climatology_is_a_postprocessor():
    assert issubclass(Climatology, Postprocessor)


def test_climatology_fit_then_predict_shape():
    model = Climatology(quantile_levels=QUANTILE_LEVELS).fit(train=np.arange(1, 101, dtype=float))
    preds = model.predict_quantiles({"n_samples": 5, "n_leads": 3})
    assert preds.shape == (5, 3, 9)


def test_climatology_predict_matches_empirical_quantiles():
    train = np.arange(1, 101, dtype=float)  # 1..100
    model = Climatology(quantile_levels=QUANTILE_LEVELS).fit(train=train)
    preds = model.predict_quantiles({"n_samples": 1, "n_leads": 1})
    expected_median = np.quantile(train, 0.5)
    median_idx = QUANTILE_LEVELS.index(0.5)
    assert preds[0, 0, median_idx] == pytest.approx(expected_median)


def test_climatology_broadcasts_same_quantiles_to_every_sample_and_lead():
    train = np.arange(1, 101, dtype=float)
    model = Climatology(quantile_levels=QUANTILE_LEVELS).fit(train=train)
    preds = model.predict_quantiles({"n_samples": 4, "n_leads": 2})
    assert np.allclose(preds[0, 0], preds[3, 1])


def test_raw_ensemble_is_passthrough():
    model = RawEnsemble(quantile_levels=QUANTILE_LEVELS).fit(train=None)
    ensemble_quantiles = np.arange(54, dtype=float).reshape(3, 2, 9)
    preds = model.predict_quantiles({"ensemble_quantiles": ensemble_quantiles})
    assert np.array_equal(preds, ensemble_quantiles)


def test_climatology_predict_before_fit_raises():
    model = Climatology(quantile_levels=QUANTILE_LEVELS)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles({"n_samples": 1, "n_leads": 1})


def test_persistence_replicates_last_value_across_quantiles():
    model = Persistence(quantile_levels=QUANTILE_LEVELS).fit(train=None)
    last_observed = np.array([5.0, 7.0])
    preds = model.predict_quantiles({"last_observed": last_observed, "n_leads": 3})
    assert preds.shape == (2, 3, 9)
    assert np.all(preds[0] == 5.0)
    assert np.all(preds[1] == 7.0)
