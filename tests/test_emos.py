import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.emos import EMOS, PersistenceAugmentedEMOS

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(42)
    n = 500
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    # true relationship: obs ~ N(ens_mean, ens_var) — EMOS should recover a~=0, b~=1
    t2m_obs = ens_mean + rng.normal(0, np.sqrt(ens_var))
    return pd.DataFrame({"station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs})


def test_emos_is_a_postprocessor():
    assert issubclass(EMOS, Postprocessor)


def test_emos_fit_returns_self(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS)
    fitted = model.fit(synthetic_train_df)
    assert fitted is model


def test_emos_fit_recovers_near_identity_mean_mapping(synthetic_train_df):
    # Since obs were generated as N(ens_mean, ens_var), a well-fit EMOS should
    # find b close to 1 and a close to 0 — a real check that the optimizer
    # is actually fitting something meaningful, not just returning defaults.
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    assert model._params is not None
    a, b, c, d = model._params
    assert abs(b - 1.0) < 0.3
    assert abs(a) < 2.0


def test_emos_predict_quantiles_shape(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((3, 2), 280.0), "ens_var": np.full((3, 2), 1.5)}
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 2, 9)


def test_emos_predict_quantiles_are_monotonic_increasing(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.5)}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_emos_predict_before_fit_raises():
    model = EMOS(quantile_levels=QUANTILE_LEVELS)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles({"ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1))})


def test_emos_median_close_to_ensemble_mean_when_b_near_one(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.5)}
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    assert abs(preds[0, 0, median_idx] - 280.0) < 3.0


# ============================================================================
# PersistenceAugmentedEMOS (Investigation 1: persistence-augmented EMOS)
# ============================================================================

@pytest.fixture
def synthetic_persistence_train_df():
    # True generating process: obs depends on BOTH ens_mean and last_obs, with
    # last_obs carrying real independent signal (weight 0.6) beyond ens_mean
    # (weight 0.4) -- a well-fit model should recover e close to 0.6 and b
    # close to 0.4, not silently ignore last_obs (e~=0).
    rng = np.random.default_rng(7)
    n = 3000
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    last_obs = ens_mean + rng.normal(0, 2.0, size=n)  # correlated with, not identical to, ens_mean
    true_b, true_e = 0.4, 0.6
    signal = true_b * ens_mean + true_e * last_obs
    # Centered so the intercept a stays near 0 despite b+e != 1.
    t2m_obs = signal - (true_b + true_e - 1.0) * 280.0 + rng.normal(0, np.sqrt(ens_var))
    return pd.DataFrame({
        "station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var,
        "last_obs": last_obs, "t2m_obs": t2m_obs,
    })


def test_persistence_augmented_emos_is_a_postprocessor():
    assert issubclass(PersistenceAugmentedEMOS, Postprocessor)


def test_persistence_augmented_emos_fit_returns_self(synthetic_persistence_train_df):
    model = PersistenceAugmentedEMOS(quantile_levels=QUANTILE_LEVELS)
    fitted = model.fit(synthetic_persistence_train_df)
    assert fitted is model


def test_persistence_augmented_emos_recovers_true_last_obs_weight(synthetic_persistence_train_df):
    model = PersistenceAugmentedEMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_persistence_train_df)
    assert model._params is not None
    a, b, c, d, e = model._params
    assert abs(e - 0.6) < 0.15
    assert abs(b - 0.4) < 0.15


def test_persistence_augmented_emos_predict_quantiles_shape(synthetic_persistence_train_df):
    model = PersistenceAugmentedEMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_persistence_train_df)
    X = {"ens_mean": np.full((3, 2), 280.0), "ens_var": np.full((3, 2), 1.5), "last_obs": np.full((3, 2), 280.0)}
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 2, 9)


def test_persistence_augmented_emos_predict_quantiles_are_monotonic_increasing(synthetic_persistence_train_df):
    model = PersistenceAugmentedEMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_persistence_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.5), "last_obs": np.full((2, 1), 280.0)}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_persistence_augmented_emos_predict_before_fit_raises():
    model = PersistenceAugmentedEMOS(quantile_levels=QUANTILE_LEVELS)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles({
            "ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1)), "last_obs": np.zeros((1, 1)),
        })


def test_persistence_augmented_emos_ignoring_last_obs_recovers_near_plain_emos(synthetic_train_df):
    # If last_obs carries NO real signal beyond noise (here: a column of pure
    # random noise, independent of obs), fitting should recover e near 0 --
    # i.e. it doesn't invent a spurious persistence effect where none exists.
    rng = np.random.default_rng(99)
    df = synthetic_train_df.copy()
    df["last_obs"] = rng.normal(280.0, 5.0, size=len(df))
    model = PersistenceAugmentedEMOS(quantile_levels=QUANTILE_LEVELS).fit(df)
    a, b, c, d, e = model._params
    assert abs(e) < 0.2
