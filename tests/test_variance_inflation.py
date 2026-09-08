import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.variance_inflation import VarianceInflationBaseline

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def synthetic_train_df():
    # True generating process: obs ~ N(ens_mean, (TRUE_MULTIPLIER * sqrt(ens_var))^2)
    # -- the reported ens_var systematically under-states the true error variance by
    # a factor of TRUE_MULTIPLIER^2, so a well-fit multiplier should recover
    # TRUE_MULTIPLIER, not 1.0.
    rng = np.random.default_rng(0)
    n = 5000
    true_multiplier = 2.0
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    t2m_obs = ens_mean + rng.normal(0, true_multiplier * np.sqrt(ens_var))
    return pd.DataFrame({"station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs})


def test_variance_inflation_is_a_postprocessor():
    assert issubclass(VarianceInflationBaseline, Postprocessor)


def test_fit_returns_self(synthetic_train_df):
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS)
    fitted = model.fit(synthetic_train_df)
    assert fitted is model


def test_fit_recovers_true_multiplier(synthetic_train_df):
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    assert model.multiplier == pytest.approx(2.0, rel=0.1)


def test_fit_with_correctly_calibrated_ensemble_recovers_multiplier_near_one():
    rng = np.random.default_rng(1)
    n = 5000
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    t2m_obs = ens_mean + rng.normal(0, np.sqrt(ens_var))  # multiplier truly 1.0
    train = pd.DataFrame({"station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs})
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS).fit(train)
    assert model.multiplier == pytest.approx(1.0, rel=0.1)


def test_predict_quantiles_shape(synthetic_train_df):
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((3, 2), 280.0), "ens_var": np.full((3, 2), 1.5)}
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 2, 9)


def test_predict_quantiles_are_monotonic_increasing(synthetic_train_df):
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.5)}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_predict_quantiles_median_equals_ens_mean(synthetic_train_df):
    # No mean adjustment (unlike EMOS): the median prediction (q0.5) must equal
    # ens_mean exactly, since norm.ppf(0.5) == 0.
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.array([[280.0], [300.0]]), "ens_var": np.array([[1.5], [2.0]])}
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    assert preds[0, 0, median_idx] == pytest.approx(280.0)
    assert preds[1, 0, median_idx] == pytest.approx(300.0)


def test_predict_before_fit_or_fixed_multiplier_raises():
    model = VarianceInflationBaseline(quantile_levels=QUANTILE_LEVELS)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles({"ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1))})


def test_from_fixed_multiplier_needs_no_fit_call():
    model = VarianceInflationBaseline.from_fixed_multiplier(1.5, quantile_levels=QUANTILE_LEVELS)
    assert model.multiplier == pytest.approx(1.5)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.0)}
    preds = model.predict_quantiles(X)
    assert preds.shape == (2, 1, 9)


def test_fit_on_a_fixed_multiplier_instance_is_a_noop(synthetic_train_df):
    # Fix-round-1 finding 4: from_fixed_multiplier(1.5).fit(train) must NOT
    # overwrite the fixed multiplier with a data-derived one -- synthetic_train_df's
    # true generating multiplier is 2.0, so a silent overwrite would change 1.5 ->
    # ~2.0 and this assertion would fail.
    model = VarianceInflationBaseline.from_fixed_multiplier(1.5, quantile_levels=QUANTILE_LEVELS)
    fitted = model.fit(synthetic_train_df)
    assert fitted.multiplier == pytest.approx(1.5)
    assert fitted is model


def test_from_coverage_target_recovers_a_multiplier_that_hits_the_target_on_train():
    # Fix-round-1 Blocking Fix 3: synthetic archive with a KNOWN true multiplier (2.0,
    # same generating process as synthetic_train_df above) -- the coverage-matched
    # multiplier for the TRUE empirical coverage@[0.1,0.9] of this exact archive must
    # recover a lambda close to 2.0 (not exactly, since target_coverage is read back
    # from a finite empirical sample, not the population value).
    rng = np.random.default_rng(2)
    n = 20000
    true_multiplier = 2.0
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    t2m_obs = ens_mean + rng.normal(0, true_multiplier * np.sqrt(ens_var))
    train = pd.DataFrame({"station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs})

    # The archive's own true multiplier gives a known target coverage for [0.1, 0.9]
    # (the theoretical 80% central interval of a correctly-scaled Gaussian).
    target_coverage = 0.80
    model = VarianceInflationBaseline.from_coverage_target(
        target_coverage=target_coverage, train_df=train, quantile_levels=QUANTILE_LEVELS
    )
    assert model.multiplier == pytest.approx(true_multiplier, rel=0.1)

    # Re-checking coverage@[0.1,0.9] ON THE SAME TRAIN SET with the returned
    # multiplier must land very close to target_coverage (that is the definition of
    # the root-find) -- verified independently of VarianceInflationBaseline's own
    # internals, via predict_quantiles + a direct coverage count.
    preds = model.predict_quantiles({"ens_mean": ens_mean.reshape(-1, 1), "ens_var": ens_var.reshape(-1, 1)})
    lo_idx, hi_idx = QUANTILE_LEVELS.index(0.1), QUANTILE_LEVELS.index(0.9)
    achieved_coverage = float(np.mean(
        (t2m_obs >= preds[:, 0, lo_idx]) & (t2m_obs <= preds[:, 0, hi_idx])
    ))
    assert achieved_coverage == pytest.approx(target_coverage, abs=0.02)


def test_from_coverage_target_is_permanently_fixed_like_from_fixed_multiplier(synthetic_train_df):
    model = VarianceInflationBaseline.from_coverage_target(
        target_coverage=0.80, train_df=synthetic_train_df, quantile_levels=QUANTILE_LEVELS
    )
    lam_before = model.multiplier
    fitted = model.fit(synthetic_train_df)
    assert fitted is model
    assert fitted.multiplier == pytest.approx(lam_before)


def test_from_coverage_target_rejects_an_unreachable_target():
    # target_coverage=1.5 is outside [0, 1] and can never be bracketed by any lambda
    # in a sensible range -- must fail loudly (assertion), not silently return a
    # meaningless root.
    rng = np.random.default_rng(3)
    n = 500
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    t2m_obs = ens_mean + rng.normal(0, np.sqrt(ens_var))
    train = pd.DataFrame({"station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs})
    with pytest.raises(AssertionError, match="bracket"):
        VarianceInflationBaseline.from_coverage_target(
            target_coverage=1.5, train_df=train, quantile_levels=QUANTILE_LEVELS
        )


def test_larger_multiplier_gives_wider_intervals():
    narrow = VarianceInflationBaseline.from_fixed_multiplier(1.0, quantile_levels=QUANTILE_LEVELS)
    wide = VarianceInflationBaseline.from_fixed_multiplier(2.0, quantile_levels=QUANTILE_LEVELS)
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.0)}
    narrow_preds = narrow.predict_quantiles(X)
    wide_preds = wide.predict_quantiles(X)
    narrow_width = narrow_preds[0, 0, -1] - narrow_preds[0, 0, 0]
    wide_width = wide_preds[0, 0, -1] - wide_preds[0, 0, 0]
    assert wide_width == pytest.approx(2.0 * narrow_width)
