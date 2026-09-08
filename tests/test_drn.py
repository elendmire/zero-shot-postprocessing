import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.drn import DRN

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(0)
    n_per_station = 200
    station_ids = [1, 2, 3]
    station_bias = {1: -2.0, 2: 0.0, 3: 3.0}  # each station has a real, distinct bias

    rows = []
    for sid in station_ids:
        ens_mean = rng.normal(280.0, 5.0, n_per_station)
        ens_var = rng.uniform(0.5, 2.0, n_per_station)
        t2m_obs = ens_mean + station_bias[sid] + rng.normal(0, np.sqrt(ens_var))
        rows.append(pd.DataFrame({"station_id": sid, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs}))
    return pd.concat(rows, ignore_index=True)


def test_drn_is_a_postprocessor():
    assert issubclass(DRN, Postprocessor)


def test_drn_fit_returns_self(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5)
    assert model.fit(synthetic_train_df) is model


def test_drn_predict_quantiles_shape(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 1, 9)


def test_drn_predict_before_fit_raises():
    model = DRN(quantile_levels=QUANTILE_LEVELS)
    X = {"ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1)), "station_id": np.array([1])}
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles(X)


def test_drn_learns_real_per_station_bias(synthetic_train_df):
    # Behavioral test: after training, DRN's median prediction for the same ens_mean
    # should differ meaningfully across stations, tracking the real, distinct biases
    # the synthetic data was generated with (station 3's median should be well above
    # station 1's, since the true generating bias differs by 5.0).
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=200, lr=0.05, seed=0).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    medians = preds[:, 0, median_idx]
    assert medians[2] - medians[0] > 2.0  # station 3 vs station 1, true gap is 5.0


def test_drn_median_prediction_is_at_the_real_target_scale(synthetic_train_df):
    # Regression test for a real bug that shipped once (found only by running the
    # real data-size sweep on real EUPPBench data, not by any unit test): the
    # network's raw mu output must be rescaled from standardized-target space back
    # to real Kelvin units (mu = mu_std * obs_scale + obs_loc), not left as a small
    # near-zero standardized value. The relative-bias test above only checks the
    # *gap* between two stations' medians, which a broken (near-zero, unrescaled)
    # mu can still produce via the per-station embedding -- it never checks that
    # medians are anywhere near the real target scale. This test checks the
    # absolute scale directly, at this class's actual PRODUCTION DEFAULTS
    # (n_epochs=50, lr=1e-2 -- not the boosted n_epochs=200/lr=0.05 used by the
    # relative-bias test above), since that's the regime the real bug bit in:
    # at ens_mean=280.0, each station's predicted median should track the real
    # generating process (ens_mean + station_bias[sid]), not sit near 0 K.
    model = DRN(quantile_levels=QUANTILE_LEVELS, seed=0).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    medians = preds[:, 0, median_idx]
    station_bias = {1: -2.0, 2: 0.0, 3: 3.0}
    for i, sid in enumerate([1, 2, 3]):
        expected = 280.0 + station_bias[sid]
        assert abs(medians[i] - expected) < 6.0, (
            f"station {sid} median={medians[i]:.2f} is not within 6K of the real "
            f"target scale ({expected:.2f}K) -- this is the failure signature of the "
            "un-rescaled-mu bug (predictions stuck near a standardized-space value "
            "like 0 instead of real ~280K temperatures)"
        )


def test_drn_unstandardize_output_rescales_mu_and_sigma_by_obs_stats():
    # Direct, deterministic regression test on DRN._unstandardize_output itself --
    # the exact rescale step at the heart of the real bug this task found (network
    # output interpreted as an absolute Kelvin value instead of a standardized-
    # target value rescaled by fit()-time obs mean/std). An end-to-end,
    # train-then-predict comparison test (tried first, see git history) turned out
    # to be UNRELIABLE for this: gradient descent on log_sigma2 can partially
    # self-compensate for a missing obs_scale factor within a shared epoch budget
    # (confirmed empirically -- a hand-mutated build missing "* self._obs_scale"
    # on the sigma line, and a separate mutation missing it on the mu line, both
    # still landed within this file's median-scale tolerance above, since the
    # per-station corrections needed for this fixture are small enough to be
    # reachable by gradient descent even without the rescale). Calling the
    # rescale function directly with known inputs sidesteps that confound
    # entirely and pins down the exact formula unambiguously.
    import torch

    model = DRN(quantile_levels=QUANTILE_LEVELS)
    model._obs_loc = 280.0
    model._obs_scale = 6.0
    mu_std = torch.tensor([0.0, 1.0, -2.0])
    log_sigma2_std = torch.tensor([0.0, 2.0, -1.0])

    mu, sigma = model._unstandardize_output(mu_std, log_sigma2_std)

    expected_mu = mu_std * 6.0 + 280.0
    expected_sigma = torch.exp(0.5 * log_sigma2_std) * 6.0
    assert torch.allclose(mu, expected_mu), f"mu={mu.tolist()}, expected={expected_mu.tolist()}"
    assert torch.allclose(sigma, expected_sigma), f"sigma={sigma.tolist()}, expected={expected_sigma.tolist()}"


def test_drn_predict_quantiles_are_monotonic_increasing(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.0), "station_id": np.array([1, 2])}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_drn_unseen_station_at_predict_time_does_not_crash(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.0), "station_id": np.array([999])}
    preds = model.predict_quantiles(X)  # station 999 was never in training data
    assert preds.shape == (1, 1, 9)
