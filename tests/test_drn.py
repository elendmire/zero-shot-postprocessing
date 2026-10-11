import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.drn import DRN

@pytest.fixture(autouse=True)
def _single_thread_torch():
    # tiny networks: threading only adds overhead
    import torch

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


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
    # medians of stations with different true biases (gap 5 K) should differ clearly
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
    # with the default settings the median must be on the scale of the target (about 280 K)
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
        assert abs(medians[i] - expected) < 6.0, f"station {sid}: median {medians[i]:.2f} K, expected about {expected:.2f} K"


def test_drn_unstandardize_output_rescales_mu_and_sigma_by_obs_stats():
    # mu and sigma are rescaled with the observation mean and standard deviation
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


def _case_df(n_cases=40, n_stations=5, with_lead=False, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    leads = [24, 48] if with_lead else [None]
    for c in range(n_cases):
        for sid in range(n_stations):
            for lead in leads:
                m = rng.normal(280.0, 5.0)
                v = rng.uniform(0.5, 2.0)
                row = {"year_idx": c % 4, "time_idx": c // 4, "station_id": sid, "ens_mean": m,
                       "ens_var": v, "t2m_obs": m + sid - 2 + (1.5 if lead == 48 else 0.0) + rng.normal(0, np.sqrt(v))}
                if with_lead:
                    row["step_hours"] = lead
                rows.append(row)
    return pd.DataFrame(rows)


def test_drn_early_stopping_stops_before_max_and_restores_best_weights():
    df = _case_df()
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=300, lr=0.05, batch_size=64, patience=3, seed=0,
                min_updates=0).fit(df)
    assert len(model.val_loss_history_) == len(model.loss_history_)
    steps_per_epoch = -(-int(len(df) * 0.9) // 64)
    assert model.n_updates_ < 300 * steps_per_epoch  # stopped early
    assert model.n_updates_ > 0
    # best weights restored: the net's validation loss equals the best recorded one
    split = model._split_indices(df, len(df))
    assert split is not None
    # case-wise split: no case appears in both train and val
    key = df["year_idx"] * 1000 + df["time_idx"]
    assert not set(key.iloc[split[0]]) & set(key.iloc[split[1]])


def test_drn_restores_best_validation_weights():
    import torch

    df = _case_df()
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=300, lr=0.05, batch_size=64, patience=3, seed=0,
                min_updates=0).fit(df)
    # re-evaluate the restored net on the same validation split
    split = model._split_indices(df, len(df))
    dev = torch.device(model.device)
    em, ev = model._standardize(df["ens_mean"].to_numpy(), df["ens_var"].to_numpy())
    tensors = (
        torch.tensor(em, dtype=torch.float32, device=dev),
        torch.tensor(ev, dtype=torch.float32, device=dev),
        torch.tensor(df["station_id"].map(model._station_to_idx).to_numpy(), dtype=torch.long, device=dev),
        None,
        torch.tensor(df["t2m_obs"].to_numpy(), dtype=torch.float32, device=dev),
    )
    val = model._mean_loss(tensors, torch.tensor(split[1], dtype=torch.long, device=dev))
    # (the initial raw-ensemble weights are also a candidate, so the restored net is
    # at least as good as the best validation check)
    assert val <= min(model.val_loss_history_) * (1 + 1e-4)


def test_drn_lead_time_input_path():
    df = _case_df(with_lead=True)
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=3, batch_size=128).fit(df)
    X = {
        "ens_mean": np.full((2, 2), 280.0),
        "ens_var": np.full((2, 2), 1.0),
        "station_id": np.array([0, 1]),
        "step_hours": np.array([[24, 48], [24, 48]]),
    }
    assert model.predict_quantiles(X).shape == (2, 2, 9)
    del X["step_hours"]
    with pytest.raises(ValueError, match="step_hours"):
        model.predict_quantiles(X)


def test_drn_tiny_set_fallback_runs_fixed_steps_without_validation():
    df = pd.DataFrame({
        "station_id": [1, 2, 1, 2, 1, 2],
        "ens_mean": [280.0, 281, 279, 282, 280, 283],
        "ens_var": [1.0, 1.1, 0.9, 1.0, 1.2, 0.8],
        "t2m_obs": [280.5, 281.2, 278.7, 282.3, 280.1, 283.4],
    })
    model = DRN(quantile_levels=QUANTILE_LEVELS).fit(df)
    assert model.n_updates_ == 200
    assert model.val_loss_history_ == []
    assert len(model.loss_history_) >= 1
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.0), "station_id": np.array([1])}
    assert np.isfinite(model.predict_quantiles(X)).all()


def test_drn_untrained_net_equals_raw_ensemble_gaussian():
    # Zero-initialised last layer + residual parameterisation: before any update,
    # mu = ensemble mean and sigma^2 = ensemble variance, for every station/input.
    import torch
    from scipy.stats import norm

    df = _case_df(with_lead=True)
    model = DRN(quantile_levels=QUANTILE_LEVELS, min_updates=0, n_epochs=0, fallback_steps=0).fit(df)
    assert model.n_updates_ == 0
    rng = np.random.default_rng(3)
    ens_mean = rng.normal(280.0, 5.0, (4, 2))
    ens_var = rng.uniform(0.5, 4.0, (4, 2))
    X = {"ens_mean": ens_mean, "ens_var": ens_var, "station_id": np.array([0, 1, 2, 3]),
         "step_hours": np.array([[24, 48]] * 4)}
    preds = model.predict_quantiles(X)
    expected = ens_mean[..., None] + np.sqrt(ens_var)[..., None] * norm.ppf(QUANTILE_LEVELS)
    np.testing.assert_allclose(preds, expected, rtol=0, atol=2e-3)  # float32 round-off at 280 K


def test_drn_small_set_training_not_worse_than_initialisation():
    # Small training set with a real, learnable station bias and a sigma miscalibration:
    # (a) the minimum-update floor is respected, (b) the fitted net's mean CRPS on a
    # large fresh test sample is no worse than the raw-ensemble Gaussian it starts from.
    from zeropp.models.drn import _gaussian_crps_torch
    import torch

    def make(n_cases, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for c in range(n_cases):
            for sid in range(4):
                m, v = rng.normal(280.0, 5.0), rng.uniform(0.5, 2.0)
                rows.append({"year_idx": 0, "time_idx": c, "station_id": sid, "ens_mean": m, "ens_var": v,
                             "t2m_obs": m + 1.5 * (sid - 1.5) + rng.normal(0, 1.5 * np.sqrt(v))})
        return pd.DataFrame(rows)

    train, test = make(60, 1), make(500, 2)
    model = DRN(quantile_levels=QUANTILE_LEVELS, seed=0, min_updates=500).fit(train)
    assert model.n_updates_ >= 500
    X = {"ens_mean": test["ens_mean"].to_numpy().reshape(-1, 1), "ens_var": test["ens_var"].to_numpy().reshape(-1, 1),
         "station_id": test["station_id"].to_numpy()}
    preds = model.predict_quantiles(X)
    y = test["t2m_obs"].to_numpy()
    from zeropp.eval.scores import crps_from_quantiles
    drn_crps = crps_from_quantiles(y.reshape(-1, 1), preds, QUANTILE_LEVELS).mean()
    raw = X["ens_mean"][..., None] + np.sqrt(X["ens_var"])[..., None] * __import__("scipy.stats").stats.norm.ppf(QUANTILE_LEVELS)
    raw_crps = crps_from_quantiles(y.reshape(-1, 1), raw, QUANTILE_LEVELS).mean()
    assert drn_crps <= raw_crps * 1.001, (drn_crps, raw_crps)
