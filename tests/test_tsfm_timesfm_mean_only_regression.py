"""The mean-only path of TimesFM3.predict_quantiles must reproduce the TimesFM-3 rows of
results/predictions_main.parquet.

Needs the timesfm package, the cached weights and the NetCDF archives. For a few (station, issue time)
instances, built as in scripts/run_timesfm.py, the current prediction is compared with the stored quantiles.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.models.tsfm_timesfm import TimesFM3

RAW_RESULTS_PATH = Path("results/predictions_main.parquet")
N_CHECK_INSTANCES = 10


@pytest.mark.skipif(not RAW_RESULTS_PATH.exists(), reason="requires the real persisted predictions_main.parquet")
def test_mean_only_path_reproduces_persisted_tsfm3_numbers():
    import run_timesfm as run_tsfm

    quantile_levels = load_experiment_config().quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]

    persisted = pd.read_parquet(RAW_RESULTS_PATH)
    persisted_tsfm3 = persisted[persisted["method"] == "tsfm3"]

    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")
    model = TimesFM3(quantile_levels=quantile_levels, device="cpu")

    checked = 0
    for station_id, station_df in test_df.groupby("station_id"):
        if checked >= N_CHECK_INSTANCES:
            break
        station_df = station_df.sort_values("valid_time")
        obs_lookup = (
            station_df.drop_duplicates("valid_time")[["valid_time", "t2m_obs"]]
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        obs_times = obs_lookup["valid_time"].to_numpy()
        obs_vals = obs_lookup["t2m_obs"].to_numpy()
        freshest_ens_mean = run_tsfm._freshest_ens_mean_lookup(station_df)
        ens_vals = freshest_ens_mean.reindex(obs_lookup["valid_time"]).to_numpy()

        for issue_time, group in station_df.groupby("issue_time"):
            if checked >= N_CHECK_INSTANCES:
                break
            idx = np.searchsorted(obs_times, np.datetime64(issue_time), side="left")
            if idx < run_tsfm.CONTEXT_LENGTH:
                continue
            past_ens_covariate = ens_vals[idx - run_tsfm.CONTEXT_LENGTH: idx]
            if np.isnan(past_ens_covariate).any():
                continue
            context = obs_vals[idx - run_tsfm.CONTEXT_LENGTH: idx]

            by_lead = (
                group.groupby("step_hours")
                .agg(ens_mean=("t2m_forecast", "mean"), valid_time=("valid_time", "first"))
                .sort_index()
            )
            future_ens_means = by_lead["ens_mean"].to_numpy()
            horizon = len(future_ens_means)
            if horizon == 0:
                continue
            covariate = np.concatenate([past_ens_covariate, future_ens_means])

            pred = model.predict_quantiles({
                "context": [context],
                "past_future_ens_mean": [covariate],
                "horizon": horizon,
            })

            for i, step_hours in enumerate(by_lead.index):
                key_match = persisted_tsfm3[
                    (persisted_tsfm3["station_id"] == station_id)
                    & (persisted_tsfm3["valid_time"] == by_lead["valid_time"].iloc[i])
                    & (persisted_tsfm3["step_hours"] == step_hours)
                ]
                if key_match.empty:
                    continue  # not in the persisted matched set; nothing to compare
                persisted_q = key_match.iloc[0][quantile_cols].to_numpy(dtype=float)
                new_q = pred[0, i]
                np.testing.assert_allclose(
                    new_q, persisted_q, rtol=1e-5, atol=1e-6,
                    err_msg=(
                        f"Mean-only prediction for station={station_id}, "
                        f"valid_time={by_lead['valid_time'].iloc[i]}, step_hours={step_hours} "
                        "no longer matches the persisted predictions_main.parquet value -- "
                        "the covariate-stacking code change broke the paper's existing TimesFM-3 results."
                    ),
                )
                checked += 1
                if checked >= N_CHECK_INSTANCES:
                    break

    assert checked > 0, (
        "No persisted instances were matched to compare against -- this test found nothing "
        "to verify, which is itself a failure of the test's own setup, not a pass."
    )
