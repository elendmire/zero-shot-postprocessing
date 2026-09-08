"""T2 (Introduction review, 2026-09-07, B9/item 6): permanent regression test
-- the mean-only code path in TimesFM3.predict_quantiles must reproduce the
already-persisted TimesFM-3 numbers in results/phase2_comparison_raw.parquet
BIT-FOR-BIT (or within a tight numerical tolerance if the model has any
nondeterminism) after the covariate-stacking change made for the two-
covariate check (docs/pending_ssh_runs.md item 6). If it does not, the
wrapper change broke the paper's main results, and that must be fixed before
anything else in item 6 is trusted.

Requires the real `timesfm` package, real cached weights, and the real
NetCDF archives -- this test can only run on the server (over SSH), exactly
like tests/test_tsfm_timesfm.py's unit tests require the real package to
import at all. It reconstructs a handful of real (station, issue_time)
instances using scripts/02_run_tsfm.py's own instance-construction logic
(imported via importlib, the same mechanism tests/test_data_size_sweep.py
and several scripts/ files already use for numbered script filenames),
calls the CURRENT TimesFM3.predict_quantiles with mean-only input (no
spread), and compares against the persisted tsfm3 rows for those exact
(station_id, valid_time, step_hours) keys.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.models.tsfm_timesfm import TimesFM3

RAW_RESULTS_PATH = Path("results/phase2_comparison_raw.parquet")
N_CHECK_INSTANCES = 10


def _load_run_tsfm_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "02_run_tsfm.py"
    spec = importlib.util.spec_from_file_location("run_tsfm_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not RAW_RESULTS_PATH.exists(), reason="requires the real persisted phase2_comparison_raw.parquet")
def test_mean_only_path_reproduces_persisted_tsfm3_numbers():
    run_tsfm = _load_run_tsfm_module()
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
                        "no longer matches the persisted phase2_comparison_raw.parquet value -- "
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
