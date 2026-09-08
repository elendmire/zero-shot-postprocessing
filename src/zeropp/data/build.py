import warnings

import pandas as pd
import xarray as xr


def build_train_ensemble_stats_with_ids(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    """Same as build_train_ensemble_stats but also keeps time_idx/year_idx so
    callers (e.g. scripts/03_data_size_sweep.py) can subsample by issue-date
    pair for the training-data-size sweep."""
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    # ddof=1 (sample variance) to match the test-side ensemble variance, which
    # is computed with pandas' .var() (default ddof=1) in scripts/01_run_baselines.py
    # and scripts/02_run_tsfm.py. xarray's .var() defaults to ddof=0 (population
    # variance), which would otherwise train EMOS on a systematically different
    # variance definition than the one it's evaluated on at test time.
    ens_var = fcs["t2m"].var(dim="number", ddof=1)

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df.rename(columns={"time": "time_idx", "year": "year_idx"})
    df = df[["station_id", "time_idx", "year_idx", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_train_ensemble_stats(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    full = build_train_ensemble_stats_with_ids(reforecast_path, obs_path)
    return full[["station_id", "ens_mean", "ens_var", "t2m_obs"]]


def build_train_ensemble_stats_with_lead(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    """Same underlying rows as build_train_ensemble_stats_with_ids, but keeps the
    reforecast archive's 'step' dimension (renamed 'step_hours', matching the test-side
    naming in build_test_long_table) INSTEAD of dropping it via column selection.

    Task 6 E4 confirmed (by direct code inspection, corroborated by
    tests/test_build.py's existing fixture: 2 stations x 2 time_idx x 1 year x 2 step =
    8 combinations, 7 after one NaN drop) that build_train_ensemble_stats_with_ids's
    rows already differ by lead time -- the function never averages over 'step' (only
    .mean(dim="number") averages over ensemble members); 'step' is merely dropped from
    the OUTPUT COLUMNS by that function's final column subselection. This function is
    the with-step-retained sibling needed for Task 6 E4's optional lead-time-grouped
    EMOS extension and Task 6 E5b's lead-time-bucketed analysis: fitting one EMOS per
    lead-time bucket requires a training dataframe that can actually be split by lead
    time, which build_train_ensemble_stats_with_ids's public four/six-column contract
    cannot provide."""
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    ens_var = fcs["t2m"].var(dim="number", ddof=1)

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df.rename(columns={"time": "time_idx", "year": "year_idx", "step": "step_hours"})
    df = df[["station_id", "time_idx", "year_idx", "step_hours", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_test_long_table(forecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(forecast_path)
    try:
        obs = xr.open_dataset(obs_path)
    except OverflowError:
        # Known intermittent bug on the forecasts-observations file: unmasked
        # fill-value sentinels in the time coordinate can overflow the int64
        # conversion during decode_times. Retry once, then fall back to
        # decode_times=False (same proven fix used for the reforecast obs file).
        try:
            obs = xr.open_dataset(obs_path)
        except OverflowError:
            obs = xr.open_dataset(obs_path, decode_times=False)
            if not pd.api.types.is_datetime64_any_dtype(obs["time"].values):
                raise RuntimeError(
                    "obs time coordinate did not decode to datetime64 even after "
                    "the decode_times=False fallback; merging on raw integer time "
                    "offsets against the forecast file's real datetimes would "
                    "silently produce zero matches. Needs manual time decoding "
                    "for this file before build_test_long_table can proceed."
                )

    fcs_df = fcs["t2m"].to_dataframe(name="t2m_forecast").reset_index()
    # Real EUPPBench forecast files carry a genuine (time, step) -> valid_time
    # auxiliary coordinate (init time + lead), which to_dataframe() surfaces as
    # its own column alongside raw "time". Prefer it when present since it is
    # the true verification timestamp; the synthetic test fixtures have no such
    # coordinate, so "time" itself is used as valid_time there.
    has_native_valid_time = "valid_time" in fcs_df.columns
    fcs_keep = ["station_id", "time", "step", "number", "t2m_forecast"]
    if has_native_valid_time:
        fcs_keep.append("valid_time")
    fcs_df = fcs_df[fcs_keep]

    obs_df = obs["t2m"].to_dataframe(name="t2m_obs").reset_index()
    obs_df = obs_df[["station_id", "time", "step", "t2m_obs"]]

    merged = fcs_df.merge(obs_df, on=["station_id", "time", "step"], how="left")
    if not has_native_valid_time:
        warnings.warn(
            "No native 'valid_time' coordinate found on this dataset — falling "
            "back to treating the raw 'time' column as valid_time. This is only "
            "correct when time already represents issue+lead (e.g. synthetic "
            "test fixtures with no step offset); for real forecast data missing "
            "this coordinate, valid_time will be wrong."
        )
        merged = merged.rename(columns={"time": "valid_time"})
    else:
        merged = merged.drop(columns=["time"])
    merged = merged.rename(columns={"step": "step_hours", "number": "member"})
    merged = merged.dropna(subset=["t2m_forecast", "t2m_obs"])
    merged = merged[["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]]
    merged = merged.sort_values(["station_id", "valid_time"]).reset_index(drop=True)
    return merged
