import warnings

import pandas as pd
import xarray as xr

# First valid time of the test period. Reforecast cases whose forecasts verify on or after this date
# (the most recent analogue year of the 2018 labels) are dropped from the training data.
TEST_PERIOD_START = pd.Timestamp("2017-01-01")


def _non_overlapping_cases(fcs: xr.Dataset) -> pd.DataFrame:
    """(time, year, case_date) of the reforecast cases that verify before TEST_PERIOD_START.

    case_date is the initialization date of the case and orders cases in time across analogue years."""
    last_valid = fcs["valid_time"].max(dim="step").to_dataframe(name="last_valid").reset_index()
    first_valid = fcs["valid_time"].min(dim="step").to_dataframe(name="case_date").reset_index()
    cases = last_valid.merge(first_valid[["time", "year", "case_date"]], on=["time", "year"])
    keep = cases[pd.to_datetime(cases["last_valid"]) < TEST_PERIOD_START]
    keep = keep.assign(case_date=pd.to_datetime(keep["case_date"]).dt.normalize())
    return keep[["time", "year", "case_date"]]


def build_train_ensemble_stats_with_ids(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    """Ensemble mean and variance of the reforecasts with the observation, one row per station, case and lead.

    Keeps time_idx, year_idx and case_date so that training cases can be subsampled."""
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    # sample variance (ddof=1), as for the test-set ensembles
    ens_var = fcs["t2m"].var(dim="number", ddof=1)

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df.merge(_non_overlapping_cases(fcs), on=["time", "year"], how="inner")
    df = df.rename(columns={"time": "time_idx", "year": "year_idx"})
    df = df[["station_id", "time_idx", "year_idx", "case_date", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_train_ensemble_stats(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    full = build_train_ensemble_stats_with_ids(reforecast_path, obs_path)
    return full[["station_id", "ens_mean", "ens_var", "t2m_obs"]]


def build_train_ensemble_stats_with_lead(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    """As build_train_ensemble_stats_with_ids, with the lead time kept as column step_hours."""
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    ens_var = fcs["t2m"].var(dim="number", ddof=1)

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df.merge(_non_overlapping_cases(fcs), on=["time", "year"], how="inner")
    df = df.rename(columns={"time": "time_idx", "year": "year_idx", "step": "step_hours"})
    df = df[["station_id", "time_idx", "year_idx", "case_date", "step_hours", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_test_long_table(forecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(forecast_path)
    try:
        obs = xr.open_dataset(obs_path)
    except OverflowError:
        # time decoding of the observation file can overflow; retry, then decode manually
        try:
            obs = xr.open_dataset(obs_path)
        except OverflowError:
            obs = xr.open_dataset(obs_path, decode_times=False)
            if not pd.api.types.is_datetime64_any_dtype(obs["time"].values):
                raise RuntimeError("observation time coordinate could not be decoded")

    fcs_df = fcs["t2m"].to_dataframe(name="t2m_forecast").reset_index()
    # the EUPPBench files carry valid_time as a coordinate; test fixtures without it use time
    has_native_valid_time = "valid_time" in fcs_df.columns
    fcs_keep = ["station_id", "time", "step", "number", "t2m_forecast"]
    if has_native_valid_time:
        fcs_keep.append("valid_time")
    fcs_df = fcs_df[fcs_keep]

    obs_df = obs["t2m"].to_dataframe(name="t2m_obs").reset_index()
    obs_df = obs_df[["station_id", "time", "step", "t2m_obs"]]

    merged = fcs_df.merge(obs_df, on=["station_id", "time", "step"], how="left")
    if not has_native_valid_time:
        warnings.warn("no 'valid_time' coordinate found; using 'time' as valid_time")
        merged = merged.rename(columns={"time": "valid_time"})
    else:
        merged = merged.drop(columns=["time"])
    merged = merged.rename(columns={"step": "step_hours", "number": "member"})
    merged = merged.dropna(subset=["t2m_forecast", "t2m_obs"])
    merged = merged[["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]]
    merged = merged.sort_values(["station_id", "valid_time"]).reset_index(drop=True)
    return merged
