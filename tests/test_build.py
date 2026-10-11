import numpy as np
import pandas as pd
import pytest
import xarray as xr

from zeropp.data.build import build_test_long_table, build_train_ensemble_stats


@pytest.fixture
def synthetic_reforecast_files(tmp_path):
    # 2 stations, 2 time_idx, 1 year, 2 step, 3 members
    station_id = [100, 200]
    time_idx = [0, 1]
    year_idx = [0]
    step = [0.0, 6.0]
    number = [0, 1, 2]

    rng = np.random.default_rng(0)
    fcs_data = rng.normal(280.0, 2.0, size=(2, 2, 1, 2, 3, 1)).astype("float32")
    # Both cases verify in 2016, before the test period, so none is dropped.
    valid_time = np.array([[[np.datetime64("2016-03-01T00"), np.datetime64("2016-03-01T06")]],
                           [[np.datetime64("2016-03-04T00"), np.datetime64("2016-03-04T06")]]])
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step", "number", "surface"), fcs_data)},
        coords={
            "station_id": station_id, "time": time_idx, "year": year_idx,
            "step": step, "number": number, "surface": [0.0],
            "valid_time": (("time", "year", "step"), valid_time),
        },
    )
    fcs_path = tmp_path / "reforecasts.nc"
    fcs.to_netcdf(fcs_path)

    obs_data = np.full((2, 2, 1, 2), 280.0, dtype="float64")
    obs_data[0, 0, 0, 0] = np.nan  # one missing observation to verify it gets dropped
    obs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step"), obs_data)},
        coords={"station_id": station_id, "time": time_idx, "year": year_idx, "step": step},
    )
    obs_path = tmp_path / "reforecast_obs.nc"
    obs.to_netcdf(obs_path)

    return str(fcs_path), str(obs_path)


@pytest.fixture
def synthetic_forecast_files(tmp_path):
    station_id = [100, 200]
    time = pd.date_range("2017-01-01", periods=3, freq="12h")
    step = [0.0, 6.0]
    number = [0, 1]

    rng = np.random.default_rng(1)
    fcs_data = rng.normal(280.0, 2.0, size=(2, 3, 2, 2, 1)).astype("float32")
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "step", "number", "surface"), fcs_data)},
        coords={"station_id": station_id, "time": time, "step": step, "number": number, "surface": [0.0]},
    )
    fcs_path = tmp_path / "forecasts.nc"
    fcs.to_netcdf(fcs_path)

    obs_data = np.full((2, 3, 2), 280.0, dtype="float64")
    obs_data[1, 2, 1] = np.nan  # one missing observation
    obs = xr.Dataset(
        {"t2m": (("station_id", "time", "step"), obs_data)},
        coords={"station_id": station_id, "time": time, "step": step},
    )
    obs_path = tmp_path / "forecast_obs.nc"
    obs.to_netcdf(obs_path)

    return str(fcs_path), str(obs_path)


def test_build_train_ensemble_stats_shape_and_columns(synthetic_reforecast_files):
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "ens_mean", "ens_var", "t2m_obs"]
    # 2 stations x 2 time_idx x 1 year x 2 step = 8 combinations, minus 1 dropped NaN obs
    assert len(df) == 7


def test_build_train_ensemble_stats_mean_and_var_are_real_ensemble_stats(synthetic_reforecast_files):
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert (df["ens_var"] >= 0).all()
    assert df["ens_mean"].between(270, 290).all()  # sanity: within the synthetic generation range


def test_build_test_long_table_shape_and_columns(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]
    # 2 stations x 3 time x 2 step x 2 members = 24 rows, minus 2 members dropped for the 1 NaN obs row
    assert len(df) == 22


def test_build_test_long_table_sorted_by_station_then_time(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    for station_id, group in df.groupby("station_id"):
        assert group["valid_time"].is_monotonic_increasing


def test_build_test_long_table_valid_time_is_real_timestamp(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    assert pd.api.types.is_datetime64_any_dtype(df["valid_time"])


def test_build_train_ensemble_stats_with_ids_includes_time_and_year(synthetic_reforecast_files):
    from zeropp.data.build import build_train_ensemble_stats_with_ids
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats_with_ids(fcs_path, obs_path)
    assert set(["station_id", "time_idx", "year_idx", "ens_mean", "ens_var", "t2m_obs"]) <= set(df.columns)
    assert len(df) == 7  # same row count/NaN-dropping as build_train_ensemble_stats


def test_build_train_ensemble_stats_with_ids_time_and_year_values_not_swapped(synthetic_reforecast_files):
    # time_idx must take both values {0, 1} and year_idx only {0} (checks the column mapping)
    from zeropp.data.build import build_train_ensemble_stats_with_ids
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats_with_ids(fcs_path, obs_path)
    assert set(df["time_idx"].unique()) == {0, 1}
    assert set(df["year_idx"].unique()) == {0}


def test_build_train_ensemble_stats_still_returns_exactly_four_columns(synthetic_reforecast_files):
    # the public function keeps its original columns
    from zeropp.data.build import build_train_ensemble_stats
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "ens_mean", "ens_var", "t2m_obs"]


def test_build_train_ensemble_stats_with_lead_includes_step_hours(synthetic_reforecast_files):
    # the with-lead variant keeps step_hours
    from zeropp.data.build import build_train_ensemble_stats_with_lead
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats_with_lead(fcs_path, obs_path)
    assert list(df.columns) == [
        "station_id", "time_idx", "year_idx", "case_date", "step_hours", "ens_mean", "ens_var", "t2m_obs",
    ]
    # same rows as with_ids: 7 of the 8 combinations survive the NaN observation
    assert len(df) == 7


def test_build_train_ensemble_stats_with_lead_step_hours_are_not_averaged_away(synthetic_reforecast_files):
    # both leads (0 and 6 h) are present; the combination that lost its 0 h row keeps its 6 h row
    from zeropp.data.build import build_train_ensemble_stats_with_lead
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats_with_lead(fcs_path, obs_path)
    assert set(df["step_hours"].unique()) == {0.0, 6.0}
    dup_key_counts = df.groupby(["station_id", "time_idx", "year_idx"]).size()
    # 3 of the 4 (station, time_idx, year_idx) combinations have both leads
    assert (dup_key_counts == 2).sum() == 3
    assert (dup_key_counts == 1).sum() == 1


def test_build_train_ensemble_stats_with_lead_same_values_as_with_ids_after_dropping_step(synthetic_reforecast_files):
    # dropping step_hours and re-sorting reproduces the with_ids output
    from zeropp.data.build import build_train_ensemble_stats_with_ids, build_train_ensemble_stats_with_lead
    fcs_path, obs_path = synthetic_reforecast_files
    with_ids = build_train_ensemble_stats_with_ids(fcs_path, obs_path)
    with_lead = build_train_ensemble_stats_with_lead(fcs_path, obs_path)
    id_cols = ["station_id", "time_idx", "year_idx", "ens_mean", "ens_var", "t2m_obs"]
    lhs = with_ids[id_cols].sort_values(id_cols).reset_index(drop=True)
    rhs = with_lead[id_cols].sort_values(id_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(lhs, rhs)


def test_training_cases_verifying_in_test_period_are_dropped(tmp_path):
    """Reforecast cases verifying on or after 2017-01-01 overlap the test period and are dropped."""
    from zeropp.data.build import build_train_ensemble_stats_with_ids

    rng = np.random.default_rng(1)
    valid_time = np.array([[[np.datetime64("2016-12-20T00"), np.datetime64("2016-12-20T06")]],
                           [[np.datetime64("2016-12-31T18"), np.datetime64("2017-01-01T00")]]])
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step", "number"), rng.normal(280, 1, (1, 2, 1, 2, 3)))},
        coords={"station_id": [1], "time": [0, 1], "year": [20], "step": [0.0, 6.0], "number": [0, 1, 2],
                "valid_time": (("time", "year", "step"), valid_time)},
    )
    obs = xr.Dataset({"t2m": (("station_id", "time", "year", "step"), np.full((1, 2, 1, 2), 280.0))},
                     coords={"station_id": [1], "time": [0, 1], "year": [20], "step": [0.0, 6.0]})
    fcs.to_netcdf(tmp_path / "f.nc")
    obs.to_netcdf(tmp_path / "o.nc")
    df = build_train_ensemble_stats_with_ids(str(tmp_path / "f.nc"), str(tmp_path / "o.nc"))
    assert set(df["time_idx"]) == {0}
