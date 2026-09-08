from unittest.mock import patch

import pandas as pd

from zeropp.data.splits import load_test, load_train


def test_load_train_calls_build_train_ensemble_stats_with_reforecast_paths():
    fake_df = pd.DataFrame({"station_id": [1], "ens_mean": [280.0], "ens_var": [1.0], "t2m_obs": [280.5]})
    with patch("zeropp.data.splits.build_train_ensemble_stats", return_value=fake_df) as mock_build:
        result = load_train(data_dir="/some/dir")
    mock_build.assert_called_once_with(
        "/some/dir/germany_ensemble_reforecasts_t2m.nc",
        "/some/dir/germany_reforecasts_observations_t2m.nc",
    )
    assert result is fake_df


def test_load_test_calls_build_test_long_table_with_forecast_paths():
    fake_df = pd.DataFrame({"station_id": [1], "valid_time": [pd.Timestamp("2017-01-01")],
                             "step_hours": [0.0], "member": [0], "t2m_forecast": [280.0], "t2m_obs": [280.5]})
    with patch("zeropp.data.splits.build_test_long_table", return_value=fake_df) as mock_build:
        result = load_test(data_dir="/some/dir")
    mock_build.assert_called_once_with(
        "/some/dir/germany_ensemble_forecasts_t2m.nc",
        "/some/dir/germany_forecasts_observations_t2m.nc",
    )
    assert result is fake_df


def test_load_train_and_load_test_never_share_a_source_file():
    import inspect
    train_src = inspect.getsource(load_train)
    test_src = inspect.getsource(load_test)
    # The EUPPBench train/test boundary is enforced by construction: these two
    # functions must never reference the same raw filename.
    train_files = {w for w in train_src.split() if w.endswith('.nc"') or w.endswith(".nc'")}
    test_files = {w for w in test_src.split() if w.endswith('.nc"') or w.endswith(".nc'")}
    assert train_files.isdisjoint(test_files)
