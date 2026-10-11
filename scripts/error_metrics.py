"""Errors of the predictive median and probabilistic scores of all methods, overall and by season.

For raw ensemble, TimesFM-3 (from predictions_main.parquet), pooled and local EMOS, DRN and the two
variance-inflation baselines (fit on the full reforecast archive), computes bias, MAE and RMSE of the median
and CRPS, qwCRPS, 80 % coverage and width on the matched test instances, for all seasons together and per
meteorological season (issue-date month). Also writes the station metadata used for Fig. 1 and Table 2.

Output: results/error_metrics.parquet (Table 5), results/station_metadata.parquet.
"""
import numpy as np
import pandas as pd
import xarray as xr

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.results import write_result
from zeropp.models.drn import DRN
from zeropp.models.variance_inflation import VarianceInflationBaseline

import seasonal_training as seasonal
import training_size as sweep

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/predictions_main.parquet"


def point_errors(y: np.ndarray, preds: np.ndarray, quantile_levels: list[float]) -> dict:
    """Bias, MAE and RMSE of the predictive median; y is (n, 1), preds (n, 1, n_quantiles), bias = forecast - observation."""
    err = preds[:, 0, quantile_levels.index(0.5)] - y[:, 0]
    return {"bias_k": float(np.mean(err)), "mae_k": float(np.mean(np.abs(err))),
            "rmse_k": float(np.sqrt(np.mean(err ** 2)))}


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]

    # station metadata for the study-area map and the data table
    fc = xr.open_dataset(FORECAST_PATH)
    meta = pd.DataFrame({
        "station_id": fc["station_id"].values,
        "station_name": fc["station_name"].values.astype(str),
        "latitude": fc["station_latitude"].values,
        "longitude": fc["station_longitude"].values,
        "altitude_m": fc["station_altitude"].values,
        "land_usage": fc["station_land_usage"].values,
    })

    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)
    key_cols = ["station_id", "valid_time", "step_hours"]
    agg = (
        test_df.groupby(key_cols)
        .agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"), obs=("t2m_obs", "first"))
        .reset_index()
    )
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    test = agg.merge(raw.loc[raw["method"] == "tsfm3", key_cols], on=key_cols, how="inner")
    assert len(test) == int((raw["method"] == "tsfm3").sum()), "instance-set join incomplete"
    issue = pd.to_datetime(test["valid_time"]) - pd.to_timedelta(test["step_hours"], unit="h")
    test["season"] = issue.dt.month.map(seasonal.season_of_month)
    meta["used_in_study"] = meta["station_id"].isin(test["station_id"].unique())

    y = test[["obs"]].to_numpy()
    test_X = {"ens_mean": test[["ens_mean"]].to_numpy(), "ens_var": test[["ens_var"]].to_numpy()}
    station_ids = test["station_id"].to_numpy()

    preds = {}
    for name in ["raw_ensemble", "tsfm3"]:
        m = test[key_cols].merge(raw[raw["method"] == name], on=key_cols, how="left")
        assert m[quantile_cols].notna().all().all()
        preds[name] = m[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    print("fitting EMOS pooled and local ...")
    preds["emos_pooled"] = sweep.fit_predict_pooled_emos(full_train, quantile_levels, test_X)
    local, covered, _ = sweep.fit_predict_local_emos(full_train, station_ids, test_X, quantile_levels)
    assert covered.all()
    preds["emos_local"] = local
    print("fitting DRN ...")
    drn = DRN(quantile_levels=quantile_levels, seed=config.data_size_sweep_seeds[0]).fit(
        full_train[["station_id", "year_idx", "time_idx", "ens_mean", "ens_var", "t2m_obs"]]
    )
    preds["drn"] = drn.predict_quantiles({**test_X, "station_id": station_ids})
    vi = VarianceInflationBaseline(quantile_levels=quantile_levels).fit(full_train[["ens_mean", "ens_var", "t2m_obs"]])
    preds["var_inflation_trainfit"] = vi.predict_quantiles(test_X)
    preds["var_inflation_fixed"] = VarianceInflationBaseline.from_fixed_multiplier(
        sweep.LAMBDA_CLIM, quantile_levels=quantile_levels
    ).predict_quantiles(test_X)

    rows = []
    for season in ["all"] + seasonal.SEASONS:
        mask = np.ones(len(test), dtype=bool) if season == "all" else (test["season"] == season).to_numpy()
        for name, p in preds.items():
            rows.append({
                "season": season, "method": name, "n_instances": int(mask.sum()),
                **point_errors(y[mask], p[mask], quantile_levels),
                **sweep.compute_metrics(y[mask], p[mask], quantile_levels),
            })
    out = pd.DataFrame(rows)
    print(out.to_string())
    cfg = {"quantile_levels": quantile_levels, "point_forecast": "q0.5",
           "lambda_clim": sweep.LAMBDA_CLIM, "var_inflation_trainfit_multiplier": vi.multiplier}
    write_result(out, name="error_metrics", model_version="error-metrics", config=cfg)
    write_result(meta, name="station_metadata", model_version="station-metadata", config={})


if __name__ == "__main__":
    main()
