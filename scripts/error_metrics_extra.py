"""Error table rows for two more EMOS variants: local lead-specific EMOS (full archive) and the single-rule
rolling EMOS (preceding 40 days).
Computes bias, MAE, RMSE of the median, CRPS, coverage and width, overall and per season, on the matched test
instances and appends them to results/error_metrics.parquet (replacing earlier rows of these methods).
Supports Table 5.
"""
import multiprocessing as mp

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_lead
from zeropp.eval.results import write_result

import emos_variants as V
import error_metrics as err
import seasonal_training as seasonal
import training_size as sweep

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/predictions_main.parquet"
OUT_NAME = "error_metrics"
WINDOW_DAYS = 40
KEY = ["station_id", "valid_time", "step_hours"]
NEW_METHODS = ["emos_local_leadspec", "rolling_emos_local_adaptive_w40"]


_SHARED = {}


def _rolling_chunk(issue_times):
    """Rolling local_adaptive EMOS for a chunk of issue times (as in emos_variants.rolling_all_chunk)."""
    ops, test, levels = _SHARED["ops"], _SHARED["test"], _SHARED["levels"]
    w = pd.Timedelta(days=WINDOW_DAYS)
    ops_issue = ops["issue_time"].to_numpy()
    groups = test.groupby("issue_time").indices
    out = []
    for t in issue_times:
        idx = groups[t]
        t = pd.Timestamp(t)
        lo, hi = np.searchsorted(ops_issue, np.datetime64(t - w)), np.searchsorted(ops_issue, np.datetime64(t))
        hist = ops.iloc[lo:hi]
        hist = hist[hist["valid_time"] < t]
        if len(hist) < V.MIN_ROWS_LEAD:
            continue
        out.append((idx, V.local_adaptive(hist, test.iloc[idx], levels)))
    return out


def main(processes: int = 32) -> None:
    cfg = load_experiment_config()
    levels = cfg.quantile_levels

    train = build_train_ensemble_stats_with_lead(REFORECAST_PATH, REFORECAST_OBS_PATH)
    long = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)
    ops = (long.groupby(KEY).agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"),
                                 t2m_obs=("t2m_obs", "first")).reset_index())
    ops["issue_time"] = ops["valid_time"] - pd.to_timedelta(ops["step_hours"], unit="h")
    ops = ops.sort_values("issue_time").reset_index(drop=True)
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    test = ops.merge(raw.loc[raw["method"] == "tsfm3", KEY], on=KEY, how="inner").reset_index(drop=True)
    assert len(test) == int((raw["method"] == "tsfm3").sum())
    y = test["t2m_obs"].to_numpy()
    season = pd.to_datetime(test["issue_time"]).dt.month.map(seasonal.season_of_month).to_numpy()

    print("full-archive local lead-specific EMOS ...", flush=True)
    preds = {"emos_local_leadspec": V.lead_specific_local(train, test, levels)}

    print("rolling local_adaptive W=40 ...", flush=True)
    _SHARED.update(ops=ops, test=test, levels=levels)
    issue_times = list(test.groupby("issue_time").indices.keys())
    chunks = [issue_times[i::processes * 4] for i in range(processes * 4)]
    q = np.full((len(test), len(levels)), np.nan)
    with mp.get_context("fork").Pool(processes) as pool:
        for res in pool.imap_unordered(_rolling_chunk, chunks):
            for idx, qq in res:
                q[idx] = qq
    preds["rolling_emos_local_adaptive_w40"] = q

    rows = []
    for name, p in preds.items():
        ok = ~np.isnan(p[:, 0])
        print(f"{name}: covered {ok.mean():.4f} of {len(ok)} instances", flush=True)
        for s in ["all"] + seasonal.SEASONS:
            mask = ok if s == "all" else ok & (season == s)
            rows.append({"season": s, "method": name, "n_instances": int(mask.sum()),
                         **err.point_errors(y[mask].reshape(-1, 1), p[mask][:, None, :], levels),
                         **sweep.compute_metrics(y[mask].reshape(-1, 1), p[mask][:, None, :], levels)})
    new = pd.DataFrame(rows)
    print(new.to_string())
    old = pd.read_parquet(f"results/{OUT_NAME}.parquet")
    old = old[~old["method"].isin(NEW_METHODS)]
    assert list(new.columns) == list(old.columns), (new.columns.tolist(), old.columns.tolist())
    write_result(pd.concat([old, new], ignore_index=True), name=OUT_NAME, model_version="error-metrics",
                 config={"quantile_levels": levels, "point_forecast": "q0.5", "added_methods": NEW_METHODS,
                         "rolling_window_days": WINDOW_DAYS})


if __name__ == "__main__":
    main()
