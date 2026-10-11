"""DRN at the smallest training sizes (k = 1, 2, 3, 5, 7 cases, contiguous arm).

Same instances and scoring as training_size.py; the DRN is fit with three seeds and the mean and SD are
stored. The rows replace the method="drn" rows of results/training_size_small.parquet.

Output: results/training_size_small.parquet (Fig. 3, Table A1).
"""
import time

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.results import write_result
from zeropp.models.drn import DRN
import training_size as sw

KEY = ["station_id", "valid_time", "step_hours"]


def main() -> None:
    cfg = load_experiment_config()
    levels, seeds = cfg.quantile_levels, cfg.data_size_sweep_seeds[:3]

    full_train = build_train_ensemble_stats_with_ids(sw.REFORECAST_PATH, sw.REFORECAST_OBS_PATH)
    long = build_test_long_table(sw.FORECAST_PATH, sw.FORECAST_OBS_PATH)
    agg = (long.groupby(KEY).agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"),
                                 obs=("t2m_obs", "first")).reset_index())
    raw = pd.read_parquet(sw.RAW_RESULTS_PATH)
    test = agg.merge(raw.loc[raw["method"] == "tsfm3", KEY], on=KEY, how="inner")
    assert len(test) == int((raw["method"] == "tsfm3").sum())
    y = test[["obs"]].to_numpy()
    X = {"ens_mean": test[["ens_mean"]].to_numpy(), "ens_var": test[["ens_var"]].to_numpy(),
         "station_id": test["station_id"].to_numpy()}

    rows = []
    for k in sw.LOW_N_K_GRID:
        train = sw.sample_contiguous(full_train, sw.n_days_for_exact_k(k))
        assert len(train[["year_idx", "time_idx"]].drop_duplicates()) == k
        cols = train[["station_id", "year_idx", "time_idx", "ens_mean", "ens_var", "t2m_obs"]]
        t0, runs = time.perf_counter(), []
        for seed in seeds:
            m = DRN(quantile_levels=levels, seed=seed).fit(cols)
            runs.append(sw.compute_metrics(y, m.predict_quantiles(X), levels))
            print(f"k={k} rows={len(train)} seed={seed}: updates={m.n_updates_} "
                  f"val_checks={len(m.val_loss_history_)} {runs[-1]}", flush=True)
        mean = {a: float(np.mean([r[a] for r in runs])) for a in runs[0]}
        rows.append({"k": k, "n_calendar_days_equiv": sw.k_to_calendar_days(k), "sampling_arm": "contiguous",
                     "seed": None, "method": "drn", **mean,
                     "crps_std": float(np.std([r["crps"] for r in runs])),
                     "coverage_80pct_std": float(np.std([r["coverage_80pct"] for r in runs])),
                     "interval_width_k_std": float(np.std([r["interval_width_k"] for r in runs])),
                     "n_stations_covered": None, "fit_seconds": time.perf_counter() - t0})
    old = pd.read_parquet("results/training_size_small.parquet")
    old = old[old["method"] != "drn"]
    new = pd.DataFrame(rows)
    print(new.to_string())
    write_result(pd.concat([old, new], ignore_index=True)[old.columns], name="training_size_small",
                 model_version="training-size-drn-small", config={"low_n_k_grid": sw.LOW_N_K_GRID, "drn_seeds": seeds})


if __name__ == "__main__":
    main()
