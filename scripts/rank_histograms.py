"""Rank histograms of the nine predicted quantiles (Fig. 7).

Methods: raw ensemble, TimesFM-3, Chronos-2 (from the prediction files), and pooled and local EMOS fit on
the full reforecast archive. All methods are scored on the same instances (the tsfm3 keys of
predictions_main.parquet). Each histogram has ten bins: the observation below q0.1, between neighbouring
quantiles, above q0.9.

Output: results/rank_histograms.parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table
from zeropp.data.splits import load_train
from zeropp.eval.calibration import quantile_rank_histogram
from zeropp.eval.results import write_result
from zeropp.models.emos import EMOS

RAW_RESULTS_PATH = "results/predictions_main.parquet"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
KEY_COLS = ["station_id", "valid_time", "step_hours"]
LOCAL_EMOS_MIN_ROWS = 5
N_BINS = 10


def fit_predict_local_emos_full(
    train_df: pd.DataFrame,
    test_station_ids: np.ndarray,
    test_X: dict,
    quantile_levels: list[float],
    min_rows: int = LOCAL_EMOS_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One EMOS per station; test rows of stations with fewer than min_rows training rows are not covered."""
    n = len(test_station_ids)
    preds = np.full((n, 1, len(quantile_levels)), np.nan)
    covered_mask = np.zeros(n, dtype=bool)
    counts = train_df["station_id"].value_counts()
    fittable_stations = counts[counts >= min_rows].index
    for sid in fittable_stations:
        station_train = train_df[train_df["station_id"] == sid]
        model = EMOS(quantile_levels=quantile_levels).fit(station_train)
        rows_mask = test_station_ids == sid
        if not rows_mask.any():
            continue
        preds[rows_mask] = model.predict_quantiles(
            {"ens_mean": test_X["ens_mean"][rows_mask], "ens_var": test_X["ens_var"][rows_mask]}
        )
        covered_mask[rows_mask] = True
    n_unique_test_stations = len(np.unique(test_station_ids))
    n_covered_test_stations = len(np.unique(test_station_ids[covered_mask])) if covered_mask.any() else 0
    coverage_fraction = float(n_covered_test_stations) / float(n_unique_test_stations or 1)
    return preds, covered_mask, coverage_fraction


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]

    raw = pd.read_parquet(RAW_RESULTS_PATH)
    canonical_keys = raw.loc[raw["method"] == "tsfm3", KEY_COLS].drop_duplicates()

    for m in [x for x in ("raw_ensemble", "emos") if x in set(raw["method"])]:
        mk = raw.loc[raw["method"] == m, KEY_COLS].drop_duplicates()
        matched = mk.merge(canonical_keys, on=KEY_COLS, how="inner")
        assert len(matched) == len(canonical_keys) == len(mk), f"instances of {m!r} differ from those of tsfm3"
    print(f"{len(canonical_keys)} instances")

    rows_out = []
    # pooled EMOS is refit below (the emos rows of the predictions file are not used)
    method_label_map = {"raw_ensemble": "raw_ensemble", "tsfm3": "tsfm3"}
    chronos_path = Path("results/chronos2_predictions.parquet")
    if chronos_path.exists():
        raw = pd.concat([raw, pd.read_parquet(chronos_path)], ignore_index=True)
        method_label_map["chronos2"] = "chronos2"
    for src_method, label in method_label_map.items():
        df_m = raw[raw["method"] == src_method].merge(canonical_keys, on=KEY_COLS, how="inner")
        y = df_m["obs"].to_numpy().reshape(-1, 1)
        qp = df_m[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
        hist = quantile_rank_histogram(y, qp, quantile_levels)
        for i, frac in enumerate(hist):
            rows_out.append({
                "method": label, "bin_index": i, "bin_lo": i / N_BINS, "bin_hi": (i + 1) / N_BINS,
                "fraction": float(frac), "n_instances": int(len(y)),
            })
        print(f"method={label} ({len(y)} instances)")

    # EMOS local and pooled, fit on the full training set
    train_df = load_train()
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)
    grouped = test_df.groupby(KEY_COLS)
    ens_means, ens_vars, obs_values = [], [], []
    key_station_ids, key_valid_times, key_step_hours = [], [], []
    for key, group in grouped:
        station_id, valid_time, step_hours = key
        ens_means.append(group["t2m_forecast"].mean())
        ens_vars.append(group["t2m_forecast"].var())
        obs_values.append(group["t2m_obs"].iloc[0])
        key_station_ids.append(station_id)
        key_valid_times.append(valid_time)
        key_step_hours.append(step_hours)
    ens_means = np.array(ens_means).reshape(-1, 1)
    ens_vars = np.array(ens_vars).reshape(-1, 1)
    obs_values = np.array(obs_values).reshape(-1, 1)
    test_station_ids = np.array(key_station_ids)
    test_keys_df = pd.DataFrame({
        "station_id": key_station_ids, "valid_time": key_valid_times, "step_hours": key_step_hours,
        "row_idx": np.arange(len(key_station_ids)),
    })

    matched_keys = test_keys_df.merge(canonical_keys, on=KEY_COLS, how="inner")
    assert len(matched_keys) == len(canonical_keys), "test set does not contain all instances of the prediction file"
    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means_m = ens_means[matched_row_idx]
    ens_vars_m = ens_vars[matched_row_idx]
    obs_m = obs_values[matched_row_idx]
    sid_m = test_station_ids[matched_row_idx]
    test_X = {"ens_mean": ens_means_m, "ens_var": ens_vars_m}

    pooled = EMOS(quantile_levels=quantile_levels).fit(train_df).predict_quantiles(test_X)
    for i, frac in enumerate(quantile_rank_histogram(obs_m, pooled, quantile_levels)):
        rows_out.append({
            "method": "emos_pooled", "bin_index": i, "bin_lo": i / N_BINS, "bin_hi": (i + 1) / N_BINS,
            "fraction": float(frac), "n_instances": int(len(obs_m)),
        })

    preds, covered_mask, coverage_fraction = fit_predict_local_emos_full(train_df, sid_m, test_X, quantile_levels)
    n_covered = int(covered_mask.sum())
    print(f"method=emos_local: station coverage_fraction={coverage_fraction:.4f} "
          f"({n_covered}/{len(sid_m)} test instances covered)")
    hist_local = quantile_rank_histogram(obs_m[covered_mask], preds[covered_mask], quantile_levels)
    for i, frac in enumerate(hist_local):
        rows_out.append({
            "method": "emos_local", "bin_index": i, "bin_lo": i / N_BINS, "bin_hi": (i + 1) / N_BINS,
            "fraction": float(frac), "n_instances": n_covered,
        })
    print(f"method=emos_local ({n_covered} instances)")

    out_df = pd.DataFrame(rows_out)
    write_result(
        out_df,
        name="rank_histograms",
        model_version="rank-histograms",
        config={
            "quantile_levels": quantile_levels,
            "n_bins": N_BINS,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "emos_local_station_coverage_fraction": coverage_fraction,
            "n_canonical_instances": int(len(canonical_keys)),
        },
    )


if __name__ == "__main__":
    main()
