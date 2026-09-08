"""Compute PIT (probability integral transform) histograms for F4 (calibration
diagnostics figure): raw_ensemble, tsfm3, emos_pooled, emos_local, all at full
training-data size (N=full), on the identical evaluation instance set.

No PIT histogram result file existed anywhere in results/ before this script.
`raw_ensemble`, `tsfm3` and `emos` (pooled, full-N) quantile predictions are
already persisted in `results/phase2_comparison_raw.parquet` (produced by
`scripts/01_run_baselines.py` / `scripts/02_run_tsfm.py`) and are reused
directly here -- no refit needed for those three. `emos_local` at full N was
never fit or persisted anywhere in this project (phase2_comparison_raw.parquet
has only pooled EMOS; the N-sweep in scripts/03_data_size_sweep.py fits local
EMOS at every N but only ever persists AGGREGATE metrics, never per-instance
predictions), so it is fit fresh here, once, at full N, following the exact
same per-station fit / min-rows / coverage-fraction pattern as
`scripts/03_data_size_sweep.py`'s `fit_predict_local_emos` (LOCAL_EMOS_MIN_ROWS
= 5 there; duplicated here rather than imported, since `03_data_size_sweep.py`
is a numerically-prefixed script module, not a package, and this project's
convention is one independent script per computation step, not cross-script
imports).

"Full N" here means the canonical EUPPBench train/test split via
`zeropp.data.splits.load_train`/`build_test_long_table` -- the SAME train_df
`scripts/01_run_baselines.py` used to fit the pooled "emos" method already in
phase2_comparison_raw.parquet -- not the reforecast-archive subsampling path
`03_data_size_sweep.py` uses for its N-axis sweep (that path targets varying N;
here N is fixed at "all of it", so the plain canonical split is the correct,
simpler full-N training set and keeps emos_local directly comparable to the
already-persisted emos_pooled numbers, which were fit the same way).

Instance-set join: phase2_comparison_raw.parquet's raw_ensemble/emos/tsfm3 rows
are already restricted (by 02_run_tsfm.py) to test instances with sufficient
clean lookback context for TimesFM-3 -- a strict subset of the full test set.
tsfm3's own (station_id, valid_time, step_hours) key set is used as the
canonical instance set for every method here (fail-fast asserts below verify
raw_ensemble/emos already match it, mirroring 03_data_size_sweep.py's own join
check), and emos_local's freshly-built test set is inner-joined onto it before
scoring, so all four methods' PIT histograms are computed on the identical
instances.

Caption-relevant note (F4, verbatim from docs/figure_style_guide.md): "PIT
uniformity is assessed descriptively; no formal uniformity test valid under
serial dependence is applied." No such test is run here either.
"""
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table
from zeropp.data.splits import load_train
from zeropp.eval.calibration import pit_histogram, pit_values
from zeropp.eval.results import write_result
from zeropp.models.emos import EMOS

RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
KEY_COLS = ["station_id", "valid_time", "step_hours"]
LOCAL_EMOS_MIN_ROWS = 5  # matches scripts/03_data_size_sweep.py's LOCAL_EMOS_MIN_ROWS
N_BINS = 10


def fit_predict_local_emos_full(
    train_df: pd.DataFrame,
    test_station_ids: np.ndarray,
    test_X: dict,
    quantile_levels: list[float],
    min_rows: int = LOCAL_EMOS_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One EMOS per station, fit only on that station's rows in train_df. A test
    row whose station has < min_rows training rows is EXCLUDED (covered_mask),
    never filled in with a pooled fallback -- same contract as
    scripts/03_data_size_sweep.py's fit_predict_local_emos, duplicated here (see
    module docstring) rather than imported."""
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

    for m in ["raw_ensemble", "emos"]:
        mk = raw.loc[raw["method"] == m, KEY_COLS].drop_duplicates()
        matched = mk.merge(canonical_keys, on=KEY_COLS, how="inner")
        assert len(matched) == len(canonical_keys) == len(mk), (
            f"instance-set join: method={m!r} keys diverge from tsfm3's key set in "
            f"{RAW_RESULTS_PATH} -- refusing to silently compare mismatched instance sets"
        )
    print(f"instance-set join: raw_ensemble/emos/tsfm3 share an identical instance-key set "
          f"({len(canonical_keys)} instances)")

    rows_out = []
    method_label_map = {"raw_ensemble": "raw_ensemble", "tsfm3": "tsfm3", "emos": "emos_pooled"}
    for src_method, label in method_label_map.items():
        df_m = raw[raw["method"] == src_method].merge(canonical_keys, on=KEY_COLS, how="inner")
        y = df_m["obs"].to_numpy().reshape(-1, 1)
        qp = df_m[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
        pit = pit_values(y, qp, quantile_levels)
        hist = pit_histogram(pit.flatten(), n_bins=N_BINS)
        for i, frac in enumerate(hist):
            rows_out.append({
                "method": label, "bin_index": i, "bin_lo": i / N_BINS, "bin_hi": (i + 1) / N_BINS,
                "fraction": float(frac), "n_instances": int(len(pit)),
            })
        print(f"method={label} ({len(pit)} instances): PIT histogram computed")

    # --- emos_local: fresh full-N fit, never persisted anywhere before ---
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
    assert len(matched_keys) == len(canonical_keys), (
        f"emos_local instance-set join incomplete: matched {len(matched_keys)} of "
        f"{len(canonical_keys)} canonical (tsfm3) instances -- refusing to score emos_local "
        "on a partial join"
    )
    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means_m = ens_means[matched_row_idx]
    ens_vars_m = ens_vars[matched_row_idx]
    obs_m = obs_values[matched_row_idx]
    sid_m = test_station_ids[matched_row_idx]
    test_X = {"ens_mean": ens_means_m, "ens_var": ens_vars_m}

    preds, covered_mask, coverage_fraction = fit_predict_local_emos_full(train_df, sid_m, test_X, quantile_levels)
    n_covered = int(covered_mask.sum())
    print(f"method=emos_local: station coverage_fraction={coverage_fraction:.4f} "
          f"({n_covered}/{len(sid_m)} test instances covered)")
    pit_local = pit_values(obs_m[covered_mask], preds[covered_mask], quantile_levels)
    hist_local = pit_histogram(pit_local.flatten(), n_bins=N_BINS)
    for i, frac in enumerate(hist_local):
        rows_out.append({
            "method": "emos_local", "bin_index": i, "bin_lo": i / N_BINS, "bin_hi": (i + 1) / N_BINS,
            "fraction": float(frac), "n_instances": n_covered,
        })
    print(f"method=emos_local ({n_covered} instances): PIT histogram computed")

    out_df = pd.DataFrame(rows_out)
    write_result(
        out_df,
        name="phase3_pit_histograms",
        model_version="phase3-pit-v1",
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
