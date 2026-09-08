"""Day-blocked (calendar-date-blocked) re-analysis of the two headline
EMOS-pooled-at-k=9 vs TimesFM-3 significance results from Task 6 E2
(`results/phase3_data_size_sweep_significance.parquet`), alongside the
existing station-blocked results.

Motivation: `zeropp.eval.significance.block_bootstrap_skill_score_ci` and
`station_blocked_paired_test` are both generic over `block_ids` -- nothing in
either function is hardcoded to "station." Every existing caller in this
project (scripts/04_summarize_results.py, scripts/03_data_size_sweep.py's E2
section) passes `station_id` as `block_ids`. Station-blocking absorbs
within-station temporal autocorrelation (across issue times and lead times),
but it does NOT absorb spatial/cross-station dependence: the same synoptic
weather system typically affects many/most of the 49 German stations
simultaneously, so treating the ~49 station-block means as independent (as
the station-blocked test does) likely still understates the true variance
and overstates significance for a between-day source of correlated error --
the mechanism Luger (2004, Bank of Canada working paper) "exact tests under
contemporaneous correlation" literature addresses. This script re-runs BOTH
significance functions verbatim (no changes to zeropp.eval.significance) with
`block_ids = valid_time` normalized to calendar date instead of station_id,
for the same two headline comparisons E2 already reports station-blocked:

  1. coverage_80pct gap: EMOS-pooled(k=9) vs TimesFM-3 (~0.0065, EMOS slightly
     higher/closer to nominal)
  2. crps gap: EMOS-pooled(k=9) vs TimesFM-3 (~0.098, EMOS better/lower)

Per-instance EMOS-pooled(k=9) predictions are RECONSTRUCTED here (no new model
fits beyond that single EMOS refit) using scripts/03_data_size_sweep.py's own
`sample_contiguous` / `fit_predict_pooled_emos` functions and its
instance-set join against `results/phase2_comparison_raw.parquet`'s `tsfm3`
rows, imported via importlib exactly the way tests/test_data_size_sweep.py
already does -- reused verbatim, not reimplemented. TimesFM-3's per-instance
arrays come directly from that same parquet file's `tsfm3` rows (method values
are `raw_ensemble`/`emos`/`tsfm3`, NOT `timesfm3`; the observation column is
`obs`, NOT `t2m_obs` -- see docs/phase2_results_schema.md).
"""
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles
from zeropp.eval.significance import block_bootstrap_skill_score_ci, station_blocked_paired_test

from importlib import util as _importlib_util
from pathlib import Path

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"

# The one headline N this task re-tests: N=30 days, which scripts/03_data_size_sweep.py's
# measured DAYS_PER_CASE (~3.4833) rounds to k=9 -- the exact N/k Task 6 E2's headline
# sentence names for both the coverage and CRPS gaps. Asserted (not just assumed) below.
N_DAYS_TARGET = 30
EXPECTED_K = 9


def _load_data_size_sweep_module():
    """Import scripts/03_data_size_sweep.py as a module, the same mechanism
    tests/test_data_size_sweep.py already uses (its filename starts with a digit,
    so a normal `import` statement is a syntax error). Only top-level
    imports/definitions execute on load; main() only runs under
    `if __name__ == "__main__"`, so this is side-effect-free."""
    script_path = Path(__file__).resolve().parent / "03_data_size_sweep.py"
    spec = _importlib_util.spec_from_file_location("data_size_sweep_module", script_path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def day_block_ids(valid_time) -> np.ndarray:
    """Calendar-date block ids from a valid_time array/Series/list-like of
    timestamps -- the day-blocked counterpart to this project's existing
    station_id block_ids. Normalizes each timestamp to midnight of its own
    calendar date (drops time-of-day and any sub-day structure), so every
    instance sharing a calendar date -- across ALL 49 stations -- lands in the
    same block. This is the block definition that can absorb cross-station
    synoptic dependence (the same weather system affecting many stations on
    the same day); station-blocking cannot absorb that by construction, since
    a station block pools across days for one station, never across stations.

    Returns a numpy array of datetime64[ns] values (time-of-day zeroed), the
    same length and row order as the input, suitable as `block_ids` for
    `station_blocked_paired_test` / `block_bootstrap_skill_score_ci` exactly
    like `station_id` already is."""
    return pd.to_datetime(pd.Series(valid_time)).dt.normalize().to_numpy()


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    # Final-review convention (scripts/04_summarize_results.py): a bootstrap seed is
    # exactly the kind of hidden randomness CLAUDE.md's "no hardcoded seeds" rule
    # exists to prevent, even though that rule's literal text is scoped to src/ --
    # routed through config.seeds[0], the same real config value that script uses.
    bootstrap_seed = config.seeds[0]

    sweep = _load_data_size_sweep_module()
    sample_contiguous = sweep.sample_contiguous
    fit_predict_pooled_emos = sweep.fit_predict_pooled_emos

    print("Loading reforecast (train) and forecast (test) archives...")
    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    # --- Aggregate test rows to one (station_id, valid_time, step_hours) instance,
    # capturing key columns row-aligned with the aggregates -- same approach as
    # scripts/03_data_size_sweep.py's main() so the instance-set join below is
    # identical in spirit. ---
    grouped = test_df.groupby(["station_id", "valid_time", "step_hours"])
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
        "station_id": key_station_ids,
        "valid_time": key_valid_times,
        "step_hours": key_step_hours,
        "row_idx": np.arange(len(key_station_ids)),
    })

    # --- Instance-set join against the parquet's tsfm3 rows (same logic as
    # scripts/03_data_size_sweep.py's fix-round-1 finding 1) ---
    key_cols = ["station_id", "valid_time", "step_hours"]
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    canonical_key_set = raw.loc[raw["method"] == "tsfm3", key_cols].drop_duplicates()

    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    n_parquet_instances = len(canonical_key_set)
    n_matched = len(matched_keys)
    print(
        f"instance-set join: parquet tsfm3 instances={n_parquet_instances}, "
        f"matched intersection={n_matched}"
    )
    assert n_matched == n_parquet_instances, (
        f"instance-set join did not fully cover the parquet's tsfm3 instances "
        f"({n_matched} matched vs {n_parquet_instances} in parquet) -- refusing to "
        "proceed with a partial join, since that would silently score EMOS and "
        "TimesFM-3 on different instance sets."
    )

    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means = ens_means[matched_row_idx]
    ens_vars = ens_vars[matched_row_idx]
    obs_values = obs_values[matched_row_idx]
    test_station_ids = test_station_ids[matched_row_idx]
    test_X = {"ens_mean": ens_means, "ens_var": ens_vars}

    # --- EMOS-pooled at k=9 (N=30 days, contiguous arm), reconstructed verbatim
    # via scripts/03_data_size_sweep.py's own sample_contiguous/fit_predict_pooled_emos ---
    train_contig = sample_contiguous(full_train, N_DAYS_TARGET)
    actual_k = len(train_contig[["year_idx", "time_idx"]].drop_duplicates())
    assert actual_k == EXPECTED_K, (
        f"expected N_DAYS_TARGET={N_DAYS_TARGET} to round to k={EXPECTED_K} via "
        f"sample_contiguous/DAYS_PER_CASE, got k={actual_k} -- the two headline gaps "
        "this script re-tests (coverage ~0.0065, crps ~0.098) were both measured at "
        "k=9; a silent k mismatch would re-test the wrong N."
    )
    pooled_preds = fit_predict_pooled_emos(train_contig, quantile_levels, test_X)

    emos_crps = crps_from_quantiles(obs_values, pooled_preds, quantile_levels).flatten()
    emos_coverage = (
        (obs_values[:, 0] >= pooled_preds[:, 0, lo_idx]) & (obs_values[:, 0] <= pooled_preds[:, 0, hi_idx])
    ).astype(float)

    # --- TimesFM-3 per-instance arrays, left-merged onto matched_keys' row order
    # (NOT raw's own incidental row order) so the paired differential lines up
    # instance-for-instance with emos_crps/emos_coverage above -- same alignment
    # discipline as scripts/03_data_size_sweep.py's E2 section. ---
    tsfm3_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all(), (
        "left-merging tsfm3 rows onto matched_keys produced an unmatched row -- "
        "matched_keys should be a subset of tsfm3's own instance keys by construction."
    )
    assert len(tsfm3_ordered) == n_matched, (
        f"tsfm3_ordered has {len(tsfm3_ordered)} rows but matched_keys has {n_matched} -- "
        "the left-merge must be exactly 1:1, or the per-instance pairing below is broken."
    )
    tsfm3_qp = tsfm3_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    tsfm3_y = tsfm3_ordered["obs"].to_numpy().reshape(-1, 1)
    tsfm3_crps = crps_from_quantiles(tsfm3_y, tsfm3_qp, quantile_levels).flatten()
    tsfm3_coverage = (
        (tsfm3_y[:, 0] >= tsfm3_qp[:, 0, lo_idx]) & (tsfm3_y[:, 0] <= tsfm3_qp[:, 0, hi_idx])
    ).astype(float)

    print(
        f"EMOS-pooled(k={actual_k}) vs TimesFM-3, n={n_matched} matched instances: "
        f"mean crps emos={emos_crps.mean():.4f} tsfm3={tsfm3_crps.mean():.4f} "
        f"(gap={emos_crps.mean() - tsfm3_crps.mean():.4f}); "
        f"mean coverage_80pct emos={emos_coverage.mean():.4f} tsfm3={tsfm3_coverage.mean():.4f} "
        f"(gap={emos_coverage.mean() - tsfm3_coverage.mean():.4f})"
    )

    # --- Day-blocked ids (this task's new block definition), row-aligned with
    # matched_keys/emos_*/tsfm3_* via matched_keys's own row order. ---
    day_ids = day_block_ids(matched_keys["valid_time"])
    n_unique_days = len(np.unique(day_ids))
    n_unique_stations = len(np.unique(test_station_ids))
    print(f"block definitions: station -> {n_unique_stations} blocks, day -> {n_unique_days} blocks")

    block_defs = [("station", test_station_ids), ("day", day_ids)]
    metric_arrays = [
        ("crps", emos_crps, tsfm3_crps),
        ("coverage_80pct", emos_coverage, tsfm3_coverage),
    ]

    rows = []
    for metric_name, emos_arr, tsfm3_arr in metric_arrays:
        for block_name, block_ids in block_defs:
            test_result = station_blocked_paired_test(emos_arr, tsfm3_arr, block_ids)
            point, ci_lo, ci_hi = block_bootstrap_skill_score_ci(
                emos_arr, tsfm3_arr, block_ids, seed=bootstrap_seed
            )
            rows.append({
                "metric": metric_name,
                "block_definition": block_name,
                "n_days": N_DAYS_TARGET,
                "n_cases": actual_k,
                "method_a": "emos_pooled_k9",
                "method_b": "tsfm3",
                **test_result,
                "skill_score_point": point,
                "skill_score_ci_low": ci_lo,
                "skill_score_ci_high": ci_hi,
                "skill_score_ci": 0.95,
            })
            print(
                f"metric={metric_name}, block={block_name} ({test_result['n_blocks']} blocks): "
                f"mean diff(emos-tsfm3)={test_result['block_mean_diff']:.5f}, "
                f"t p={test_result['t_pvalue']:.5f}, wilcoxon p={test_result['wilcoxon_pvalue']:.5f}, "
                f"skill score={point:.5f} [{ci_lo:.5f}, {ci_hi:.5f}]"
            )

    results_df = pd.DataFrame(rows)

    # --- Agreement/divergence check: does the day-blocked test's significance
    # verdict (alpha=0.05, t-test) match the station-blocked test's, for each metric?
    print("\n===== Station-blocked vs day-blocked agreement (alpha=0.05, t-test) =====")
    for metric_name, _, _ in metric_arrays:
        station_row = results_df[(results_df["metric"] == metric_name) & (results_df["block_definition"] == "station")].iloc[0]
        day_row = results_df[(results_df["metric"] == metric_name) & (results_df["block_definition"] == "day")].iloc[0]
        station_sig = station_row["t_pvalue"] < 0.05
        day_sig = day_row["t_pvalue"] < 0.05
        verdict = "AGREE" if station_sig == day_sig else "DIVERGE"
        print(
            f"{metric_name}: station-blocked significant={station_sig} (p={station_row['t_pvalue']:.5f}), "
            f"day-blocked significant={day_sig} (p={day_row['t_pvalue']:.5f}) -> {verdict}"
        )

    write_result(
        results_df,
        name="phase3_spatial_block_significance",
        model_version="phase3-spatial-block-v1",
        config={
            "quantile_levels": quantile_levels,
            "n_days_target": N_DAYS_TARGET,
            "expected_k": EXPECTED_K,
            "bootstrap_seed": bootstrap_seed,
            "n_matched_instances": int(n_matched),
            "n_unique_stations": int(n_unique_stations),
            "n_unique_days": int(n_unique_days),
        },
    )


if __name__ == "__main__":
    main()
