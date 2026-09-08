"""Investigation 3 (docs/references/priority123_investigation_report.md,
per-lead-group significance testing): extends scripts/07_spatial_block_significance.py's
day-blocked + station-blocked paired significance testing (over
zeropp.eval.significance.station_blocked_paired_test /
block_bootstrap_skill_score_ci, both generic over `block_ids`, reused
UNCHANGED here) to run SEPARATELY within each of the three lead-time buckets
(0-24h, 24-72h, 72-120h) instead of pooled across all lead times.

This directly tests the strong claim (docs/results_index.md,
phase3_lead_time_bucketed_breakpoints.parquet row): "0-24h/CRPS/emos_pooled is
a genuine 'no crossing, worse throughout' (TimesFM-3 wins the whole tested
range in the nowcasting bucket)" -- i.e. EMOS never beats TimesFM-3 at 0-24h
EVEN WITH FULL TRAINING DATA. That claim is about FULL-N EMOS-pooled (the
breakpoint search's largest tested N, k=4180="full"), so this script tests
FULL-N EMOS-pooled (lead-pooled training, exactly
scripts/03_data_size_sweep.py's fit_predict_pooled_emos applied to the
ENTIRE reforecast archive, no subsampling) vs. TimesFM-3, per bucket, for
both the CRPS gap and the coverage_80pct gap -- not the k=9 EMOS this
project's OTHER significance script (09) tests.

Day-blocked is the primary block definition per this project's established
finding (scripts/07_spatial_block_significance.py, T1.3): station-blocking
understates cross-station synoptic dependence (the same weather system
typically affects many/most of the 49 German stations simultaneously).
Station-blocked is reported alongside as the (cheap) secondary comparison.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles
from zeropp.eval.significance import block_bootstrap_skill_score_ci, station_blocked_paired_test

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"

# --- Reuse verbatim via importlib (same mechanism every later script in this
# project uses for a filename starting with a digit): 07's fit_predict_pooled_emos,
# 08's LEAD_TIME_BUCKETS/assign_lead_time_bucket, 09's day_block_ids. No
# reimplementation of any of these. ---
_SCRIPT_07_PATH = Path(__file__).resolve().parent / "03_data_size_sweep.py"
_SPEC_07 = importlib.util.spec_from_file_location("data_size_sweep_module_12", _SCRIPT_07_PATH)
_sweep07 = importlib.util.module_from_spec(_SPEC_07)
_SPEC_07.loader.exec_module(_sweep07)
fit_predict_pooled_emos = _sweep07.fit_predict_pooled_emos

_SCRIPT_08_PATH = Path(__file__).resolve().parent / "06_lead_time_grouped_analysis.py"
_SPEC_08 = importlib.util.spec_from_file_location("lead_time_grouped_analysis_module_12", _SCRIPT_08_PATH)
_lead08 = importlib.util.module_from_spec(_SPEC_08)
_SPEC_08.loader.exec_module(_lead08)
assign_lead_time_bucket = _lead08.assign_lead_time_bucket
LEAD_TIME_BUCKETS = _lead08.LEAD_TIME_BUCKETS

_SCRIPT_09_PATH = Path(__file__).resolve().parent / "07_spatial_block_significance.py"
_SPEC_09 = importlib.util.spec_from_file_location("spatial_block_significance_module_12", _SCRIPT_09_PATH)
_spatial09 = importlib.util.module_from_spec(_SPEC_09)
_SPEC_09.loader.exec_module(_spatial09)
day_block_ids = _spatial09.day_block_ids


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    bootstrap_seed = config.seeds[0]  # same convention as 09: real config value, not a new hardcoded seed

    print("Loading reforecast (train) and forecast (test) archives...")
    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

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
        "station_id": key_station_ids, "valid_time": key_valid_times, "step_hours": key_step_hours,
        "row_idx": np.arange(len(key_station_ids)),
    })

    key_cols = ["station_id", "valid_time", "step_hours"]
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    canonical_key_set = raw.loc[raw["method"] == "tsfm3", key_cols].drop_duplicates()
    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    n_matched = len(matched_keys)
    assert n_matched == len(canonical_key_set), (
        f"instance-set join incomplete: matched {n_matched} of {len(canonical_key_set)} tsfm3 instances."
    )
    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means = ens_means[matched_row_idx]
    ens_vars = ens_vars[matched_row_idx]
    obs_values = obs_values[matched_row_idx]
    test_station_ids = test_station_ids[matched_row_idx]
    test_X = {"ens_mean": ens_means, "ens_var": ens_vars}
    lead_buckets_matched = assign_lead_time_bucket(matched_keys["step_hours"].to_numpy())
    day_ids = day_block_ids(matched_keys["valid_time"])

    # --- FULL-N pooled EMOS (lead-pooled training, no subsampling -- the
    # "0-24h EMOS never beats TimesFM-3 even at full N" claim's own EMOS). ---
    print(f"Fitting EMOS-pooled on the FULL reforecast archive ({len(full_train)} rows, no subsampling)...")
    full_preds = fit_predict_pooled_emos(full_train, quantile_levels, test_X)
    emos_crps_all = crps_from_quantiles(obs_values, full_preds, quantile_levels).flatten()
    emos_coverage_all = (
        (obs_values[:, 0] >= full_preds[:, 0, lo_idx]) & (obs_values[:, 0] <= full_preds[:, 0, hi_idx])
    ).astype(float)

    tsfm3_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all()
    tsfm3_qp = tsfm3_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    tsfm3_y = tsfm3_ordered["obs"].to_numpy().reshape(-1, 1)
    tsfm3_crps_all = crps_from_quantiles(tsfm3_y, tsfm3_qp, quantile_levels).flatten()
    tsfm3_coverage_all = (
        (tsfm3_y[:, 0] >= tsfm3_qp[:, 0, lo_idx]) & (tsfm3_y[:, 0] <= tsfm3_qp[:, 0, hi_idx])
    ).astype(float)

    rows = []
    for low, high, label in LEAD_TIME_BUCKETS:
        bucket_mask = lead_buckets_matched == label
        n_bucket = int(bucket_mask.sum())

        emos_crps = emos_crps_all[bucket_mask]
        emos_coverage = emos_coverage_all[bucket_mask]
        tsfm3_crps = tsfm3_crps_all[bucket_mask]
        tsfm3_coverage = tsfm3_coverage_all[bucket_mask]
        station_ids_bucket = test_station_ids[bucket_mask]
        day_ids_bucket = day_ids[bucket_mask]

        print(
            f"\n=== bucket={label} (n={n_bucket}): full-N EMOS-pooled vs TimesFM-3 ===\n"
            f"mean crps emos={emos_crps.mean():.4f} tsfm3={tsfm3_crps.mean():.4f} "
            f"(gap={emos_crps.mean() - tsfm3_crps.mean():+.4f}); "
            f"mean coverage_80pct emos={emos_coverage.mean():.4f} tsfm3={tsfm3_coverage.mean():.4f} "
            f"(gap={emos_coverage.mean() - tsfm3_coverage.mean():+.4f})"
        )

        block_defs = [("station", station_ids_bucket), ("day", day_ids_bucket)]
        metric_arrays = [
            ("crps", emos_crps, tsfm3_crps),
            ("coverage_80pct", emos_coverage, tsfm3_coverage),
        ]
        for metric_name, emos_arr, tsfm3_arr in metric_arrays:
            for block_name, block_ids in block_defs:
                test_result = station_blocked_paired_test(emos_arr, tsfm3_arr, block_ids)
                point, ci_lo, ci_hi = block_bootstrap_skill_score_ci(
                    emos_arr, tsfm3_arr, block_ids, seed=bootstrap_seed
                )
                rows.append({
                    "lead_time_bucket": label, "metric": metric_name, "block_definition": block_name,
                    "n_instances": n_bucket, "method_a": "emos_pooled_full_n", "method_b": "tsfm3",
                    **test_result,
                    "skill_score_point": point, "skill_score_ci_low": ci_lo, "skill_score_ci_high": ci_hi,
                    "skill_score_ci": 0.95,
                })
                print(
                    f"  metric={metric_name}, block={block_name} ({test_result['n_blocks']} blocks): "
                    f"mean diff(emos-tsfm3)={test_result['block_mean_diff']:.5f}, "
                    f"t p={test_result['t_pvalue']:.2e}, wilcoxon p={test_result['wilcoxon_pvalue']:.2e}, "
                    f"skill score={point:.5f} [{ci_lo:.5f}, {ci_hi:.5f}]"
                )

    results_df = pd.DataFrame(rows)

    print("\n===== Per-bucket significance summary (alpha=0.05, day-blocked t-test primary) =====")
    for low, high, label in LEAD_TIME_BUCKETS:
        for metric_name in ["crps", "coverage_80pct"]:
            day_row = results_df[
                (results_df["lead_time_bucket"] == label) & (results_df["metric"] == metric_name)
                & (results_df["block_definition"] == "day")
            ].iloc[0]
            station_row = results_df[
                (results_df["lead_time_bucket"] == label) & (results_df["metric"] == metric_name)
                & (results_df["block_definition"] == "station")
            ].iloc[0]
            day_sig = day_row["t_pvalue"] < 0.05
            station_sig = station_row["t_pvalue"] < 0.05
            emos_better = day_row["block_mean_diff"] < 0  # loss diff = emos - tsfm3; negative = emos lower/better
            print(
                f"{label} / {metric_name}: day-blocked p={day_row['t_pvalue']:.2e} "
                f"({'significant' if day_sig else 'not significant'}), "
                f"station-blocked p={station_row['t_pvalue']:.2e} "
                f"({'significant' if station_sig else 'not significant'}), "
                f"direction: {'EMOS better/lower' if emos_better else 'TimesFM-3 better/lower'} "
                f"(mean diff={day_row['block_mean_diff']:+.5f})"
            )

    write_result(
        results_df,
        name="phase3_per_lead_group_significance",
        model_version="phase3-per-lead-significance-v1",
        config={
            "quantile_levels": quantile_levels, "bootstrap_seed": bootstrap_seed,
            "n_matched_instances": int(n_matched), "lead_time_buckets": LEAD_TIME_BUCKETS,
            "method_a": "emos_pooled_full_n_lead_pooled_training",
        },
    )


if __name__ == "__main__":
    main()
