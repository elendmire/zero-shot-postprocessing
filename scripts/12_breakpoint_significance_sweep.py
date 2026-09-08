"""BL-1 (Introduction review, 2026-09-07): is the EMOS-vs-TimesFM-3 CRPS
BREAKPOINT itself statistically significant, not just a separately chosen,
coarser training size?

Motivation: the paper's headline breakpoint (results/phase3_low_n_grid_breakpoints.parquet)
puts EMOS pooled's CRPS crossing at k~1.6 cases and EMOS local's at k~1.8 --
both below k=9, the only training size scripts/07_spatial_block_significance.py
(Task T1.3) ever significance-tested. That test's own k=9 point is a
DIFFERENT, larger training size than the breakpoint itself, chosen because it
was the round Task 4's original headline sentence used, not because it is the
breakpoint. Whether the CRPS advantage is already significant at the training
sizes surrounding the actual crossing (k=1..9) was never tested. This script
closes that gap directly.

Reuses scripts/03_data_size_sweep.py's LOW_N_K_GRID ([1, 2, 3, 5, 7]) plus the
existing k=9 point (scripts/07_spatial_block_significance.py's own N_DAYS_TARGET
converted via n_days_for_exact_k for consistency), for BOTH emos_pooled and
emos_local (the T1.3 script only ever tested emos_pooled), against TimesFM-3,
CRPS only (per the review's explicit scope -- not coverage, not other
metrics), with BOTH station- and day-blocked paired tests (the day block is
this project's primary block definition, per scripts/07_spatial_block_significance.py's
own finding that it is the more defensible reading of this panel's true
dependence structure).

Local EMOS's `covered_mask` (scripts/03_data_size_sweep.py's
`fit_predict_local_emos`) excludes test stations whose local training subset
at a given k has fewer than LOCAL_EMOS_MIN_ROWS rows; the significance test
at each k is computed only on the covered subset, exactly as every other
local-EMOS result in this project already does, and covered_fraction is
persisted alongside every local-EMOS row so a reader can see how much of the
test set that k's local-EMOS row actually speaks for.

Output: results/phase3_breakpoint_significance_sweep.parquet, one row per
(k, emos_variant, block_definition) -- 6 k-values x 2 variants x 2 blocks = 24
rows, CRPS only.
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

# LOW_N_K_GRID (scripts/03_data_size_sweep.py) plus the existing k=9 headline
# point (scripts/07_spatial_block_significance.py), so this sweep spans both
# the breakpoint region and the previously-tested point -- k=9's row here is
# expected to reproduce that script's own numbers as a consistency check.
K_GRID = [1, 2, 3, 5, 7, 9]


def _load_data_size_sweep_module():
    """Same mechanism as scripts/07_spatial_block_significance.py and
    tests/test_data_size_sweep.py: 03_data_size_sweep.py's filename starts
    with a digit, so a normal `import` is a syntax error. Only top-level
    imports/definitions execute on load; main() only runs under
    `if __name__ == "__main__"`, so this is side-effect-free."""
    script_path = Path(__file__).resolve().parent / "03_data_size_sweep.py"
    spec = _importlib_util.spec_from_file_location("data_size_sweep_module", script_path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def day_block_ids(valid_time) -> np.ndarray:
    """Verbatim copy of scripts/07_spatial_block_significance.py's function of
    the same name -- calendar-date block ids from a valid_time array, dropping
    time-of-day so every instance sharing a calendar date across all 49
    stations lands in the same block."""
    return pd.to_datetime(pd.Series(valid_time)).dt.normalize().to_numpy()


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    bootstrap_seed = config.seeds[0]

    sweep = _load_data_size_sweep_module()
    sample_contiguous = sweep.sample_contiguous
    fit_predict_pooled_emos = sweep.fit_predict_pooled_emos
    fit_predict_local_emos = sweep.fit_predict_local_emos
    n_days_for_exact_k = sweep.n_days_for_exact_k
    LOCAL_EMOS_MIN_ROWS = sweep.LOCAL_EMOS_MIN_ROWS

    print("Loading reforecast (train) and forecast (test) archives...")
    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    # --- Aggregate test rows to one (station_id, valid_time, step_hours)
    # instance, identical to scripts/07_spatial_block_significance.py's own
    # main(), so the instance-set join below behaves identically. ---
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

    key_cols = ["station_id", "valid_time", "step_hours"]
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    canonical_key_set = raw.loc[raw["method"] == "tsfm3", key_cols].drop_duplicates()

    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    n_matched = len(matched_keys)
    assert n_matched == len(canonical_key_set), (
        f"instance-set join did not fully cover the parquet's tsfm3 instances "
        f"({n_matched} matched vs {len(canonical_key_set)} in parquet) -- refusing to "
        "proceed with a partial join."
    )

    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means = ens_means[matched_row_idx]
    ens_vars = ens_vars[matched_row_idx]
    obs_values = obs_values[matched_row_idx]
    test_station_ids = test_station_ids[matched_row_idx]
    test_X = {"ens_mean": ens_means, "ens_var": ens_vars}

    # --- TimesFM-3 per-instance CRPS, left-merged onto matched_keys' row order
    # ONCE (N-independent, reused for every k in the sweep below). ---
    tsfm3_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all(), (
        "left-merging tsfm3 rows onto matched_keys produced an unmatched row."
    )
    assert len(tsfm3_ordered) == n_matched
    tsfm3_qp = tsfm3_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    tsfm3_y = tsfm3_ordered["obs"].to_numpy().reshape(-1, 1)
    tsfm3_crps_full = crps_from_quantiles(tsfm3_y, tsfm3_qp, quantile_levels).flatten()

    day_ids_full = day_block_ids(matched_keys["valid_time"])

    rows = []
    for k in K_GRID:
        n_days = n_days_for_exact_k(k)
        train_contig = sample_contiguous(full_train, n_days)
        actual_k = len(train_contig[["year_idx", "time_idx"]].drop_duplicates())
        assert actual_k == k, (
            f"n_days_for_exact_k round-trip failed: wanted k={k}, got {actual_k} -- "
            "a silent k mismatch would test the wrong training size."
        )

        # --- emos_pooled at this k: every matched instance is scored. ---
        pooled_preds = fit_predict_pooled_emos(train_contig, quantile_levels, test_X)
        pooled_crps = crps_from_quantiles(obs_values, pooled_preds, quantile_levels).flatten()

        for block_name, block_ids in [("station", test_station_ids), ("day", day_ids_full)]:
            test_result = station_blocked_paired_test(pooled_crps, tsfm3_crps_full, block_ids)
            point, ci_lo, ci_hi = block_bootstrap_skill_score_ci(
                pooled_crps, tsfm3_crps_full, block_ids, seed=bootstrap_seed
            )
            rows.append({
                "k_cases": k, "emos_variant": "emos_pooled", "block_definition": block_name,
                "n_instances": len(pooled_crps), "covered_fraction": 1.0,
                **test_result, "skill_score_point": point,
                "skill_score_ci_low": ci_lo, "skill_score_ci_high": ci_hi, "skill_score_ci": 0.95,
            })
            print(
                f"k={k} emos_pooled block={block_name}: mean diff(emos-tsfm3)="
                f"{test_result['block_mean_diff']:.5f}, t p={test_result['t_pvalue']:.5f}, "
                f"wilcoxon p={test_result['wilcoxon_pvalue']:.5f}"
            )

        # --- emos_local at this k: only the covered subset (test stations with
        # >= LOCAL_EMOS_MIN_ROWS local training rows at this k) is scored; the
        # significance test is computed on that subset only, and
        # covered_fraction is persisted so the row's real population is
        # visible, not silently assumed to be the full test set. ---
        local_preds, covered_mask, coverage_fraction = fit_predict_local_emos(
            train_contig, test_station_ids, test_X, quantile_levels, min_rows=LOCAL_EMOS_MIN_ROWS
        )
        if covered_mask.any():
            local_crps_full = crps_from_quantiles(obs_values, local_preds, quantile_levels).flatten()
            local_crps = local_crps_full[covered_mask]
            tsfm3_crps_covered = tsfm3_crps_full[covered_mask]
            station_ids_covered = test_station_ids[covered_mask]
            day_ids_covered = day_ids_full[covered_mask]
            for block_name, block_ids in [("station", station_ids_covered), ("day", day_ids_covered)]:
                test_result = station_blocked_paired_test(local_crps, tsfm3_crps_covered, block_ids)
                point, ci_lo, ci_hi = block_bootstrap_skill_score_ci(
                    local_crps, tsfm3_crps_covered, block_ids, seed=bootstrap_seed
                )
                rows.append({
                    "k_cases": k, "emos_variant": "emos_local", "block_definition": block_name,
                    "n_instances": len(local_crps), "covered_fraction": coverage_fraction,
                    **test_result, "skill_score_point": point,
                    "skill_score_ci_low": ci_lo, "skill_score_ci_high": ci_hi, "skill_score_ci": 0.95,
                })
                print(
                    f"k={k} emos_local (covered_fraction={coverage_fraction:.4f}) block={block_name}: "
                    f"mean diff(emos-tsfm3)={test_result['block_mean_diff']:.5f}, "
                    f"t p={test_result['t_pvalue']:.5f}, wilcoxon p={test_result['wilcoxon_pvalue']:.5f}"
                )
        else:
            print(f"k={k} emos_local: no stations covered (all below LOCAL_EMOS_MIN_ROWS={LOCAL_EMOS_MIN_ROWS}), skipping.")

    results_df = pd.DataFrame(rows)

    write_result(
        results_df,
        name="phase3_breakpoint_significance_sweep",
        model_version="phase3-breakpoint-significance-v1",
        config={
            "quantile_levels": quantile_levels,
            "k_grid": K_GRID,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "bootstrap_seed": bootstrap_seed,
            "n_matched_instances": int(n_matched),
        },
    )


if __name__ == "__main__":
    main()
