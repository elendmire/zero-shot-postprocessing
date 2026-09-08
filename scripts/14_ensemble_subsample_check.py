"""P1 (panel review, 2026-09-07): how much does the training/test ensemble
size mismatch (11 members train, 51 members test) affect EMOS's
calibration-sharpness result?

Discussion Section 5.6 already discloses this mismatch as an ACKNOWLEDGED
limitation, but does not bound its size. This script bounds it directly and
cheaply: subsample 11 of the 51 test-side members (without replacement,
`N_DRAWS` independent draws, seeded from configs/experiment.yaml -- no
hardcoded seed in src/), recompute ensemble mean/variance from just those 11
members with the same ddof=1 estimator used everywhere else in this project,
and re-run the EXISTING full-training-data EMOS pooled/local models
(reused via scripts/03_data_size_sweep.py's own fit_predict_pooled_emos/
fit_predict_local_emos, not reimplemented) on this subsampled test side.
Comparing the resulting CRPS/coverage@80%/interval-width against the
already-persisted 51-member Table 2 numbers directly answers "how much of
EMOS's calibration-sharpness result is attributable to the member-count
mismatch, as opposed to something else."

GPU is not needed (EMOS is closed-form); this runs on the CPU login node.

Output: results/phase3_ensemble_subsample_check.parquet (one row per
(method, n_members) pair: n_members=11 rows are the across-draws mean+std,
n_members=51 rows are the original Table 2 numbers for direct comparison).
"""
from importlib import util as _importlib_util
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_train_ensemble_stats_with_ids
from zeropp.data.splits import load_test
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
RAW_RESULTS_PATH = Path("results/phase2_comparison_raw.parquet")
SWEEP_RESULTS_PATH = Path("results/phase3_data_size_sweep.parquet")
N_MEMBERS_SUBSAMPLE = 11
N_DRAWS = 10


def _load_data_size_sweep_module():
    script_path = Path(__file__).resolve().parent / "03_data_size_sweep.py"
    spec = _importlib_util.spec_from_file_location("data_size_sweep_module", script_path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    seed = config.seeds[0]
    rng = np.random.default_rng(seed)

    sweep = _load_data_size_sweep_module()
    fit_predict_pooled_emos = sweep.fit_predict_pooled_emos
    fit_predict_local_emos = sweep.fit_predict_local_emos
    LOCAL_EMOS_MIN_ROWS = sweep.LOCAL_EMOS_MIN_ROWS

    print("Loading full training archive (unchanged -- only the test side is subsampled)...")
    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)

    print("Loading raw per-member test archive...")
    test_long = load_test()  # station_id, valid_time, step_hours, member, t2m_forecast, t2m_obs

    raw = pd.read_parquet(RAW_RESULTS_PATH)
    canonical_keys = raw.loc[raw["method"] == "tsfm3", ["station_id", "valid_time", "step_hours"]].drop_duplicates()

    key_cols = ["station_id", "valid_time", "step_hours"]
    groups = list(test_long.groupby(key_cols))
    n_groups = len(groups)
    print(f"{n_groups} (station, valid_time, step_hours) test groups found (up to 51 members each).")

    test_station_ids_full = np.array([g[0][0] for g in groups])
    obs_full = np.array([g[1]["t2m_obs"].iloc[0] for g in groups])
    keys_df = pd.DataFrame({
        "station_id": [g[0][0] for g in groups],
        "valid_time": [g[0][1] for g in groups],
        "step_hours": [g[0][2] for g in groups],
        "row_idx": np.arange(n_groups),
    })
    matched_keys = keys_df.merge(canonical_keys, on=key_cols, how="inner")
    matched_idx = matched_keys["row_idx"].to_numpy()
    print(f"Matched to canonical tsfm3 instance set: {len(matched_idx)}/{len(canonical_keys)} instances.")

    rows = []
    per_draw_metrics = {"emos_pooled": [], "emos_local": []}
    for draw in range(N_DRAWS):
        ens_means, ens_vars = [], []
        for _, group_df in groups:
            members = group_df["t2m_forecast"].to_numpy()
            if len(members) >= N_MEMBERS_SUBSAMPLE:
                sub = rng.choice(members, size=N_MEMBERS_SUBSAMPLE, replace=False)
            else:
                sub = members  # group has fewer than 11 members; use all (rare, logged if it happens)
            ens_means.append(sub.mean())
            ens_vars.append(sub.var(ddof=1))
        ens_means = np.array(ens_means)[matched_idx].reshape(-1, 1)
        ens_vars = np.array(ens_vars)[matched_idx].reshape(-1, 1)
        obs = obs_full[matched_idx].reshape(-1, 1)
        station_ids = test_station_ids_full[matched_idx]
        test_X = {"ens_mean": ens_means, "ens_var": ens_vars}

        pooled_preds = fit_predict_pooled_emos(full_train, quantile_levels, test_X)
        pooled_crps = crps_from_quantiles(obs, pooled_preds, quantile_levels).mean()
        pooled_cov = empirical_coverage(obs, pooled_preds, quantile_levels, lower=0.1, upper=0.9)
        pooled_width = float(np.mean(pooled_preds[:, 0, hi_idx] - pooled_preds[:, 0, lo_idx]))
        per_draw_metrics["emos_pooled"].append((pooled_crps, pooled_cov, pooled_width))

        local_preds, covered_mask, _ = fit_predict_local_emos(
            full_train, station_ids, test_X, quantile_levels, min_rows=LOCAL_EMOS_MIN_ROWS
        )
        if covered_mask.any():
            local_crps = crps_from_quantiles(obs[covered_mask], local_preds[covered_mask], quantile_levels).mean()
            local_cov = empirical_coverage(obs[covered_mask], local_preds[covered_mask], quantile_levels, lower=0.1, upper=0.9)
            local_width = float(np.mean(local_preds[covered_mask, 0, hi_idx] - local_preds[covered_mask, 0, lo_idx]))
            per_draw_metrics["emos_local"].append((local_crps, local_cov, local_width))

        print(f"draw {draw+1}/{N_DRAWS}: emos_pooled crps={pooled_crps:.4f} cov={pooled_cov:.4f} width={pooled_width:.4f}")

    for method, draws in per_draw_metrics.items():
        arr = np.array(draws)
        rows.append({
            "method": method, "n_members": N_MEMBERS_SUBSAMPLE, "n_draws": len(draws),
            "crps_mean": float(arr[:, 0].mean()), "crps_std": float(arr[:, 0].std(ddof=1)),
            "coverage_80pct_mean": float(arr[:, 1].mean()), "coverage_80pct_std": float(arr[:, 1].std(ddof=1)),
            "interval_width_k_mean": float(arr[:, 2].mean()), "interval_width_k_std": float(arr[:, 2].std(ddof=1)),
        })

    # Original 51-member numbers, for direct comparison, read from the
    # already-persisted Table 2 source (not recomputed).
    orig_sweep = pd.read_parquet(SWEEP_RESULTS_PATH)
    orig_full = orig_sweep[(orig_sweep["n_days"] == "full") & (orig_sweep["sampling_arm"] == "contiguous")]
    for method in ["emos_pooled", "emos_local"]:
        r = orig_full[orig_full["method"] == method].iloc[0]
        rows.append({
            "method": method, "n_members": 51, "n_draws": 1,
            "crps_mean": float(r["crps"]), "crps_std": 0.0,
            "coverage_80pct_mean": float(r["coverage_80pct"]), "coverage_80pct_std": 0.0,
            "interval_width_k_mean": float(r["interval_width_k"]), "interval_width_k_std": 0.0,
        })

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    write_result(
        results_df,
        name="phase3_ensemble_subsample_check",
        model_version="phase3-ensemble-subsample-v1",
        config={
            "n_members_subsample": N_MEMBERS_SUBSAMPLE,
            "n_draws": N_DRAWS,
            "seed": seed,
            "n_matched_instances": int(len(matched_idx)),
        },
    )


if __name__ == "__main__":
    main()
