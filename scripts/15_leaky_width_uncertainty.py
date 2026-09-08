"""B21 (panel review, 2026-09-07): the leakage-optimistic width margin
(5.5344 K vs. TimesFM-3's 5.1379 K, a 0.3965 K / ~7.7% gap) was reported as
"settled" and "real" with no uncertainty quantification, inconsistent with
this paper's own standard elsewhere (block bootstrap CIs, two test
statistics, block-definition checks for every other significance claim).
This script closes that gap: a day-blocked bootstrap confidence interval on
the mean per-instance width difference (leaky variant minus TimesFM-3),
reusing the same block-resampling logic as
`zeropp.eval.significance.block_bootstrap_skill_score_ci` but for a plain
mean difference rather than a CRPS skill-score ratio (skill score is not a
sensible framing for a width comparison, so this is not a call into that
function, but the same resampling discipline).

The leaky variant's per-instance quantiles are recomputed deterministically
from the SAME matched test set's ens_mean/ens_var (its lambda is fit by
root-finding on test coverage, no randomness); TimesFM-3's per-instance
quantiles are read directly from the already-persisted
results/phase2_comparison_raw.parquet -- no re-fitting or re-inference,
just recombining already-known real numbers. GPU is not needed.

Output: results/phase3_leaky_width_uncertainty.parquet (point estimate +
95% CI for the mean width difference, day-blocked).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_train_ensemble_stats_with_ids, build_test_long_table
from zeropp.eval.results import write_result
from zeropp.models.variance_inflation import VarianceInflationBaseline

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = Path("results/phase2_comparison_raw.parquet")
N_BOOT = 2000


def _day_block_ids(valid_time) -> np.ndarray:
    return pd.to_datetime(pd.Series(valid_time)).dt.normalize().to_numpy()


def _block_bootstrap_mean_diff(diff: np.ndarray, block_ids: np.ndarray, n_boot: int, seed: int):
    unique_blocks = np.unique(block_ids)
    block_index = {b: np.where(block_ids == b)[0] for b in unique_blocks}
    point = float(diff.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_index[b] for b in sampled])
        boot[i] = diff[idx].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    seed = config.seeds[0]

    print("Loading full training archive (for target_coverage's own reference, unused for fitting lambda) "
          "and test archive...")
    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_long = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    raw = pd.read_parquet(RAW_RESULTS_PATH)
    tsfm3_rows = raw[raw["method"] == "tsfm3"].copy()
    tsfm3_rows = tsfm3_rows.drop_duplicates(subset=["station_id", "valid_time", "step_hours"])
    quantile_cols = [f"q{q}" for q in quantile_levels]
    tsfm3_width = (tsfm3_rows[f"q{quantile_levels[hi_idx]}"] - tsfm3_rows[f"q{quantile_levels[lo_idx]}"]).to_numpy()

    # Aggregate test-side ensemble mean/var at (station, valid_time, step_hours)
    # resolution, matched to the same tsfm3 instance keys, same convention as
    # scripts/07/15/16.
    grouped = test_long.groupby(["station_id", "valid_time", "step_hours"]).agg(
        ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"), obs=("t2m_obs", "first")
    ).reset_index()
    matched = tsfm3_rows[["station_id", "valid_time", "step_hours"]].merge(
        grouped, on=["station_id", "valid_time", "step_hours"], how="left"
    )
    assert matched["ens_mean"].notna().all(), "instance-set join dropped rows -- matched must be 1:1 with tsfm3 keys"

    # target_coverage: TimesFM-3's real empirical coverage on this exact
    # matched set, computed directly from its persisted per-instance
    # quantiles and observations -- phase2_comparison_raw.parquet has no
    # per-row aggregate coverage column, so this is not a re-fit, just the
    # same empirical_coverage computation every other script in this
    # project already does.
    obs = matched["obs"].to_numpy().reshape(-1, 1)
    qp_tsfm3 = tsfm3_rows[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    target_coverage = float(
        np.mean((obs[:, 0] >= qp_tsfm3[:, 0, lo_idx]) & (obs[:, 0] <= qp_tsfm3[:, 0, hi_idx]))
    )
    print(f"target_coverage (TimesFM-3, this matched set) = {target_coverage:.6f}")

    leaky_train_df = matched.rename(columns={"obs": "t2m_obs"})[["ens_mean", "ens_var", "t2m_obs"]]
    leaky_model = VarianceInflationBaseline.from_coverage_target(
        target_coverage=target_coverage, train_df=leaky_train_df, quantile_levels=quantile_levels,
    )
    leaky_preds = leaky_model.predict_quantiles({
        "ens_mean": matched["ens_mean"].to_numpy().reshape(-1, 1),
        "ens_var": matched["ens_var"].to_numpy().reshape(-1, 1),
    })
    leaky_width = leaky_preds[:, 0, hi_idx] - leaky_preds[:, 0, lo_idx]
    print(f"leaky lambda={leaky_model.multiplier:.4f}, mean width={leaky_width.mean():.4f} K")
    print(f"tsfm3 mean width (same matched set)={tsfm3_width.mean():.4f} K")

    diff = leaky_width - tsfm3_width
    day_ids = _day_block_ids(matched["valid_time"])
    point, ci_lo, ci_hi = _block_bootstrap_mean_diff(diff, day_ids, N_BOOT, seed)
    print(
        f"Day-blocked bootstrap (n_boot={N_BOOT}, {len(np.unique(day_ids))} day-blocks): "
        f"mean width diff (leaky - tsfm3) = {point:.4f} K, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
    )
    if ci_lo > 0:
        print("CI excludes zero and is entirely positive: leaky variant is significantly WIDER than TimesFM-3.")
    elif ci_hi < 0:
        print("CI excludes zero and is entirely negative: leaky variant is significantly NARROWER than TimesFM-3.")
    else:
        print("CI includes zero: the width difference is not statistically distinguishable from zero at this block resolution.")

    results_df = pd.DataFrame([{
        "comparison": "leaky_width_minus_tsfm3_width",
        "point_estimate_k": point, "ci_low_k": ci_lo, "ci_high_k": ci_hi, "ci": 0.95,
        "n_instances": int(len(diff)), "n_day_blocks": int(len(np.unique(day_ids))),
        "n_boot": N_BOOT,
    }])
    write_result(
        results_df,
        name="phase3_leaky_width_uncertainty",
        model_version="phase3-leaky-width-uncertainty-v1",
        config={"seed": seed, "n_boot": N_BOOT, "target_coverage": target_coverage},
    )


if __name__ == "__main__":
    main()
