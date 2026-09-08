"""D44 (panel review, 2026-09-08): a cheap, useful cross-check on Finding
2 (calibration-sharpness). The paper's headline leaky-variant comparison
(Section 5.2/Table 2) fits the leakage-optimistic multiplier at ONE
coverage target -- TimesFM-3's own mean-only coverage, 0.7603. This
script asks the same question at a SECOND coverage target: the
two-covariate TimesFM-3 configuration's own achieved coverage, 0.8027
(Section 5.4/results/phase3_two_covariate_timesfm.parquet). If the leaky
variant is also wider than 5.8246 K at 0.8027, Finding 2 is confirmed at
two independent coverage points, not resting on a single one.

No GPU needed: this is the same root-finding + Gaussian-quantile
evaluation pattern as scripts/15_leaky_width_uncertainty.py, just at a
different target_coverage. Reuses the identical matched test set.

Output: results/phase3_leaky_lambda_at_two_cov_coverage.parquet
(point estimate for the leaky variant's width at target_coverage=0.8027).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table
from zeropp.eval.results import write_result
from zeropp.models.variance_inflation import VarianceInflationBaseline

FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = Path("results/phase2_comparison_raw.parquet")
TWO_COV_TARGET_COVERAGE = 0.802690  # results/phase3_two_covariate_timesfm.parquet, tsfm3_two_covariate row


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)

    print("Loading test archive...")
    test_long = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    raw = pd.read_parquet(RAW_RESULTS_PATH)
    tsfm3_rows = raw[raw["method"] == "tsfm3"].copy()
    tsfm3_rows = tsfm3_rows.drop_duplicates(subset=["station_id", "valid_time", "step_hours"])

    grouped = test_long.groupby(["station_id", "valid_time", "step_hours"]).agg(
        ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"), obs=("t2m_obs", "first")
    ).reset_index()
    matched = tsfm3_rows[["station_id", "valid_time", "step_hours"]].merge(
        grouped, on=["station_id", "valid_time", "step_hours"], how="left"
    )
    assert matched["ens_mean"].notna().all(), "instance-set join dropped rows -- matched must be 1:1 with tsfm3 keys"

    leaky_train_df = matched.rename(columns={"obs": "t2m_obs"})[["ens_mean", "ens_var", "t2m_obs"]]
    leaky_model = VarianceInflationBaseline.from_coverage_target(
        target_coverage=TWO_COV_TARGET_COVERAGE, train_df=leaky_train_df, quantile_levels=quantile_levels,
    )
    leaky_preds = leaky_model.predict_quantiles({
        "ens_mean": matched["ens_mean"].to_numpy().reshape(-1, 1),
        "ens_var": matched["ens_var"].to_numpy().reshape(-1, 1),
    })
    leaky_width = leaky_preds[:, 0, hi_idx] - leaky_preds[:, 0, lo_idx]

    obs = leaky_train_df["t2m_obs"].to_numpy().reshape(-1, 1)
    achieved_coverage = float(
        np.mean((obs[:, 0] >= leaky_preds[:, 0, lo_idx]) & (obs[:, 0] <= leaky_preds[:, 0, hi_idx]))
    )

    print(f"leaky lambda={leaky_model.multiplier:.4f}")
    print(f"target_coverage={TWO_COV_TARGET_COVERAGE:.6f}, achieved_coverage={achieved_coverage:.6f}")
    print(f"leaky mean width at this target = {leaky_width.mean():.4f} K")
    print("(for comparison: two-covariate TimesFM-3's own width at this coverage = 5.8246 K)")

    results_df = pd.DataFrame([{
        "target_coverage": TWO_COV_TARGET_COVERAGE,
        "achieved_coverage": achieved_coverage,
        "leaky_lambda": float(leaky_model.multiplier),
        "leaky_width_k": float(leaky_width.mean()),
        "n_instances": int(len(matched)),
    }])
    write_result(
        results_df,
        name="phase3_leaky_lambda_at_two_cov_coverage",
        model_version="phase3-leaky-lambda-at-two-cov-coverage-v1",
        config={"target_coverage": TWO_COV_TARGET_COVERAGE},
    )


if __name__ == "__main__":
    main()
