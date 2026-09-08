"""Recompute rich summary metrics (CRPS, MAE, twCRPS, interval width, coverage,
reliability) from the already-persisted Phase 2 per-instance results — does NOT
re-run the GPU job. See docs/phase2_results_schema.md for the real input schema.

Real schema confirmed on the server (docs/phase2_results_schema.md):
  columns: method, station_id, valid_time, step_hours, obs, q0.1..q0.9
  method values: raw_ensemble, emos, tsfm3 (NOT "timesfm3" as originally guessed)
  observation column: obs (NOT "t2m_obs" as originally guessed)
"""
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.calibration import empirical_coverage, pit_values, reliability_index
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles, mae_from_quantiles, twcrps_from_quantiles
from zeropp.eval.significance import block_bootstrap_skill_score_ci, station_blocked_paired_test

RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"


def per_instance_crps(df_method: pd.DataFrame, quantile_levels: list[float]) -> np.ndarray:
    quantile_cols = [f"q{q}" for q in quantile_levels]
    quantile_preds = df_method[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    obs = df_method["obs"].to_numpy().reshape(-1, 1)
    return crps_from_quantiles(obs, quantile_preds, quantile_levels).flatten()


def summarize_method(df_method: pd.DataFrame, quantile_levels: list[float]) -> dict:
    quantile_cols = [f"q{q}" for q in quantile_levels]
    quantile_preds = df_method[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    obs = df_method["obs"].to_numpy().reshape(-1, 1)

    interval_width = float(np.mean(quantile_preds[..., -1] - quantile_preds[..., 0]))
    coverage_80 = empirical_coverage(obs, quantile_preds, quantile_levels, lower=0.1, upper=0.9)
    pit = pit_values(obs, quantile_preds, quantile_levels)

    return {
        "crps": float(crps_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
        "mae": float(mae_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
        "twcrps": float(twcrps_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
        "interval_width_p10_p90_kelvin": interval_width,
        "coverage_80pct_nominal": coverage_80,
        "reliability_index": reliability_index(pit.flatten()),
        "n_instances": len(df_method),
    }


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    raw = pd.read_parquet(RAW_RESULTS_PATH)

    rows = []
    for method, df_method in raw.groupby("method"):
        summary = summarize_method(df_method, quantile_levels)
        summary["method"] = method
        rows.append(summary)

    summary_df = pd.DataFrame(rows)

    print("\n===== Phase 3 corrected summary (from persisted Phase 2 results) =====")
    print("NOTE: 'coverage_80pct_nominal' measures the [q0.1, q0.9] interval, which is the")
    print("80%-nominal band, NOT 90% (empirical_coverage(lower=0.1, upper=0.9) cannot compute")
    print("CLAUDE.md's literal 'nominal 90%' ask, which would need q0.05/q0.95 — not available")
    print("on this project's fixed 9-level [0.1,...,0.9] quantile grid).")
    print("LIMITATION: TimesFM-3's quantile output spans only p10-p90 (no tail beyond) — twcrps")
    print("(tail-weighted at q0.8/q0.9) is therefore not a fully fair comparison against EMOS's")
    print("full Gaussian tail or raw ensemble's empirical 51-member tail.")
    print(summary_df.to_string(index=False))

    write_result(
        summary_df,
        name="phase3_summary_metrics",
        model_version="phase3-summary-v1",
        config={"quantile_levels": quantile_levels, "source": RAW_RESULTS_PATH},
    )

    # Statistical significance: block bootstrap CI + station-blocked paired test.
    # Real confirmed method values are raw_ensemble / emos / tsfm3 (see
    # docs/phase2_results_schema.md), NOT the brief's guessed "timesfm3".
    raw_ens = raw[raw["method"] == "raw_ensemble"].reset_index(drop=True)
    emos = raw[raw["method"] == "emos"].reset_index(drop=True)
    tsfm = raw[raw["method"] == "tsfm3"].reset_index(drop=True)

    crps_raw = per_instance_crps(raw_ens, quantile_levels)
    crps_emos = per_instance_crps(emos, quantile_levels)
    crps_tsfm = per_instance_crps(tsfm, quantile_levels)
    stations = tsfm["station_id"].to_numpy()

    # Final-review fix (minor item): both seed=0 literals below were CLAUDE.md
    # rule-5-violating hardcoded seeds (the rule is scoped to src/ by its literal
    # text, but this script's bootstrap seed is exactly the kind of hidden
    # randomness that rule exists to prevent) -- routed through
    # config.seeds[0] instead, so the bootstrap's seed is configurable the same
    # way every other seed in this project is.
    bootstrap_seed = config.seeds[0]

    print("\n===== Significance: TimesFM-3 vs raw ensemble =====")
    point, lo, hi = block_bootstrap_skill_score_ci(crps_tsfm, crps_raw, stations, seed=bootstrap_seed)
    print(f"CRPS skill score (block bootstrap, 95% CI): {point:.4f} [{lo:.4f}, {hi:.4f}]")
    test_result = station_blocked_paired_test(crps_raw, crps_tsfm, stations)
    print(f"Station-blocked paired test (n={test_result['n_blocks']} stations): "
          f"mean diff={test_result['block_mean_diff']:.4f}, "
          f"t p-value={test_result['t_pvalue']:.4f}, wilcoxon p-value={test_result['wilcoxon_pvalue']:.4f}")

    print("\n===== Significance: TimesFM-3 vs EMOS =====")
    point, lo, hi = block_bootstrap_skill_score_ci(crps_tsfm, crps_emos, stations, seed=bootstrap_seed)
    print(f"CRPS skill score (block bootstrap, 95% CI): {point:.4f} [{lo:.4f}, {hi:.4f}]")
    test_result = station_blocked_paired_test(crps_emos, crps_tsfm, stations)
    print(f"Station-blocked paired test (n={test_result['n_blocks']} stations): "
          f"mean diff={test_result['block_mean_diff']:.4f}, "
          f"t p-value={test_result['t_pvalue']:.4f}, wilcoxon p-value={test_result['wilcoxon_pvalue']:.4f}")


if __name__ == "__main__":
    main()
