"""Per-lead-time CRPS and coverage breakdown, from the already-persisted Phase 2
results — no GPU re-run. Produces a real figure (not a notebook).

Real schema confirmed on the server (docs/phase2_results_schema.md):
  columns: method, station_id, valid_time, step_hours, obs, q0.1..q0.9
  method values: raw_ensemble, emos, tsfm3 (NOT "timesfm3" as originally guessed)
  observation column: obs (NOT "t2m_obs" as originally guessed)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles

RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"
FIGURE_PATH = "figures/lead_time_breakdown.png"


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    raw = pd.read_parquet(RAW_RESULTS_PATH)

    rows = []
    for (method, step_hours), group in raw.groupby(["method", "step_hours"]):
        quantile_preds = group[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
        obs = group["obs"].to_numpy().reshape(-1, 1)
        rows.append({
            "method": method,
            "step_hours": step_hours,
            "crps": float(crps_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
            "coverage_80pct": empirical_coverage(obs, quantile_preds, quantile_levels, lower=0.1, upper=0.9),
            "n_instances": len(group),
        })

    breakdown_df = pd.DataFrame(rows).sort_values(["method", "step_hours"])
    print(breakdown_df.to_string(index=False))

    write_result(
        breakdown_df,
        name="phase3_lead_time_breakdown",
        model_version="phase3-lead-breakdown-v1",
        config={"quantile_levels": quantile_levels, "source": RAW_RESULTS_PATH},
    )

    fig, (ax_crps, ax_cov) = plt.subplots(1, 2, figsize=(12, 5))
    for method, group in breakdown_df.groupby("method"):
        ax_crps.plot(group["step_hours"], group["crps"], marker="o", label=method)
        ax_cov.plot(group["step_hours"], group["coverage_80pct"], marker="o", label=method)

    ax_crps.set_xlabel("Lead time (hours)")
    ax_crps.set_ylabel("CRPS")
    ax_crps.set_title("CRPS vs. lead time")
    ax_crps.legend()

    ax_cov.axhline(0.80, color="gray", linestyle="--", label="nominal 80%")
    ax_cov.set_xlabel("Lead time (hours)")
    ax_cov.set_ylabel("Coverage (p10-p90 band)")
    ax_cov.set_title("Coverage vs. lead time")
    ax_cov.legend()

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
