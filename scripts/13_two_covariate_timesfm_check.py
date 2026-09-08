"""BL-4 / B4 (Introduction review, 2026-09-07): does TimesFM-3's single
`past_future_covariates` parameter accept a second, simultaneous covariate
(ensemble spread alongside ensemble mean)?

This is a real, previously-unresolved question, not a confirmed API limit.
`zeropp.models.tsfm_timesfm.TimesFM3` was verified (session of
2026-09-03, `docs/superpowers/plans/2026-09-03-zeropp-phase2-real-slice.md`)
to expose exactly one PARAMETER named `past_future_covariates` -- that
verification never tested whether the parameter's array can be
multi-channel (shape (T, 2)) to carry two covariates at once. TimesFM 3.0's
own release documentation (Google Research model card + release notes,
cited in paper/sections/03_methods.tex) describes native multivariate
forecasting and flexible covariate support, consistent with (but not proof
of) a multi-channel-capable single parameter. `TimesFM3.predict_quantiles`
now attempts this (stacks mean+spread into a (T, 2) array) when
`past_future_ens_spread` is provided -- see its class docstring and
`tests/test_tsfm_timesfm.py` for what this project's own code does with the
shape; what the REAL server-side model does with it has never been checked.
This script checks it.

Two phases, in order:

1. SMOKE TEST (cheap, ~3 groups, run first): call TimesFM3.predict_quantiles
   with both covariates for a handful of real instances. If the underlying
   `timesfm` call raises (e.g. a shape/dimension error), this is caught,
   logged clearly to results/phase3_two_covariate_timesfm_smoketest.json,
   and the script STOPS -- do not burn a multi-hour GPU run on a call shape
   the server has already rejected. If it succeeds, print the returned
   quantile shape and a coherence check (does the covariate-with-spread
   forecast actually differ numerically from a mean-only forecast on the
   same instances -- if it's bit-identical, the second channel was likely
   silently ignored server-side, not genuinely used, and that is reported
   as its own finding, not treated as success).

2. FULL RUN (only if the smoke test passes and shows a real numeric
   difference): the same (station, issue_time) grouping and CONTEXT_LENGTH
   discipline as scripts/02_run_tsfm.py, with an added past-future ensemble
   STANDARD DEVIATION covariate (sqrt of the ensemble variance already
   computed for EMOS/DRN/variance-inflation elsewhere in this project),
   built the same "freshest available NWP guidance" way as the mean
   covariate's past portion. Predictions are restricted to the SAME matched
   instance set already established in results/phase2_comparison_raw.parquet
   (737,809 instances, tsfm3 rows) via the same instance-set join used in
   scripts/07/09/14 -- if adding the spread covariate's own coverage
   requirement shrinks that set further, this is asserted and reported
   explicitly, not silently accepted as "the same 737,809."

Output:
  results/phase3_two_covariate_timesfm_smoketest.json -- always written.
  results/phase3_two_covariate_timesfm.parquet -- only if the full run
    proceeds. Columns: crps, coverage_80pct, interval_width_k, n_instances,
    plus the matched single-covariate (mean-only) TimesFM-3 numbers on the
    identical instance set for a direct, apples-to-apples comparison.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.tsfm_timesfm import TimesFM3

CONTEXT_LENGTH = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW_RESULTS_PATH = Path("results/phase2_comparison_raw.parquet")
SMOKETEST_PATH = Path("results/phase3_two_covariate_timesfm_smoketest.json")
N_SMOKETEST_GROUPS = 3


def _freshest_lookup(station_df: pd.DataFrame, agg: str) -> pd.Series:
    """Generalizes scripts/02_run_tsfm.py's _freshest_ens_mean_lookup to any
    across-member aggregate ("mean" or "std") of t2m_forecast: per-valid_time
    aggregate at the SHORTEST available step_hours for that valid_time (the
    freshest NWP guidance available), matching the mean covariate's own
    "freshest available" convention so both channels are built the same way."""
    min_step = station_df.groupby("valid_time")["step_hours"].transform("min")
    freshest_rows = station_df[station_df["step_hours"] == min_step]
    grouped = freshest_rows.groupby("valid_time")["t2m_forecast"]
    return grouped.mean() if agg == "mean" else grouped.std()


def _build_instances(test_df: pd.DataFrame, limit_groups: int | None = None):
    """Shared instance-building loop for both the smoke test and the full
    run -- (station, issue_time) grouping identical to scripts/02_run_tsfm.py,
    with a second, ensemble-STD past-future covariate added alongside mean.
    Yields dicts, one per (station, valid_time, step_hours) instance."""
    n_yielded_groups = 0
    for station_id, station_df in test_df.groupby("station_id"):
        if limit_groups is not None and n_yielded_groups >= limit_groups:
            return
        station_df = station_df.sort_values("valid_time")

        obs_lookup = (
            station_df.drop_duplicates("valid_time")[["valid_time", "t2m_obs"]]
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        obs_times = obs_lookup["valid_time"].to_numpy()
        obs_vals = obs_lookup["t2m_obs"].to_numpy()

        freshest_mean = _freshest_lookup(station_df, "mean")
        freshest_std = _freshest_lookup(station_df, "std")
        mean_vals = freshest_mean.reindex(obs_lookup["valid_time"]).to_numpy()
        std_vals = freshest_std.reindex(obs_lookup["valid_time"]).to_numpy()

        for issue_time, group in station_df.groupby("issue_time"):
            idx = np.searchsorted(obs_times, np.datetime64(issue_time), side="left")
            if idx < CONTEXT_LENGTH:
                continue

            past_mean = mean_vals[idx - CONTEXT_LENGTH: idx]
            past_std = std_vals[idx - CONTEXT_LENGTH: idx]
            if np.isnan(past_mean).any() or np.isnan(past_std).any():
                continue
            context = obs_vals[idx - CONTEXT_LENGTH: idx]

            by_lead = (
                group.groupby("step_hours")
                .agg(
                    ens_mean=("t2m_forecast", "mean"),
                    ens_std=("t2m_forecast", "std"),
                    obs=("t2m_obs", "first"),
                    valid_time=("valid_time", "first"),
                )
                .sort_index()
            )
            future_mean = by_lead["ens_mean"].to_numpy()
            future_std = by_lead["ens_std"].to_numpy()
            horizon = len(future_mean)
            if horizon == 0 or np.isnan(future_std).any():
                continue

            mean_covariate = np.concatenate([past_mean, future_mean])
            std_covariate = np.concatenate([past_std, future_std])

            yield {
                "station_id": station_id,
                "context": context,
                "mean_covariate": mean_covariate,
                "std_covariate": std_covariate,
                "horizon": horizon,
                "by_lead": by_lead,
            }
            n_yielded_groups += 1
            if limit_groups is not None and n_yielded_groups >= limit_groups:
                return


# Distinguishability threshold, in the same units as quantile output (K).
# Well above floating-point noise (~1e-7 for this model's numerics) and well
# below anything that would matter for a reported CRPS difference elsewhere
# in this paper (differences of ~0.01-0.1 K move CRPS at the third decimal).
# A pair of outputs closer than this is treated as "the same," not merely
# "very similar."
DISTINGUISH_ATOL_K = 1e-3


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run_smoketest(quantile_levels: list[float], seed: int) -> bool:
    """T1 (channel-effectiveness ablation, per the 2026-09-07 review's more
    rigorous design than a plain mean-only-vs-two-covariate diff): for each
    of a few real instances, calls TimesFM3 four ways -- mean only, mean +
    REAL spread, mean + PERMUTED spread (same values, shuffled along time,
    destroying temporal structure but keeping the marginal distribution),
    and mean + CONSTANT spread (same overall magnitude, no information at
    all) -- and checks that the real-spread output is genuinely
    distinguishable from all three alternatives, not just "not bit-identical
    to mean-only." If real ends up indistinguishable from permuted and/or
    constant, the model is reacting to the channel's mere presence or scale,
    not its actual informational content, and the second channel is not
    usable this way even though the API accepts the shape -- reported as a
    negative finding, not treated as success. Returns True iff the full run
    should proceed."""
    print(f"Running T1 channel-ablation smoke test ({N_SMOKETEST_GROUPS} groups)...")
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")
    rng = np.random.default_rng(seed)

    # One shared, frozen model instance for all four calls per group -- it
    # is stateless (zero-shot, no fit()), so there is no reason to load the
    # weights more than once.
    model = TimesFM3(quantile_levels=quantile_levels, device=DEVICE)

    result = {
        "n_groups_attempted": 0, "call_succeeded": None, "error": None,
        "per_group_diffs": [], "distinguishable_from_permuted": None,
        "distinguishable_from_constant": None, "distinguishable_from_mean_only": None,
    }
    diffs_real_vs_mean, diffs_real_vs_permuted, diffs_real_vs_constant = [], [], []
    try:
        for inst in _build_instances(test_df, limit_groups=N_SMOKETEST_GROUPS):
            result["n_groups_attempted"] += 1
            std_real = inst["std_covariate"]
            std_permuted = rng.permutation(std_real)
            std_constant = np.full_like(std_real, fill_value=float(np.mean(std_real)))

            def _call(spread=None):
                X = {
                    "context": [inst["context"]],
                    "past_future_ens_mean": [inst["mean_covariate"]],
                    "horizon": inst["horizon"],
                }
                if spread is not None:
                    X["past_future_ens_spread"] = [spread]
                return model.predict_quantiles(X)

            pred_mean_only = _call()
            pred_real = _call(std_real)
            pred_permuted = _call(std_permuted)
            pred_constant = _call(std_constant)

            d_mean = _mean_abs_diff(pred_real, pred_mean_only)
            d_perm = _mean_abs_diff(pred_real, pred_permuted)
            d_const = _mean_abs_diff(pred_real, pred_constant)
            diffs_real_vs_mean.append(d_mean)
            diffs_real_vs_permuted.append(d_perm)
            diffs_real_vs_constant.append(d_const)
            result["per_group_diffs"].append({
                "station_id": str(inst["station_id"]),
                "real_vs_mean_only_K": d_mean, "real_vs_permuted_K": d_perm, "real_vs_constant_K": d_const,
            })
        result["call_succeeded"] = True
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this IS the check
        result["call_succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"T1 SMOKE TEST FAILED (API call itself raised): {result['error']}")
        SMOKETEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        SMOKETEST_PATH.write_text(json.dumps(result, indent=2, default=str))
        return False

    dist_mean = max(diffs_real_vs_mean) > DISTINGUISH_ATOL_K
    dist_perm = max(diffs_real_vs_permuted) > DISTINGUISH_ATOL_K
    dist_const = max(diffs_real_vs_constant) > DISTINGUISH_ATOL_K
    result["distinguishable_from_mean_only"] = bool(dist_mean)
    result["distinguishable_from_permuted"] = bool(dist_perm)
    result["distinguishable_from_constant"] = bool(dist_const)

    print(
        f"T1: max|real-mean_only|={max(diffs_real_vs_mean):.6f} K, "
        f"max|real-permuted|={max(diffs_real_vs_permuted):.6f} K, "
        f"max|real-constant|={max(diffs_real_vs_constant):.6f} K "
        f"(threshold={DISTINGUISH_ATOL_K} K)"
    )
    passed = dist_mean and dist_perm and dist_const
    if not passed:
        print(
            "T1 FAILED: real spread is not distinguishable from at least one ablation "
            "(mean-only / permuted / constant) -- the model is not genuinely using the "
            "second channel's actual information content. Treating this as a NEGATIVE "
            "finding, not proceeding to the full run."
        )
    else:
        print("T1 PASSED: real spread is distinguishable from all three ablations -- proceeding to full run.")

    SMOKETEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKETEST_PATH.write_text(json.dumps(result, indent=2, default=str))
    return passed


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)

    if not run_smoketest(quantile_levels, seed=config.seeds[0]):
        print("Stopping before the full run -- see results/phase3_two_covariate_timesfm_smoketest.json")
        return

    print("Smoke test passed. Running full evaluation on the matched instance set...")
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")
    model = TimesFM3(quantile_levels=quantile_levels, device=DEVICE)

    raw = pd.read_parquet(RAW_RESULTS_PATH)
    canonical_keys = raw.loc[raw["method"] == "tsfm3", ["station_id", "valid_time", "step_hours"]].drop_duplicates()

    two_cov_preds, obs_values = [], []
    meta_station, meta_valid_time, meta_step_hours = [], [], []
    n_groups = 0
    for inst in _build_instances(test_df):
        n_groups += 1
        pred = model.predict_quantiles({
            "context": [inst["context"]],
            "past_future_ens_mean": [inst["mean_covariate"]],
            "past_future_ens_spread": [inst["std_covariate"]],
            "horizon": inst["horizon"],
        })
        by_lead = inst["by_lead"]
        for i, step_hours in enumerate(by_lead.index):
            two_cov_preds.append(pred[0, i])
            obs_values.append(by_lead["obs"].iloc[i])
            meta_station.append(inst["station_id"])
            meta_valid_time.append(by_lead["valid_time"].iloc[i])
            meta_step_hours.append(step_hours)

    pred_df = pd.DataFrame({
        "station_id": meta_station, "valid_time": meta_valid_time, "step_hours": meta_step_hours,
        "obs": obs_values,
    })
    for qi, q in enumerate(quantile_levels):
        pred_df[f"q{q}"] = [p[qi] for p in two_cov_preds]

    matched = canonical_keys.merge(pred_df, on=["station_id", "valid_time", "step_hours"], how="inner")
    n_matched = len(matched)
    n_canonical = len(canonical_keys)
    print(f"Instance-set join: canonical tsfm3 instances={n_canonical}, matched={n_matched}")
    if n_matched != n_canonical:
        print(
            f"WARNING: two-covariate run covers {n_matched}/{n_canonical} of the canonical matched "
            "instance set -- the added spread-covariate coverage requirement excluded some instances "
            "the mean-only run did not. Report this honestly; do not claim 'the same 737,809 instances' "
            "if this number is smaller."
        )

    y = matched["obs"].to_numpy().reshape(-1, 1)
    qp = matched[[f"q{q}" for q in quantile_levels]].to_numpy().reshape(-1, 1, len(quantile_levels))
    crps = float(crps_from_quantiles(y, qp, quantile_levels).mean())
    coverage = float(empirical_coverage(y, qp, quantile_levels, lower=0.1, upper=0.9))
    width = float(np.mean(qp[:, 0, hi_idx] - qp[:, 0, lo_idx]))

    tsfm3_mean_only = raw[raw["method"] == "tsfm3"].merge(
        matched[["station_id", "valid_time", "step_hours"]], on=["station_id", "valid_time", "step_hours"], how="inner"
    )
    y_m = tsfm3_mean_only["obs"].to_numpy().reshape(-1, 1)
    qp_m = tsfm3_mean_only[[f"q{q}" for q in quantile_levels]].to_numpy().reshape(-1, 1, len(quantile_levels))
    crps_m = float(crps_from_quantiles(y_m, qp_m, quantile_levels).mean())
    coverage_m = float(empirical_coverage(y_m, qp_m, quantile_levels, lower=0.1, upper=0.9))
    width_m = float(np.mean(qp_m[:, 0, hi_idx] - qp_m[:, 0, lo_idx]))

    print(
        f"two-covariate (mean+spread): crps={crps:.4f}, coverage_80pct={coverage:.4f}, width={width:.4f}\n"
        f"mean-only (original, same instances): crps={crps_m:.4f}, coverage_80pct={coverage_m:.4f}, width={width_m:.4f}"
    )

    results_df = pd.DataFrame([
        {"method": "tsfm3_two_covariate", "crps": crps, "coverage_80pct": coverage, "interval_width_k": width, "n_instances": n_matched},
        {"method": "tsfm3_mean_only_same_instances", "crps": crps_m, "coverage_80pct": coverage_m, "interval_width_k": width_m, "n_instances": n_matched},
    ])
    write_result(
        results_df,
        name="phase3_two_covariate_timesfm",
        model_version="phase3-two-covariate-v1",
        config={
            "quantile_levels": quantile_levels,
            "context_length": CONTEXT_LENGTH,
            "n_canonical_matched_instances": int(n_canonical),
            "n_matched_instances_this_run": int(n_matched),
        },
    )


if __name__ == "__main__":
    main()
