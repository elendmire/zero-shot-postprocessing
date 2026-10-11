"""TimesFM-3 with the ensemble spread as a second covariate.
A check on a few instances (real, permuted and constant spread) decides whether the second channel is used; if
so, the full evaluation runs on the instances of run_timesfm.py and is compared with mean-only TimesFM-3.
Usage: python scripts/run_timesfm_spread_covariate.py --context-length 160 --out spread_covariate
Output: results/spread_covariate.parquet and .json, spread_covariate_smoketest.json, spread_covariate_raw.parquet
(per-instance quantiles; not in the repository). Scores enter numbers.tex.
"""
import argparse
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
RAW_RESULTS_PATH = Path("results/predictions_main.parquet")
SMOKETEST_PATH = Path("results/spread_covariate_smoketest.json")
N_SMOKETEST_GROUPS = 3


def _freshest_lookup(station_df: pd.DataFrame, agg: str) -> pd.Series:
    """Member mean or standard deviation at the shortest available lead of each valid time."""
    min_step = station_df.groupby("valid_time")["step_hours"].transform("min")
    freshest_rows = station_df[station_df["step_hours"] == min_step]
    grouped = freshest_rows.groupby("valid_time")["t2m_forecast"]
    return grouped.mean() if agg == "mean" else grouped.std()


def _build_instances(test_df: pd.DataFrame, limit_groups: int | None = None, context_length: int | None = None):
    """Yield one dict per (station, issue time): context, mean and spread covariates, horizon and per-lead table."""
    CONTEXT = CONTEXT_LENGTH if context_length is None else context_length
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
            if idx < CONTEXT:
                continue

            past_mean = mean_vals[idx - CONTEXT: idx]
            past_std = std_vals[idx - CONTEXT: idx]
            if np.isnan(past_mean).any() or np.isnan(past_std).any():
                continue
            context = obs_vals[idx - CONTEXT: idx]

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


# quantile outputs closer than this (K) are treated as identical
DISTINGUISH_ATOL_K = 1e-3


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run_smoketest(quantile_levels: list[float], seed: int) -> bool:
    """Check on a few instances that the spread covariate changes the forecast.

    TimesFM-3 is called with the mean only, with the real spread, with a permuted spread and with a constant
    spread. Returns True if the real-spread forecast differs from all three, i.e. the second channel is used."""
    print(f"Running spread-covariate check on {N_SMOKETEST_GROUPS} groups...")
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")
    rng = np.random.default_rng(seed)

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
    except Exception as exc:  # noqa: BLE001
        result["call_succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Spread-covariate check failed: {result['error']}")
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
        f"max|real-mean_only|={max(diffs_real_vs_mean):.6f} K, "
        f"max|real-permuted|={max(diffs_real_vs_permuted):.6f} K, "
        f"max|real-constant|={max(diffs_real_vs_constant):.6f} K "
        f"(threshold={DISTINGUISH_ATOL_K} K)"
    )
    passed = dist_mean and dist_perm and dist_const
    if not passed:
        print("Real spread is not distinguishable from at least one of the alternatives; not running the full evaluation.")
    else:
        print("Real spread is distinguishable from all three alternatives.")

    SMOKETEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKETEST_PATH.write_text(json.dumps(result, indent=2, default=str))
    return passed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    ap.add_argument("--out", default="spread_covariate")
    ap.add_argument("--skip-smoketest", action="store_true")
    ap.add_argument("--common", default=None,
                    help="parquet whose (station_id, valid_time, step_hours) keys define the evaluation set "
                         "(default: tsfm3 rows of results/predictions_main.parquet)")
    ap.add_argument("--baseline", default=None,
                    help="parquet with the 1-covariate run (method=tsfm3 rows if a 'method' column exists); "
                         "default: results/predictions_main.parquet")
    args = ap.parse_args()
    context_length = args.context_length
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)

    if not args.skip_smoketest and not run_smoketest(quantile_levels, seed=config.seeds[0]):
        print("Stopping; see results/spread_covariate_smoketest.json")
        return

    print("Running the full evaluation on the matched instance set...")
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")
    model = TimesFM3(quantile_levels=quantile_levels, device=DEVICE)

    raw = pd.read_parquet(args.baseline or RAW_RESULTS_PATH)
    raw = raw[raw["method"] == "tsfm3"] if "method" in raw.columns else raw
    raw = raw.assign(method="tsfm3")
    KEYS = ["station_id", "valid_time", "step_hours"]
    canonical_keys = (pd.read_parquet(args.common, columns=KEYS) if args.common
                      else raw[KEYS]).drop_duplicates()
    raw = raw.merge(canonical_keys, on=KEYS, how="inner")

    two_cov_preds, obs_values = [], []
    meta_station, meta_valid_time, meta_step_hours = [], [], []
    n_groups = 0
    for inst in _build_instances(test_df, context_length=context_length):
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
    print(f"matched instances: {n_matched} of {n_canonical}")
    if n_matched != n_canonical:
        print(f"Warning: the two-covariate run covers only {n_matched}/{n_canonical} of the matched instances")

    Path("results").mkdir(exist_ok=True)
    matched.assign(method="tsfm3_two_covariate").to_parquet(Path("results") / f"{args.out}_raw.parquet", index=False)
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
        name=args.out,
        model_version="spread-covariate",
        config={
            "quantile_levels": quantile_levels,
            "context_length": context_length,
            "n_canonical_matched_instances": int(n_canonical),
            "n_matched_instances_this_run": int(n_matched),
        },
    )


if __name__ == "__main__":
    main()
