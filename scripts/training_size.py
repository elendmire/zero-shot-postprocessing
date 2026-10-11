"""Training size: EMOS, DRN and variance inflation fit on increasing amounts of reforecast data.
A case is one (year_idx, time_idx) pair of the archive; k cases span k * DAYS_PER_CASE days. Cases are the k most
recent (contiguous) or k random ones (5 seeds). Methods: pooled and local EMOS, DRN, variance inflation, and
the raw ensemble, TimesFM-3 and fixed variance inflation as reference lines, all on the instances of
predictions_main.parquet.
Output (results/): training_size, training_size_small (k = 1 ... 7), training_size_breakpoints,
training_size_small_breakpoints (size at which CRPS or coverage crosses TimesFM-3), training_size_significance.
Used for Fig. 3 and Table A1; the helper functions are reused by other scripts.
"""
import time
import warnings

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles, twcrps_from_quantiles
from zeropp.eval.significance import station_blocked_paired_test
from zeropp.models.drn import DRN
from zeropp.models.emos import EMOS
from zeropp.models.variance_inflation import VarianceInflationBaseline

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/predictions_main.parquet"

# The archive has 20 analogue years x 209 issue-date labels. After dropping the 108 cases that verify in the
# test period, 4072 cases remain, spanning 1997-01-02 to 2016-12-25 (7297 days over 4071 intervals).
REFORECAST_TOTAL_ISSUE_DATES = 209
REFORECAST_KEPT_CASES = 4072
REFORECAST_CASE_SPAN_DAYS = 7297.0
DAYS_PER_CASE = REFORECAST_CASE_SPAN_DAYS / (REFORECAST_KEPT_CASES - 1)  # measured ~1.7924
LOCAL_EMOS_MIN_ROWS = 5

# k grid of training_size_small
LOW_N_K_GRID = [1, 2, 3, 5, 7]

# Fixed variance-inflation multiplier (1 / spread-skill ratio of about 0.67; Buizza 1997, Mon. Wea. Rev.
# 125, 99-119; Fortin et al. 2014, J. Hydrometeor. 15, 1708-1713); no fitting.
LAMBDA_CLIM = 1.5

# sort cases from the most recent to the oldest
CHRONOLOGICAL_DESCENDING = True


def n_days_to_k(n_days) -> int | str:
    """Case count k for a label of N days (or 'full')."""
    if n_days == "full":
        return "full"
    return round(n_days / DAYS_PER_CASE)


def k_to_calendar_days(k: int) -> int:
    """Calendar days spanned by k cases."""
    return round(k * DAYS_PER_CASE)


def bp_to_calendar_days(bp: float | None) -> int | None:
    """Calendar days of an interpolated breakpoint k (rounded once)."""
    if bp is None:
        return None
    return round(bp * DAYS_PER_CASE)


def n_days_for_exact_k(k: int) -> float:
    """Number of days for which n_days_to_k gives exactly k cases."""
    return k * DAYS_PER_CASE


def k_and_calendar_days(n_days, n_pairs_full: int) -> tuple[int, int | None]:
    """(k, calendar days) for a label of N days; 'full' uses all cases and has no calendar-day equivalent."""
    if n_days == "full":
        return n_pairs_full, None
    k = n_days_to_k(n_days)
    return k, k_to_calendar_days(k)


def sample_contiguous(full_train: pd.DataFrame, n_days) -> pd.DataFrame:
    """The k cases closest in time to the test period, going backwards."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx", "case_date"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    ordered = unique_pairs.sort_values("case_date", ascending=not CHRONOLOGICAL_DESCENDING)
    sampled_pairs = ordered.iloc[:k][["year_idx", "time_idx"]]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def sample_random(full_train: pd.DataFrame, n_days, seed: int) -> pd.DataFrame:
    """k cases drawn at random from the archive."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(unique_pairs), size=k, replace=False)
    sampled_pairs = unique_pairs.iloc[sampled_idx]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def compute_metrics(y: np.ndarray, preds: np.ndarray, quantile_levels: list[float]) -> dict:
    """CRPS, qwCRPS (upper quantile levels 0.8 and 0.9), 80 % coverage and mean 10-90 % width (K).

    y has shape (n, 1), preds (n, 1, n_quantiles)."""
    crps = float(crps_from_quantiles(y, preds, quantile_levels).mean())
    twcrps = float(twcrps_from_quantiles(y, preds, quantile_levels).mean())
    coverage = float(empirical_coverage(y, preds, quantile_levels, lower=0.1, upper=0.9))
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    width = float(np.mean(preds[:, 0, hi_idx] - preds[:, 0, lo_idx]))
    return {"crps": crps, "twcrps": twcrps, "coverage_80pct": coverage, "interval_width_k": width}


def fit_predict_pooled_emos(train_subset: pd.DataFrame, quantile_levels: list[float], test_X: dict) -> np.ndarray:
    train_for_emos = train_subset[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
    model = EMOS(quantile_levels=quantile_levels).fit(train_for_emos)
    return model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})


def fit_predict_local_emos(
    train_subset: pd.DataFrame,
    test_station_ids: np.ndarray,
    test_X: dict,
    quantile_levels: list[float],
    min_rows: int = LOCAL_EMOS_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One EMOS per station. Test rows of stations with fewer than min_rows training rows are left as NaN
    (covered_mask is False); coverage_fraction is the share of test stations that could be fitted."""
    n = len(test_station_ids)
    preds = np.full((n, 1, len(quantile_levels)), np.nan)
    covered_mask = np.zeros(n, dtype=bool)
    counts = train_subset["station_id"].value_counts()
    fittable_stations = counts[counts >= min_rows].index
    for sid in fittable_stations:
        station_train = train_subset[train_subset["station_id"] == sid][
            ["station_id", "ens_mean", "ens_var", "t2m_obs"]
        ]
        model = EMOS(quantile_levels=quantile_levels).fit(station_train)
        rows_mask = test_station_ids == sid
        if not rows_mask.any():
            continue
        station_preds = model.predict_quantiles(
            {"ens_mean": test_X["ens_mean"][rows_mask], "ens_var": test_X["ens_var"][rows_mask]}
        )
        preds[rows_mask] = station_preds
        covered_mask[rows_mask] = True
    n_unique_test_stations = len(np.unique(test_station_ids))
    n_covered_test_stations = len(np.unique(test_station_ids[covered_mask])) if covered_mask.any() else 0
    coverage_fraction = float(n_covered_test_stations) / float(n_unique_test_stations or 1)
    return preds, covered_mask, coverage_fraction


def find_breakpoint(n_cases_axis: list[int], method_values: list[float], reference_value: float, better: str = "lower") -> float | None:
    """Linearly interpolated number of cases at which method_values crosses reference_value, or None.

    better="lower": the method starts above the reference and falls below it (CRPS);
    otherwise it starts below and rises above (coverage)."""
    pairs = sorted(zip(n_cases_axis, method_values), key=lambda p: p[0])
    for (n0, v0), (n1, v1) in zip(pairs, pairs[1:]):
        if v0 is None or v1 is None or (isinstance(v0, float) and np.isnan(v0)) or (isinstance(v1, float) and np.isnan(v1)):
            continue
        starts_worse = (v0 > reference_value) if better == "lower" else (v0 < reference_value)
        ends_better = (v1 <= reference_value) if better == "lower" else (v1 >= reference_value)
        if starts_worse and ends_better and v1 != v0:
            frac = (reference_value - v0) / (v1 - v0)
            return n0 + frac * (n1 - n0)
    return None


def _real_pairs(n_cases_axis: list[float], method_values: list[float]) -> list[tuple[float, float]]:
    """Non-NaN (n_cases, value) pairs sorted by n_cases."""
    return sorted(
        (
            (n, v) for n, v in zip(n_cases_axis, method_values)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ),
        key=lambda p: p[0],
    )


def breakpoint_and_direction(
    metric: str, n_cases_axis: list[float], method_values: list[float], reference_value: float
) -> tuple[float | None, str]:
    """find_breakpoint with the direction rule of each metric (crps, coverage_80pct).

    Returns the breakpoint (or None) and a text saying whether the curve crossed the reference, started
    ahead of it or never reached it. Coverage direction follows the sign of the overall trend.
    interval_width_k is not handled, since sharpness has no direction on its own."""
    real = _real_pairs(n_cases_axis, method_values)
    if not real:
        return None, "no real (non-NaN) data points to evaluate"
    smallest = real[0][1]

    if metric == "crps":
        bp = find_breakpoint(n_cases_axis, method_values, reference_value, better="lower")
        if bp is not None:
            return bp, "value crossed reference downward (CRPS improved past the tsfm3 reference)"
        if smallest <= reference_value:
            return None, "already better than (or equal to) the reference at the smallest tested N — no crossing needed"
        return None, "no crossing observed in tested range — remains worse than the reference throughout"

    elif metric == "coverage_80pct":
        if len(real) < 2:
            return None, "insufficient real data points to determine an overall trend direction"
        xs = np.array([p[0] for p in real], dtype=float)
        ys = np.array([p[1] for p in real], dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])
        better = "higher" if slope > 0 else "lower"
        bp = find_breakpoint(n_cases_axis, method_values, reference_value, better=better)
        if bp is not None:
            word = "upward" if better == "higher" else "downward"
            return bp, (
                f"value crossed reference {word} (overall trend slope={slope:.6f}; coverage "
                f"{'rose to' if better == 'higher' else 'fell to'} match the tsfm3 reference's calibration)"
            )
        already_matched = (better == "higher" and smallest >= reference_value) or (better == "lower" and smallest <= reference_value)
        if already_matched:
            return None, f"already at/beyond the reference at the smallest tested N (overall trend slope={slope:.6f}) — no crossing needed"
        return None, f"no crossing observed in tested range (overall trend slope={slope:.6f}) — remains on the same side of the reference throughout"

    else:
        raise ValueError(f"no direction rule for metric {metric!r}")


def _catch_overflow_warnings(fn, *args, **kwargs):
    """Call fn and count the 'overflow encountered in exp' warnings raised during the fit."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    n_overflow = sum(1 for w in caught if "overflow encountered in exp" in str(w.message))
    return result, n_overflow


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    data_size_days = config.data_size_days
    sweep_seeds = config.data_size_sweep_seeds

    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    # one ensemble per (station, valid_time, lead); keep the keys aligned with the aggregates
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

    # restrict the test set to the instances of the prediction file
    key_cols = ["station_id", "valid_time", "step_hours"]
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    per_method_keys = {m: raw.loc[raw["method"] == m, key_cols].drop_duplicates() for m in raw["method"].unique()}
    canonical_key_set = per_method_keys["tsfm3"]
    methods_share_identical_keys = all(
        len(canonical_key_set.merge(per_method_keys[m], on=key_cols, how="inner")) == len(canonical_key_set) == len(per_method_keys[m])
        for m in per_method_keys
    )
    assert methods_share_identical_keys, "raw_ensemble/emos/tsfm3 instance keys differ in the prediction file"

    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    n_test_instances = len(test_keys_df)
    n_parquet_instances = len(canonical_key_set)
    n_matched = len(matched_keys)
    print(f"test instances={n_test_instances}, instances in prediction file={n_parquet_instances}, matched={n_matched}")
    assert n_matched == n_parquet_instances, "not all instances of the prediction file are in the test set"

    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means = ens_means[matched_row_idx]
    ens_vars = ens_vars[matched_row_idx]
    obs_values = obs_values[matched_row_idx]
    test_station_ids = test_station_ids[matched_row_idx]
    test_X = {"ens_mean": ens_means, "ens_var": ens_vars}
    n_unique_test_stations = len(np.unique(test_station_ids))

    unique_pairs_full = full_train[["year_idx", "time_idx"]].drop_duplicates()
    n_pairs_full = len(unique_pairs_full)

    rows = []
    n_overflow_by_ndays = {}
    e2_cache = {}  # per-instance scores of pooled EMOS at 30 and 90 days, for the significance test
    for n_days in data_size_days:
        n_overflow_this_ndays = 0

        # contiguous arm
        train_contig = sample_contiguous(full_train, n_days)
        pooled_metrics = None  # reused for the random arm at "full"
        if len(train_contig) == 0:
            k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)
            print(f"N={n_days} days, arm=contiguous (0 training rows): nothing to fit")
            for method in ["emos_pooled", "emos_local", "drn", "var_inflation_trainfit"]:
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "contiguous", "seed": None, "method": method,
                    "crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan"),
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None, "fit_seconds": None,
                })
        else:
            k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)

            t0 = time.perf_counter()
            pooled_preds, n_overflow = _catch_overflow_warnings(fit_predict_pooled_emos, train_contig, quantile_levels, test_X)
            pooled_fit_seconds = time.perf_counter() - t0
            n_overflow_this_ndays += n_overflow
            pooled_metrics = compute_metrics(obs_values, pooled_preds, quantile_levels)
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "emos_pooled",
                **pooled_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None, "fit_seconds": pooled_fit_seconds,
            })
            print(f"N={n_days} days, arm=contiguous, method=emos_pooled ({len(train_contig)} rows, {pooled_fit_seconds:.2f}s): {pooled_metrics}")

            # per-instance scores for the significance test below
            if n_days in (30, 90):
                lo_idx_e2, hi_idx_e2 = quantile_levels.index(0.1), quantile_levels.index(0.9)
                e2_cache[n_days] = {
                    "crps_per_instance": crps_from_quantiles(obs_values, pooled_preds, quantile_levels).flatten(),
                    "coverage_indicator": (
                        (obs_values[:, 0] >= pooled_preds[:, 0, lo_idx_e2])
                        & (obs_values[:, 0] <= pooled_preds[:, 0, hi_idx_e2])
                    ).astype(float),
                }

            t0 = time.perf_counter()
            (local_preds, covered_mask, coverage_fraction), n_overflow = _catch_overflow_warnings(
                fit_predict_local_emos, train_contig, test_station_ids, test_X, quantile_levels
            )
            local_fit_seconds = time.perf_counter() - t0
            n_overflow_this_ndays += n_overflow
            n_stations_covered = len(np.unique(test_station_ids[covered_mask])) if covered_mask.any() else 0
            if covered_mask.any():
                local_metrics = compute_metrics(
                    obs_values[covered_mask], local_preds[covered_mask], quantile_levels
                )
            else:
                local_metrics = {"crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan")}
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "emos_local",
                **local_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": n_stations_covered, "fit_seconds": local_fit_seconds,
            })
            print(
                f"N={n_days} days, arm=contiguous, method=emos_local "
                f"(coverage {n_stations_covered}/{n_unique_test_stations} test stations, {local_fit_seconds:.2f}s): {local_metrics}"
            )

            # DRN (Rasp and Lerch 2018): mean and SD over three seeds
            t0 = time.perf_counter()
            drn_train = train_contig[["station_id", "year_idx", "time_idx", "ens_mean", "ens_var", "t2m_obs"]]
            drn_runs = []
            for drn_seed in sweep_seeds[:3]:
                drn_model = DRN(quantile_levels=quantile_levels, seed=drn_seed).fit(drn_train)
                drn_preds = drn_model.predict_quantiles({
                    "ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"], "station_id": test_station_ids,
                })
                drn_runs.append(compute_metrics(obs_values, drn_preds, quantile_levels))
                print(f"N={n_days} DRN seed={drn_seed}: {drn_model.n_updates_} updates, "
                      f"{len(drn_model.loss_history_)} epochs, {drn_runs[-1]}")
            drn_fit_seconds = time.perf_counter() - t0
            drn_metrics = {m: float(np.mean([r[m] for r in drn_runs])) for m in drn_runs[0]}
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "drn",
                **drn_metrics,
                "crps_std": float(np.std([r["crps"] for r in drn_runs])),
                "coverage_80pct_std": float(np.std([r["coverage_80pct"] for r in drn_runs])),
                "interval_width_k_std": float(np.std([r["interval_width_k"] for r in drn_runs])),
                "n_stations_covered": None, "fit_seconds": drn_fit_seconds,
            })
            print(f"N={n_days} days ({len(train_contig)} training rows, {drn_fit_seconds:.2f}s): DRN CRPS={drn_metrics['crps']:.4f}")

            # variance inflation, multiplier fit on the same training cases
            t0 = time.perf_counter()
            vi_train = train_contig[["ens_mean", "ens_var", "t2m_obs"]]
            vi_model = VarianceInflationBaseline(quantile_levels=quantile_levels).fit(vi_train)
            vi_preds = vi_model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})
            vi_fit_seconds = time.perf_counter() - t0
            vi_metrics = compute_metrics(obs_values, vi_preds, quantile_levels)
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "var_inflation_trainfit",
                **vi_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None, "fit_seconds": vi_fit_seconds,
            })
            print(
                f"N={n_days} days, arm=contiguous, method=var_inflation_trainfit "
                f"(multiplier={vi_model.multiplier:.4f}, {vi_fit_seconds:.4f}s): {vi_metrics}"
            )

        # random arm
        k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)

        if n_days == "full":
            # all seeds use the full archive: reuse the contiguous fit (SD is 0 by construction)
            assert pooled_metrics is not None
            for seed in sweep_seeds:
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "random", "seed": seed, "method": "emos_pooled",
                    **pooled_metrics,
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None, "fit_seconds": None,
                })
            print(
                f"N=full days, arm=random (all {len(sweep_seeds)} seeds), method=emos_pooled: "
                f"reused full-archive fit (identical to contiguous by construction, no refit): {pooled_metrics}"
            )
            mean_row = {
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "random_mean", "seed": None, "method": "emos_pooled",
                "crps": pooled_metrics["crps"],
                "coverage_80pct": pooled_metrics["coverage_80pct"],
                "interval_width_k": pooled_metrics["interval_width_k"],
                "crps_std": 0.0, "coverage_80pct_std": 0.0, "interval_width_k_std": 0.0,
                "n_stations_covered": None, "fit_seconds": None,
            }
        else:
            seed_metrics = []
            seed_fit_seconds = []
            for seed in sweep_seeds:
                train_rand = sample_random(full_train, n_days, seed)
                if len(train_rand) == 0:
                    rand_metrics = {"crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan")}
                    rand_fit_seconds = float("nan")
                    print(f"N={n_days} days, arm=random, seed={seed} (0 training rows): EMOS undefined, no data to fit")
                else:
                    t0 = time.perf_counter()
                    rand_preds, n_overflow = _catch_overflow_warnings(fit_predict_pooled_emos, train_rand, quantile_levels, test_X)
                    rand_fit_seconds = time.perf_counter() - t0
                    n_overflow_this_ndays += n_overflow
                    rand_metrics = compute_metrics(obs_values, rand_preds, quantile_levels)
                    print(f"N={n_days} days, arm=random, seed={seed}, method=emos_pooled ({len(train_rand)} rows, {rand_fit_seconds:.2f}s): {rand_metrics}")
                seed_metrics.append(rand_metrics)
                seed_fit_seconds.append(rand_fit_seconds)
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "random", "seed": seed, "method": "emos_pooled",
                    **rand_metrics,
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None, "fit_seconds": rand_fit_seconds,
                })

            seed_df = pd.DataFrame(seed_metrics)
            mean_row = {
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "random_mean", "seed": None, "method": "emos_pooled",
                "crps": float(seed_df["crps"].mean()),
                "coverage_80pct": float(seed_df["coverage_80pct"].mean()),
                "interval_width_k": float(seed_df["interval_width_k"].mean()),
                "crps_std": float(seed_df["crps"].std()),
                "coverage_80pct_std": float(seed_df["coverage_80pct"].std()),
                "interval_width_k_std": float(seed_df["interval_width_k"].std()),
                "n_stations_covered": None,
                "fit_seconds": float(np.nanmean(seed_fit_seconds)) if not all(np.isnan(seed_fit_seconds)) else float("nan"),
            }
        rows.append(mean_row)
        print(f"N={n_days} days, arm=random_mean, method=emos_pooled: {mean_row}")

        n_overflow_by_ndays[str(n_days)] = n_overflow_this_ndays
        if n_overflow_this_ndays:
            print(f"N={n_days} days: {n_overflow_this_ndays} EMOS.fit call(s) raised an overflow warning")

    # reference lines that do not depend on the training size: raw ensemble and TimesFM-3
    quantile_cols = [f"q{q}" for q in quantile_levels]
    tsfm3_ref_metrics = None
    for method in ["raw_ensemble", "tsfm3"]:
        df_method = raw[raw["method"] == method].merge(matched_keys[key_cols], on=key_cols, how="inner")
        qp = df_method[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
        y = df_method["obs"].to_numpy().reshape(-1, 1)
        ref_metrics = compute_metrics(y, qp, quantile_levels)
        if method == "tsfm3":
            tsfm3_ref_metrics = ref_metrics
        for n_days in data_size_days:
            rows.append({
                "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
                "sampling_arm": "n_independent", "seed": None, "method": method,
                **ref_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None, "fit_seconds": None,
            })
        print(f"method={method} (N-independent, {len(df_method)} matched instances): {ref_metrics}")
    assert tsfm3_ref_metrics is not None

    # fixed variance inflation (no fitting)
    vi_fixed_model = VarianceInflationBaseline.from_fixed_multiplier(LAMBDA_CLIM, quantile_levels=quantile_levels)
    vi_fixed_preds = vi_fixed_model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})
    vi_fixed_metrics = compute_metrics(obs_values, vi_fixed_preds, quantile_levels)
    for n_days in data_size_days:
        rows.append({
            "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
            "sampling_arm": "n_independent", "seed": None, "method": "var_inflation_fixed",
            **vi_fixed_metrics,
            "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
            "n_stations_covered": None, "fit_seconds": None,
        })
    print(f"method=var_inflation_fixed (N-independent, multiplier={LAMBDA_CLIM}): {vi_fixed_metrics}")

    # variance inflation with the multiplier chosen to match the coverage of TimesFM-3 on the training archive
    vi_covmatch_train = full_train[["ens_mean", "ens_var", "t2m_obs"]]
    vi_covmatch_model = VarianceInflationBaseline.from_coverage_target(
        target_coverage=float(tsfm3_ref_metrics["coverage_80pct"]),
        train_df=vi_covmatch_train,
        quantile_levels=quantile_levels,
    )
    vi_covmatch_preds = vi_covmatch_model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})
    vi_covmatch_metrics = compute_metrics(obs_values, vi_covmatch_preds, quantile_levels)
    for n_days in data_size_days:
        rows.append({
            "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
            "sampling_arm": "n_independent", "seed": None, "method": "var_inflation_coverage_matched",
            **vi_covmatch_metrics,
            "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
            "n_stations_covered": None, "fit_seconds": None,
        })
    print(f"method=var_inflation_coverage_matched (lambda_c={vi_covmatch_model.multiplier:.4f}): {vi_covmatch_metrics}")

    # upper bound: multiplier fit directly on the test set (not a usable baseline)
    vi_leaky_test_df = pd.DataFrame({
        "ens_mean": test_X["ens_mean"].ravel(),
        "ens_var": test_X["ens_var"].ravel(),
        "t2m_obs": obs_values.ravel(),
    })
    vi_leaky_model = VarianceInflationBaseline.from_coverage_target(
        target_coverage=float(tsfm3_ref_metrics["coverage_80pct"]),
        train_df=vi_leaky_test_df,
        quantile_levels=quantile_levels,
    )
    vi_leaky_preds = vi_leaky_model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})
    vi_leaky_metrics = compute_metrics(obs_values, vi_leaky_preds, quantile_levels)
    for n_days in data_size_days:
        rows.append({
            "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
            "sampling_arm": "n_independent", "seed": None, "method": "var_inflation_test_leakage",
            **vi_leaky_metrics,
            "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
            "n_stations_covered": None, "fit_seconds": None,
        })
    print(f"method=var_inflation_test_leakage (lambda={vi_leaky_model.multiplier:.4f}): {vi_leaky_metrics}")

    sweep_df = pd.DataFrame(rows)

    # n_cases and n_calendar_days_equiv must be numeric or None (not the label 'full') to be written to parquet
    def _numeric_or_none(v):
        return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, float) and np.isnan(v))

    assert sweep_df["n_cases"].map(_numeric_or_none).all()
    assert sweep_df["n_calendar_days_equiv"].map(_numeric_or_none).all()

    write_result(
        sweep_df,
        name="training_size",
        model_version="training-size",
        config={
            "data_size_days": data_size_days,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "chronological_descending": CHRONOLOGICAL_DESCENDING,
            "days_per_case_measured": DAYS_PER_CASE,
            "n_matched_instances": int(n_matched),
            "n_overflow_warnings_by_n_days": n_overflow_by_ndays,
        },
    )

    # training size at which the CRPS and coverage of each method cross those of TimesFM-3
    # (interval width has no direction on its own and is not tested)
    tsfm3_row = sweep_df[(sweep_df["method"] == "tsfm3")].iloc[0]
    breakpoint_rows = []
    for metric in ["crps", "coverage_80pct"]:
        reference_value = tsfm3_row[metric]
        for emos_variant in ["emos_pooled", "emos_local", "drn", "var_inflation_trainfit"]:
            arm_df = sweep_df[
                (sweep_df["sampling_arm"] == "contiguous") & (sweep_df["method"] == emos_variant)
            ].copy()
            if emos_variant == "emos_local":
                arm_df = arm_df[arm_df["n_stations_covered"].fillna(0) > 0]
            n_cases_axis = arm_df["n_cases"].astype(float).tolist()
            method_values = arm_df[metric].tolist()

            bp, reason = breakpoint_and_direction(metric, n_cases_axis, method_values, reference_value)
            bp_calendar = bp_to_calendar_days(bp)
            breakpoint_rows.append({
                "metric": metric,
                "emos_variant": emos_variant,
                "breakpoint_n_cases": bp,
                "breakpoint_calendar_days": bp_calendar,
                "crossing_direction": reason,
            })
            if bp is None:
                print(f"breakpoint: metric={metric}, variant={emos_variant}: no crossing — {reason}")
            else:
                print(f"breakpoint: metric={metric}, variant={emos_variant}: {bp:.1f} cases (~{bp_calendar} calendar days) — {reason}")

    breakpoints_df = pd.DataFrame(breakpoint_rows)
    assert breakpoints_df["breakpoint_n_cases"].map(_numeric_or_none).all()
    assert breakpoints_df["breakpoint_calendar_days"].map(_numeric_or_none).all()

    write_result(
        breakpoints_df,
        name="training_size_breakpoints",
        model_version="training-size",
        config={
            "data_size_days": data_size_days,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "chronological_descending": CHRONOLOGICAL_DESCENDING,
            "days_per_case_measured": DAYS_PER_CASE,
        },
    )

    # station-blocked paired test of pooled EMOS (30 and 90 days) against TimesFM-3, in the row order of matched_keys
    tsfm3_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all()
    assert len(tsfm3_ordered) == n_matched
    tsfm3_qp_ordered = tsfm3_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    tsfm3_y_ordered = tsfm3_ordered["obs"].to_numpy().reshape(-1, 1)
    lo_idx_e2, hi_idx_e2 = quantile_levels.index(0.1), quantile_levels.index(0.9)
    tsfm3_crps_instance = crps_from_quantiles(tsfm3_y_ordered, tsfm3_qp_ordered, quantile_levels).flatten()
    tsfm3_coverage_indicator = (
        (tsfm3_y_ordered[:, 0] >= tsfm3_qp_ordered[:, 0, lo_idx_e2])
        & (tsfm3_y_ordered[:, 0] <= tsfm3_qp_ordered[:, 0, hi_idx_e2])
    ).astype(float)
    significance_rows = []
    for n_days_e2, cached in sorted(e2_cache.items()):
        k_e2, n_calendar_e2 = k_and_calendar_days(n_days_e2, n_pairs_full)
        for metric_name, emos_arr, tsfm3_arr in [
            ("crps", cached["crps_per_instance"], tsfm3_crps_instance),
            ("coverage_80pct", cached["coverage_indicator"], tsfm3_coverage_indicator),
        ]:
            test_result = station_blocked_paired_test(emos_arr, tsfm3_arr, test_station_ids)
            significance_rows.append({
                "n_days": str(n_days_e2), "n_cases": k_e2, "n_calendar_days_equiv": n_calendar_e2,
                "metric": metric_name, "method_a": "emos_pooled", "method_b": "tsfm3",
                **test_result,
            })
            print(
                f"significance: N={n_days_e2} days (k={k_e2}), metric={metric_name}, "
                f"emos_pooled vs tsfm3: mean diff(a-b)={test_result['block_mean_diff']:.4f}, "
                f"t p={test_result['t_pvalue']:.4f}, wilcoxon p={test_result['wilcoxon_pvalue']:.4f}"
            )

    significance_df = pd.DataFrame(significance_rows)
    write_result(
        significance_df,
        name="training_size_significance",
        model_version="training-size",
        config={"quantile_levels": quantile_levels, "compared_n_days": sorted(e2_cache.keys())},
    )

    # small training sizes (k = 1 ... 7), chosen directly by k; random arm with 5 seeds for pooled EMOS
    low_n_rows = []
    for k_low in LOW_N_K_GRID:
        synthetic_n_days = n_days_for_exact_k(k_low)
        n_calendar_low = k_to_calendar_days(k_low)

        train_contig_low = sample_contiguous(full_train, synthetic_n_days)
        actual_k = len(train_contig_low[["year_idx", "time_idx"]].drop_duplicates())
        assert actual_k == k_low

        t0 = time.perf_counter()
        pooled_preds_low, _ = _catch_overflow_warnings(fit_predict_pooled_emos, train_contig_low, quantile_levels, test_X)
        pooled_fit_seconds_low = time.perf_counter() - t0
        pooled_metrics_low = compute_metrics(obs_values, pooled_preds_low, quantile_levels)
        low_n_rows.append({
            "k": k_low, "n_calendar_days_equiv": n_calendar_low, "sampling_arm": "contiguous", "seed": None,
            "method": "emos_pooled", **pooled_metrics_low,
            "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
            "n_stations_covered": None, "fit_seconds": pooled_fit_seconds_low,
        })
        print(f"small grid: k={k_low} (~{n_calendar_low} days), arm=contiguous, method=emos_pooled: {pooled_metrics_low}")

        t0 = time.perf_counter()
        (local_preds_low, covered_mask_low, _), _ = _catch_overflow_warnings(
            fit_predict_local_emos, train_contig_low, test_station_ids, test_X, quantile_levels
        )
        local_fit_seconds_low = time.perf_counter() - t0
        n_stations_covered_low = len(np.unique(test_station_ids[covered_mask_low])) if covered_mask_low.any() else 0
        if covered_mask_low.any():
            local_metrics_low = compute_metrics(obs_values[covered_mask_low], local_preds_low[covered_mask_low], quantile_levels)
        else:
            local_metrics_low = {"crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan")}
        low_n_rows.append({
            "k": k_low, "n_calendar_days_equiv": n_calendar_low, "sampling_arm": "contiguous", "seed": None,
            "method": "emos_local", **local_metrics_low,
            "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
            "n_stations_covered": n_stations_covered_low, "fit_seconds": local_fit_seconds_low,
        })
        print(
            f"small grid: k={k_low} (~{n_calendar_low} days), arm=contiguous, method=emos_local "
            f"(coverage {n_stations_covered_low}/{n_unique_test_stations} test stations): {local_metrics_low}"
        )

        # random arm (pooled EMOS only)
        seed_metrics_low = []
        for seed in sweep_seeds:
            train_rand_low = sample_random(full_train, synthetic_n_days, seed)
            actual_k_rand = len(train_rand_low[["year_idx", "time_idx"]].drop_duplicates())
            assert actual_k_rand == k_low
            rand_preds_low, _ = _catch_overflow_warnings(fit_predict_pooled_emos, train_rand_low, quantile_levels, test_X)
            rand_metrics_low = compute_metrics(obs_values, rand_preds_low, quantile_levels)
            seed_metrics_low.append(rand_metrics_low)
            low_n_rows.append({
                "k": k_low, "n_calendar_days_equiv": n_calendar_low, "sampling_arm": "random", "seed": seed,
                "method": "emos_pooled", **rand_metrics_low,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None, "fit_seconds": None,
            })
        seed_df_low = pd.DataFrame(seed_metrics_low)
        low_n_random_mean = {
            "k": k_low, "n_calendar_days_equiv": n_calendar_low, "sampling_arm": "random_mean", "seed": None,
            "method": "emos_pooled",
            "crps": float(seed_df_low["crps"].mean()),
            "coverage_80pct": float(seed_df_low["coverage_80pct"].mean()),
            "interval_width_k": float(seed_df_low["interval_width_k"].mean()),
            "crps_std": float(seed_df_low["crps"].std()),
            "coverage_80pct_std": float(seed_df_low["coverage_80pct"].std()),
            "interval_width_k_std": float(seed_df_low["interval_width_k"].std()),
            "n_stations_covered": None, "fit_seconds": None,
        }
        low_n_rows.append(low_n_random_mean)
        print(
            f"small grid: k={k_low} (~{n_calendar_low} days), arm=random_mean (5 seeds), method=emos_pooled: "
            f"crps={low_n_random_mean['crps']:.4f} (std={low_n_random_mean['crps_std']:.4f}), "
            f"coverage_80pct={low_n_random_mean['coverage_80pct']:.4f} (std={low_n_random_mean['coverage_80pct_std']:.4f})"
        )

    low_n_df = pd.DataFrame(low_n_rows)
    write_result(
        low_n_df,
        name="training_size_small",
        model_version="training-size",
        config={
            "low_n_k_grid": LOW_N_K_GRID,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "days_per_case_measured": DAYS_PER_CASE,
        },
    )

    # breakpoints over the combined axis of the small grid and the main grid
    low_n_breakpoint_rows = []
    for metric in ["crps", "coverage_80pct"]:
        reference_value = tsfm3_row[metric]
        for emos_variant in ["emos_pooled", "emos_local"]:
            low_axis = low_n_df[(low_n_df["method"] == emos_variant) & (low_n_df["sampling_arm"] == "contiguous")]
            main_axis = sweep_df[(sweep_df["sampling_arm"] == "contiguous") & (sweep_df["method"] == emos_variant)]
            if emos_variant == "emos_local":
                low_axis = low_axis[low_axis["n_stations_covered"].fillna(0) > 0]
                main_axis = main_axis[main_axis["n_stations_covered"].fillna(0) > 0]
            n_cases_axis = low_axis["k"].astype(float).tolist() + main_axis["n_cases"].astype(float).tolist()
            method_values = low_axis[metric].tolist() + main_axis[metric].tolist()

            bp, reason = breakpoint_and_direction(metric, n_cases_axis, method_values, reference_value)
            bp_calendar = bp_to_calendar_days(bp)
            low_n_breakpoint_rows.append({
                "metric": metric, "emos_variant": emos_variant,
                "breakpoint_n_cases": bp, "breakpoint_calendar_days": bp_calendar,
                "crossing_direction": reason,
            })
            if bp is None:
                print(f"extended-axis breakpoint: metric={metric}, variant={emos_variant}: no crossing — {reason}")
            else:
                print(f"extended-axis breakpoint: metric={metric}, variant={emos_variant}: {bp:.2f} cases (~{bp_calendar} calendar days) — {reason}")

    low_n_breakpoints_df = pd.DataFrame(low_n_breakpoint_rows)
    write_result(
        low_n_breakpoints_df,
        name="training_size_small_breakpoints",
        model_version="training-size",
        config={"low_n_k_grid": LOW_N_K_GRID, "quantile_levels": quantile_levels},
    )


if __name__ == "__main__":
    main()
