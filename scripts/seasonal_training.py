"""Season-matched training: how many same-season training cases EMOS needs to match the zero-shot models.
Cases and test instances are assigned to meteorological seasons by the issue-date month. For each season, EMOS
(pooled, local, local lead-specific) is fit on the k most recent cases of the season (pooled also on k random
cases) and scored on the season's test instances, with raw ensemble, TimesFM-3 and Chronos-2; day-blocked
t-tests and HAC Diebold-Mariano tests give the first k from which EMOS is significantly better.
Output (results/): seasonal_sweep, seasonal_breakpoints, seasonal_significance, seasonal_counts.
Used for Fig. 4 and Tables 2, 7, A2.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_lead
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles
from zeropp.eval.robust_significance import robust_comparison
from zeropp.eval.significance import station_blocked_paired_test

import emos_variants
import training_size as sweep

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/predictions_main.parquet"

SEASONS = ["DJF", "MAM", "JJA", "SON"]
# training cases per season; the last point is every case of the season
K_GRID = [1, 2, 3, 5, 7, 9, 13, 26, 52, 105, 314]
ALPHA = 0.05
# local lead-specific EMOS is fit from this many cases on (cells need >= 5 rows)
LEADSPEC_MIN_K = 5


def season_of_month(month: int) -> str:
    """Meteorological season (DJF/MAM/JJA/SON) of a calendar month 1..12."""
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"month must be 1..12, got {month!r}")


def time_idx_to_issue_date(time_idx: pd.Series, template_dates: np.ndarray) -> pd.Series:
    """Map time_idx values to issue dates (already datetimes, or positions in the 209-entry time coordinate)."""
    if np.issubdtype(time_idx.dtype, np.datetime64):
        return pd.to_datetime(time_idx)
    return pd.Series(pd.to_datetime(template_dates)[time_idx.to_numpy().astype(int)], index=time_idx.index)


def sample_contiguous_k(cases: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k most recent cases (same ordering as training_size.sample_contiguous)."""
    ordered = cases.sort_values("case_date", ascending=False)
    return ordered.iloc[: min(k, len(ordered))]


def sample_random_k(cases: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cases), size=min(k, len(cases)), replace=False)
    return cases.iloc[idx]


def first_durable_significant_k(sig: pd.DataFrame) -> float | None:
    """Smallest k from which EMOS is significantly better (mean diff < 0, p < ALPHA) at that and every larger k."""
    sig = sig.sort_values("k_cases")
    good = ((sig["block_mean_diff"] < 0) & (sig["t_pvalue"] < ALPHA)).to_numpy()
    for i in range(len(good)):
        if good[i:].all():
            return float(sig["k_cases"].iloc[i])
    return None


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    seeds = config.data_size_sweep_seeds

    lead_specific_local = emos_variants.lead_specific_local
    compute_metrics = sweep.compute_metrics
    fit_predict_pooled_emos = sweep.fit_predict_pooled_emos
    fit_predict_local_emos = sweep.fit_predict_local_emos
    breakpoint_and_direction = sweep.breakpoint_and_direction
    days_per_case = sweep.DAYS_PER_CASE

    print("Loading archives...")
    full_train = build_train_ensemble_stats_with_lead(REFORECAST_PATH, REFORECAST_OBS_PATH)
    template_dates = xr.open_dataset(REFORECAST_PATH)["time"].values
    full_train["issue_date"] = pd.to_datetime(full_train["case_date"])
    full_train["season"] = full_train["issue_date"].dt.month.map(season_of_month)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    # test instances, joined to the matched TimesFM-3 instances
    key_cols = ["station_id", "valid_time", "step_hours"]
    agg = (
        test_df.groupby(key_cols)
        .agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"), obs=("t2m_obs", "first"))
        .reset_index()
    )
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    tsfm3 = raw[raw["method"] == "tsfm3"]
    rawens = raw[raw["method"] == "raw_ensemble"]
    chronos_path = Path("results/chronos2_predictions.parquet")
    chronos = pd.read_parquet(chronos_path) if chronos_path.exists() else None
    test = agg.merge(tsfm3[key_cols], on=key_cols, how="inner")
    assert len(test) == len(tsfm3)
    issue_time = pd.to_datetime(test["valid_time"]) - pd.to_timedelta(test["step_hours"], unit="h")
    test["issue_date"] = issue_time.dt.normalize()
    test["season"] = test["issue_date"].dt.month.map(season_of_month)
    test["day_block"] = pd.to_datetime(test["valid_time"]).dt.normalize()

    def ordered_preds(ref: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
        m = keys[key_cols].merge(ref, on=key_cols, how="left")
        assert m[quantile_cols].notna().all().all()
        return m[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))

    sweep_rows, bp_rows, sig_rows, count_rows = [], [], [], []
    for season in SEASONS:
        tr = full_train[full_train["season"] == season]
        te = test[test["season"] == season].reset_index(drop=True)
        cases = tr[["year_idx", "time_idx", "case_date"]].drop_duplicates()
        n_cases_full = len(cases)
        y = te["obs"].to_numpy().reshape(-1, 1)
        test_X = {"ens_mean": te[["ens_mean"]].to_numpy(), "ens_var": te[["ens_var"]].to_numpy()}
        station_ids = te["station_id"].to_numpy()

        tsfm3_qp = ordered_preds(tsfm3, te)
        ref_tsfm3 = compute_metrics(y, tsfm3_qp, quantile_levels)
        ref_raw = compute_metrics(y, ordered_preds(rawens, te), quantile_levels)
        tsfm3_crps = crps_from_quantiles(y, tsfm3_qp, quantile_levels).flatten()
        chronos_crps = (crps_from_quantiles(y, ordered_preds(chronos, te), quantile_levels).flatten()
                        if chronos is not None else None)

        count_rows.append({
            "season": season,
            "train_issue_date_labels": int(tr["time_idx"].nunique()),
            "train_cases": n_cases_full,
            "train_rows": len(tr),
            "train_first_label": tr["issue_date"].min(), "train_last_label": tr["issue_date"].max(),
            "test_issue_dates": int(te["issue_date"].nunique()),
            "test_instances": len(te),
            "test_first_issue": te["issue_date"].min(), "test_last_issue": te["issue_date"].max(),
            "obs_mean_k": float(te["obs"].mean()), "obs_std_k": float(te["obs"].std()),
        })
        refs = [("tsfm3", ref_tsfm3), ("raw_ensemble", ref_raw)]
        if chronos is not None:
            refs.append(("chronos2", compute_metrics(y, ordered_preds(chronos, te), quantile_levels)))
        for name, ref in refs:
            sweep_rows.append({"season": season, "k_cases": None, "sampling_arm": "n_independent",
                               "seed": None, "method": name, "n_stations_covered": None, **ref})
        print(f"[{season}] train cases={n_cases_full}, test instances={len(te)}, tsfm3={ref_tsfm3}")

        k_values = [k for k in K_GRID if k < n_cases_full] + [n_cases_full]
        for k in k_values:
            sub_cases = sample_contiguous_k(cases, k)
            train_k = tr.merge(sub_cases[["year_idx", "time_idx"]], on=["year_idx", "time_idx"], how="inner")

            pooled = fit_predict_pooled_emos(train_k, quantile_levels, test_X)
            sweep_rows.append({"season": season, "k_cases": k, "sampling_arm": "contiguous", "seed": None,
                               "method": "emos_pooled", "n_stations_covered": None,
                               **compute_metrics(y, pooled, quantile_levels)})
            pooled_crps = crps_from_quantiles(y, pooled, quantile_levels).flatten()
            res = station_blocked_paired_test(pooled_crps, tsfm3_crps, te["day_block"].to_numpy())
            rob = robust_comparison(pooled_crps, tsfm3_crps, te["valid_time"].to_numpy())
            row = {"season": season, "k_cases": k, "method": "emos_pooled", "block_definition": "day",
                   "n_instances": len(te), **res, **{f"hac_{a}": b for a, b in rob.items()}}
            if chronos is not None:
                rc = robust_comparison(pooled_crps, chronos_crps, te["valid_time"].to_numpy())
                row.update({f"hac_chronos_{a}": b for a, b in rc.items()})
            sig_rows.append(row)

            local, covered, frac = fit_predict_local_emos(train_k, station_ids, test_X, quantile_levels)
            if covered.any():
                sweep_rows.append({"season": season, "k_cases": k, "sampling_arm": "contiguous", "seed": None,
                                   "method": "emos_local",
                                   "n_stations_covered": int(len(np.unique(station_ids[covered]))),
                                   **compute_metrics(y[covered], local[covered], quantile_levels)})
                local_crps = crps_from_quantiles(y[covered], local[covered], quantile_levels).flatten()
                res = station_blocked_paired_test(local_crps, tsfm3_crps[covered], te["day_block"].to_numpy()[covered])
                rob = robust_comparison(local_crps, tsfm3_crps[covered], te["valid_time"].to_numpy()[covered])
                row = {"season": season, "k_cases": k, "method": "emos_local", "block_definition": "day",
                       "n_instances": int(covered.sum()), "covered_fraction": frac, **res,
                       **{f"hac_{a}": b for a, b in rob.items()}}
                if chronos is not None:
                    rc = robust_comparison(local_crps, chronos_crps[covered], te["valid_time"].to_numpy()[covered])
                    row.update({f"hac_chronos_{a}": b for a, b in rc.items()})
                sig_rows.append(row)

            if k >= LEADSPEC_MIN_K:
                ls = lead_specific_local(train_k, te, quantile_levels)
                covered = ~np.isnan(ls[:, 0])
                if covered.any():
                    ls = ls[:, None, :]
                    sweep_rows.append({"season": season, "k_cases": k, "sampling_arm": "contiguous", "seed": None,
                                       "method": "emos_local_leadspec",
                                       "n_stations_covered": int(len(np.unique(station_ids[covered]))),
                                       **compute_metrics(y[covered], ls[covered], quantile_levels)})
                    ls_crps = crps_from_quantiles(y[covered], ls[covered], quantile_levels).flatten()
                    res = station_blocked_paired_test(ls_crps, tsfm3_crps[covered], te["day_block"].to_numpy()[covered])
                    rob = robust_comparison(ls_crps, tsfm3_crps[covered], te["valid_time"].to_numpy()[covered])
                    row = {"season": season, "k_cases": k, "method": "emos_local_leadspec", "block_definition": "day",
                           "n_instances": int(covered.sum()), "covered_fraction": len(np.unique(station_ids[covered])) / len(np.unique(station_ids)), **res,
                           **{f"hac_{a}": b for a, b in rob.items()}}
                    if chronos is not None:
                        rc = robust_comparison(ls_crps, chronos_crps[covered], te["valid_time"].to_numpy()[covered])
                        row.update({f"hac_chronos_{a}": b for a, b in rc.items()})
                    sig_rows.append(row)

            if k < n_cases_full:
                per_seed = []
                for seed in seeds:
                    train_r = tr.merge(sample_random_k(cases, k, seed)[["year_idx", "time_idx"]], on=["year_idx", "time_idx"], how="inner")
                    per_seed.append(compute_metrics(y, fit_predict_pooled_emos(train_r, quantile_levels, test_X), quantile_levels))
                mean = {m: float(np.mean([d[m] for d in per_seed])) for m in per_seed[0]}
                std = {f"{m}_std": float(np.std([d[m] for d in per_seed])) for m in per_seed[0]}
                sweep_rows.append({"season": season, "k_cases": k, "sampling_arm": "random", "seed": None,
                                   "method": "emos_pooled", "n_stations_covered": None, **mean, **std})
            print(f"[{season}] k={k} done")

        sweep_df = pd.DataFrame(sweep_rows)
        sig_df = pd.DataFrame(sig_rows)
        for method in ["emos_pooled", "emos_local", "emos_local_leadspec"]:
            for arm in ["contiguous", "random"]:
                cur = sweep_df[(sweep_df["season"] == season) & (sweep_df["method"] == method)
                               & (sweep_df["sampling_arm"] == arm)].sort_values("k_cases")
                if cur.empty:
                    continue
                for metric, ref_val in [("crps", ref_tsfm3["crps"]), ("coverage_80pct", ref_tsfm3["coverage_80pct"])]:
                    bp, why = breakpoint_and_direction(metric, cur["k_cases"].tolist(), cur[metric].tolist(), ref_val)
                    sig_k = None
                    if metric == "crps" and arm == "contiguous":
                        sig_k = first_durable_significant_k(
                            sig_df[(sig_df["season"] == season) & (sig_df["method"] == method)]
                        )
                    bp_rows.append({
                        "season": season, "method": method, "sampling_arm": arm, "metric": metric,
                        "reference_value_tsfm3": ref_val,
                        "breakpoint_k": bp, "breakpoint_days": None if bp is None else round(bp * days_per_case),
                        "crossing_direction": why,
                        "first_durable_sig_k": sig_k,
                        "first_durable_sig_days": None if sig_k is None else round(sig_k * days_per_case),
                    })

    common_config = {"quantile_levels": quantile_levels, "k_grid": K_GRID, "seeds": seeds,
                     "alpha": ALPHA, "days_per_case": days_per_case, "season_basis": "issue-date month"}
    for name, df in [
        ("seasonal_sweep", pd.DataFrame(sweep_rows)),
        ("seasonal_breakpoints", pd.DataFrame(bp_rows)),
        ("seasonal_significance", pd.DataFrame(sig_rows)),
        ("seasonal_counts", pd.DataFrame(count_rows)),
    ]:
        write_result(df, name=name, model_version="seasonal-training", config=common_config)
    print(pd.DataFrame(bp_rows).to_string())
    print(pd.DataFrame(count_rows).to_string())


if __name__ == "__main__":
    main()
