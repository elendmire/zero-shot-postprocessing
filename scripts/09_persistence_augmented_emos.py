"""Investigation 1 (docs/references/priority123_investigation_report.md,
"is the 0-24h nowcasting advantage a real architectural advantage, or just an
unfair information asymmetry?"): TimesFM-3 receives the last ~40 observations
plus NWP ensemble-mean/spread as covariates at every lead time; plain EMOS
receives ONLY ens_mean/ens_var, no observation history at all. At very short
leads (0-24h) the most recent observation is close to a perfect predictor
(strong persistence), so TimesFM-3's short-lead CRPS advantage over EMOS
could be substantially explained by TimesFM-3 having access to information
EMOS structurally lacks, rather than by anything architectural.

This script builds a "persistence-augmented EMOS"
(zeropp.models.emos.PersistenceAugmentedEMOS: mu = a + b*ens_mean + e*last_obs,
sigma unchanged from plain EMOS) restricted to the 0-24h lead-time bucket ONLY
(step_hours in {0, 6, 12, 18} -- this project's LEAD_TIME_BUCKETS[0], see
scripts/06_lead_time_grouped_analysis.py), fit on the training reforecast
archive (never on test-set information), and compares it against:
  (a1) plain EMOS-pooled (lead-pooled training, full N) -- the number
       docs/references/priority123_investigation_report.md's brief explicitly
       points to, already persisted in
       results/phase3_lead_time_bucketed_sweep.parquet (n_days="full", bucket
       0-24h).
  (a2) plain EMOS bucket-specific (trained ONLY on 0-24h-bucket-restricted
       reforecast rows, full N) -- already persisted in
       results/phase3_lead_time_grouped_emos.parquet. This is the more
       directly comparable baseline for isolating last_obs's effect (same
       training-data lead-time scope as the persistence-augmented model
       below; only difference is the extra predictor), so it is refit HERE
       on the identical post-join row set the persistence model uses, for an
       apples-to-apples "does adding last_obs help" delta.
  (b) TimesFM-3 (zero-shot), 0-24h bucket -- same source file as (a1).

=== step_hours=0: real forecast lead, or degenerate analysis time? ===

Checked directly (this task, altay, 2026-09):
  1. Raw NetCDF CF metadata (germany_ensemble_forecasts_t2m.nc's `step`
     coordinate): standard_name="forecast_period", long_name="time since
     forecast_reference_time", units="hours", value 0 for the first step.
     This is the archive's own definition of step_hours=0 as a genuine T+0
     forecast (valid at the same instant as issuance), not a flag for
     "already-assimilated / equals the observation."
  2. Empirically (results/phase3_lead_time_breakdown.parquet): if step_hours=0
     forecasts were degenerate (forecast == obs), CRPS there would be near
     zero for every method. Instead: raw_ensemble CRPS at step_hours=0
     (1.2624 K) is WORSE than at step_hours=6 (1.0932 K); EMOS CRPS at
     step_hours=0 (1.1369 K) is likewise worse than at step_hours=6
     (0.9671 K); TimesFM-3's CRPS at step_hours=0 (0.9195 K) is essentially
     unchanged from step_hours=6 (0.9188 K), showing no special "free win"
     either. raw_ensemble's coverage@80% at step_hours=0 (0.2374) is exactly
     as badly under-dispersed as every other short lead, not near 1.0 as a
     degenerate/trivial case would show.
  CONCLUSION: step_hours=0 is a real, non-degenerate forecast lead time and
  IS INCLUDED in this script's 0-24h bucket headline comparison.

Including it correctly, however, requires care in the last_obs CONSTRUCTION
below (not in whether it belongs in the bucket): at step_hours=0, issue_time
== valid_time, so "the observation at issue_time" IS the observation being
predicted -- using it as last_obs would be pure leakage, not persistence. See
build_train_last_obs()/build_test_last_obs() for how this is avoided (a
genuinely earlier observation is used instead, at valid_time - 6h, both for
train and test), with a strict "never after issue time, never the same
physical instant" discipline throughout.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_lead
from zeropp.eval.results import write_result
from zeropp.models.emos import EMOS, PersistenceAugmentedEMOS

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"
LEAD_TIME_BUCKETED_SWEEP_PATH = "results/phase3_lead_time_bucketed_sweep.parquet"
LEAD_TIME_GROUPED_EMOS_PATH = "results/phase3_lead_time_grouped_emos.parquet"

BUCKET_LABEL = "0-24h"
BUCKET_LOW, BUCKET_HIGH = 0.0, 24.0  # half-open [0, 24) -- matches 08's LEAD_TIME_BUCKETS[0]

# --- Reuse 08's bucket-assignment / instance-set-join helpers verbatim, same
# importlib-from-file mechanism every later script in this project uses for a
# filename starting with a digit (not a valid module name). ---
_SCRIPT_08_PATH = Path(__file__).resolve().parent / "06_lead_time_grouped_analysis.py"
_SPEC_08 = importlib.util.spec_from_file_location("lead_time_grouped_analysis_module_11", _SCRIPT_08_PATH)
_lead08 = importlib.util.module_from_spec(_SPEC_08)
_SPEC_08.loader.exec_module(_lead08)
assign_lead_time_bucket = _lead08.assign_lead_time_bucket

_SCRIPT_07_PATH = Path(__file__).resolve().parent / "03_data_size_sweep.py"
_SPEC_07 = importlib.util.spec_from_file_location("data_size_sweep_module_11", _SCRIPT_07_PATH)
_sweep07 = importlib.util.module_from_spec(_SPEC_07)
_SPEC_07.loader.exec_module(_sweep07)
compute_metrics = _sweep07.compute_metrics


def build_train_last_obs(full_train_lead: pd.DataFrame) -> pd.DataFrame:
    """Builds a leakage-free `last_obs` column for every 0-24h-bucket row of
    the TRAINING reforecast archive (full_train_lead, from
    build_train_ensemble_stats_with_lead -- NOT restricted to the bucket yet,
    since step_hours=0 rows need OTHER rows of the same forecast ISSUE as
    their last_obs source).

    IMPORTANT: `time_idx` here (build_train_ensemble_stats_with_ids/_with_lead's
    naming) is the raw archive's real `time` coordinate -- confirmed (this
    project's own prior measurement, scripts/03_data_size_sweep.py's module
    docstring) to decode to genuine datetime64 values, NOT an ordinal
    position index. `year_idx` (the raw `year` coordinate) is a bare
    1..20 analog-year label with no confirmed real-calendar meaning, so all
    matching below is done WITHIN one year_idx at a time, using real datetime
    arithmetic on `time_idx` + `step_hours` only -- never assuming adjacent
    `time_idx` values are 1 apart in real time (they are real timestamps,
    spaced 72h or 96h apart per the measured template), and never crossing
    year_idx (whose real-calendar ordering is unconfirmed).

    Two cases, both using ONLY information that genuinely predates or
    coincides with the row's own forecast issue_time (never the row's own
    valid_time when that would equal issue_time):

    step_hours in {6, 12, 18}: issue_time = this row's own time_idx (the real
    issue timestamp), strictly BEFORE this row's own valid_time
    (time_idx + step_hours). The SAME forecast's own step_hours=0 row has
    valid_time == time_idx == issue_time exactly -- i.e. it IS the
    observation available at issue time. last_obs = that sibling row's
    t2m_obs (a same-(station_id, year_idx, time_idx) self-join).

    step_hours == 0: issue_time == this row's OWN valid_time (== time_idx),
    so the same forecast's own step=0 obs is the target itself and cannot be
    used. Instead this looks up ANY OTHER row in the SAME year_idx (any
    time_idx, any step_hours) whose real valid instant
    (time_idx + step_hours) equals (this row's time_idx - 6h) -- a genuinely
    earlier physical observation, via a direct datetime match rather than an
    assumed ordinal adjacency. Rows with no such match anywhere in the same
    year_idx (e.g. the very first template date, with nothing earlier to
    match against) are dropped; count reported.
    """
    working = full_train_lead.copy()
    working["valid_datetime"] = working["time_idx"] + pd.to_timedelta(working["step_hours"], unit="h")

    step0 = working[working["step_hours"] == 0.0][
        ["station_id", "year_idx", "time_idx", "t2m_obs"]
    ].rename(columns={"t2m_obs": "last_obs"})

    bucket = working[
        (working["step_hours"] >= BUCKET_LOW) & (working["step_hours"] < BUCKET_HIGH)
    ].copy()

    nonzero = bucket[bucket["step_hours"] != 0.0].merge(
        step0, on=["station_id", "year_idx", "time_idx"], how="left"
    ).drop(columns=["valid_datetime"])

    # --- step_hours == 0 rows: same-year_idx, direct-datetime-match lookup ---
    zero_rows = bucket[bucket["step_hours"] == 0.0].copy()
    n_zero_total = len(zero_rows)
    zero_rows["target_datetime"] = zero_rows["time_idx"] - pd.to_timedelta(6, unit="h")

    # A small number of (station_id, year_idx, valid_datetime) triples turn out
    # to carry CONFLICTING t2m_obs values across different (time_idx,
    # step_hours) combinations that this construction's "time_idx + step_hours
    # == real wall-clock instant" arithmetic maps to the same nominal
    # timestamp (checked directly: max conflict range up to 12.6 K, clustered
    # tightly around specific month-day/hour combinations -- consistent with
    # this project's known, previously-documented DST/calendar-template
    # landmine, CLAUDE.md: "DST gecisleri ve UTC/yerel saat karisikligi sahte
    # delik yaratir"). These keys are AMBIGUOUS, not resolvable from this data
    # alone -- rather than guess which of the conflicting values is "real,"
    # every ambiguous key is dropped from the lookup entirely (so any row that
    # would have matched one gets no last_obs and is excluded downstream,
    # exactly like a genuine no-match); count reported, never silently picked.
    obs_by_key = working[["station_id", "year_idx", "valid_datetime", "t2m_obs"]].drop_duplicates()
    key_cols_lookup = ["station_id", "year_idx", "valid_datetime"]
    conflict_counts = obs_by_key.groupby(key_cols_lookup)["t2m_obs"].transform("nunique")
    ambiguous_keys = obs_by_key.loc[conflict_counts > 1, key_cols_lookup].drop_duplicates()
    n_conflicting_keys = len(ambiguous_keys)
    print(
        f"train last_obs lookup: {n_conflicting_keys} (station_id, year_idx, valid_datetime) "
        "keys have conflicting t2m_obs across different (time_idx, step_hours) combinations "
        "(DST/calendar-template artifact) -- excluded from the lookup entirely, not guessed at."
    )
    unambiguous = obs_by_key.merge(ambiguous_keys, on=key_cols_lookup, how="left", indicator=True)
    unambiguous = unambiguous[unambiguous["_merge"] == "left_only"].drop(columns=["_merge"])
    source_lookup = unambiguous.drop_duplicates(subset=key_cols_lookup).rename(columns={"t2m_obs": "last_obs"})

    zero_joined = zero_rows.merge(
        source_lookup, left_on=["station_id", "year_idx", "target_datetime"],
        right_on=["station_id", "year_idx", "valid_datetime"], how="left",
    )
    zero_joined = zero_joined.drop(
        columns=[c for c in ("target_datetime", "valid_datetime", "valid_datetime_x", "valid_datetime_y")
                 if c in zero_joined.columns]
    )

    n_step0_dropped_no_match = int(zero_joined["last_obs"].isna().sum())
    print(
        f"train last_obs (step_hours=0): {n_zero_total} rows in bucket; "
        f"{n_step0_dropped_no_match} dropped (no matching earlier observation found "
        "anywhere in the same year_idx)."
    )

    combined = pd.concat([nonzero, zero_joined], ignore_index=True)
    combined = combined.dropna(subset=["last_obs"])
    print(
        f"train last_obs, all step_hours in {BUCKET_LABEL}: {len(bucket)} bucket rows total, "
        f"{len(combined)} rows with a valid last_obs after both joins "
        f"({len(bucket) - len(combined)} dropped overall)."
    )
    return combined


def build_test_last_obs(test_df: pd.DataFrame, bucket_keys: pd.DataFrame) -> pd.DataFrame:
    """Leakage-free `last_obs` for the TEST forecast archive's 0-24h-bucket
    matched instances. Unlike the training reforecast archive (issue dates
    72-96h apart), the test forecast archive issues DAILY (see
    docs/euppbench_forecasts_schema.md), so pooling every (station_id,
    valid_time) -> t2m_obs pair across ALL issue times and steps in the test
    archive gives dense ~6-hourly coverage of the whole test period per
    station -- a single pooled lookup suffices for every step_hours in the
    bucket, including step_hours=0 (which looks up valid_time - 6h, a
    genuinely earlier physical instant, via some OTHER issue date's step=18h
    forecast reaching that same timestamp)."""
    # Same discipline as build_train_last_obs: (station_id, valid_time) pairs
    # with CONFLICTING t2m_obs values across different issue_time/step
    # combinations (if any) are ambiguous, not resolvable from this data
    # alone -- excluded from the lookup entirely rather than guessed at;
    # count reported.
    obs_by_key = test_df[["station_id", "valid_time", "t2m_obs"]].drop_duplicates()
    conflict_counts = obs_by_key.groupby(["station_id", "valid_time"])["t2m_obs"].transform("nunique")
    ambiguous_keys = obs_by_key.loc[conflict_counts > 1, ["station_id", "valid_time"]].drop_duplicates()
    print(
        f"test last_obs lookup: {len(ambiguous_keys)} (station_id, valid_time) keys have "
        "conflicting t2m_obs across different issue_time/step combinations -- excluded from "
        "the lookup entirely, not guessed at."
    )
    unambiguous = obs_by_key.merge(ambiguous_keys, on=["station_id", "valid_time"], how="left", indicator=True)
    unambiguous = unambiguous[unambiguous["_merge"] == "left_only"].drop(columns=["_merge"])
    obs_lookup = unambiguous.drop_duplicates(subset=["station_id", "valid_time"]).rename(
        columns={"t2m_obs": "last_obs"}
    )

    keys = bucket_keys.copy()
    keys["target_time"] = keys["valid_time"] - pd.to_timedelta(
        np.maximum(keys["step_hours"].to_numpy(), 6.0), unit="h"
    )
    joined = keys.merge(
        obs_lookup, left_on=["station_id", "target_time"], right_on=["station_id", "valid_time"],
        how="left", suffixes=("", "_lookup"),
    )
    n_no_match = int(joined["last_obs"].isna().sum())
    print(
        f"test last_obs, {BUCKET_LABEL} bucket: {len(keys)} matched test instances; "
        f"{n_no_match} ({n_no_match / len(keys):.2%}) have no last_obs match and are dropped."
    )
    joined = joined.dropna(subset=["last_obs"])
    return joined[["station_id", "valid_time", "step_hours", "row_idx", "last_obs"]]


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels

    # ================= Training side =================
    print("Loading reforecast (train) archive with lead time retained...")
    full_train_lead = build_train_ensemble_stats_with_lead(REFORECAST_PATH, REFORECAST_OBS_PATH)
    train_bucket = build_train_last_obs(full_train_lead)

    persistence_model = PersistenceAugmentedEMOS(quantile_levels=quantile_levels).fit(
        train_bucket[["station_id", "ens_mean", "ens_var", "last_obs", "t2m_obs"]]
    )
    a, b, c, d, e = persistence_model._params
    print(
        f"PersistenceAugmentedEMOS fit on {BUCKET_LABEL} bucket ({len(train_bucket)} training rows): "
        f"a={a:.4f} b={b:.4f} c={c:.4f} d={d:.4f} e={e:.4f} "
        f"({'last_obs carries real weight' if abs(e) > 0.05 else 'last_obs weight near zero'})"
    )

    plain_bucket_model = EMOS(quantile_levels=quantile_levels).fit(
        train_bucket[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
    )

    # ================= Test side =================
    print("Loading forecast (test) archive...")
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)
    raw = pd.read_parquet(RAW_RESULTS_PATH)

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
    ens_means = np.array(ens_means)
    ens_vars = np.array(ens_vars)
    obs_values = np.array(obs_values)
    test_station_ids = np.array(key_station_ids)
    test_keys_df = pd.DataFrame({
        "station_id": key_station_ids, "valid_time": key_valid_times, "step_hours": key_step_hours,
        "row_idx": np.arange(len(key_station_ids)),
    })

    key_cols = ["station_id", "valid_time", "step_hours"]
    canonical_key_set = raw.loc[raw["method"] == "tsfm3", key_cols].drop_duplicates()
    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    assert len(matched_keys) == len(canonical_key_set), (
        f"instance-set join incomplete: matched {len(matched_keys)} of {len(canonical_key_set)} "
        "tsfm3 reference instances."
    )
    lead_buckets_matched = assign_lead_time_bucket(matched_keys["step_hours"].to_numpy())
    bucket_mask = lead_buckets_matched == BUCKET_LABEL
    bucket_keys = matched_keys[bucket_mask].reset_index(drop=True)
    print(f"Test instances matched to tsfm3's key set, {BUCKET_LABEL} bucket: {len(bucket_keys)}")

    last_obs_df = build_test_last_obs(test_df, bucket_keys)
    eval_keys = bucket_keys.merge(
        last_obs_df[["row_idx", "last_obs"]], on="row_idx", how="inner"
    )
    row_idx = eval_keys["row_idx"].to_numpy()
    ens_mean_eval = ens_means[row_idx].reshape(-1, 1)
    ens_var_eval = ens_vars[row_idx].reshape(-1, 1)
    obs_eval = obs_values[row_idx].reshape(-1, 1)
    last_obs_eval = eval_keys["last_obs"].to_numpy().reshape(-1, 1)
    n_eval = len(eval_keys)
    print(f"Final evaluation set ({BUCKET_LABEL}, last_obs available): {n_eval} instances")

    persistence_preds = persistence_model.predict_quantiles({
        "ens_mean": ens_mean_eval, "ens_var": ens_var_eval, "last_obs": last_obs_eval,
    })
    persistence_metrics = compute_metrics(obs_eval, persistence_preds, quantile_levels)

    plain_bucket_preds = plain_bucket_model.predict_quantiles({
        "ens_mean": ens_mean_eval, "ens_var": ens_var_eval,
    })
    plain_bucket_metrics = compute_metrics(obs_eval, plain_bucket_preds, quantile_levels)

    # tsfm3 on the SAME (last_obs-matched) evaluation rows, for a like-for-like comparison
    quantile_cols = [f"q{q}" for q in quantile_levels]
    tsfm3_ordered = eval_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all()
    tsfm3_qp = tsfm3_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    tsfm3_y = tsfm3_ordered["obs"].to_numpy().reshape(-1, 1)
    tsfm3_metrics_matched = compute_metrics(tsfm3_y, tsfm3_qp, quantile_levels)

    # --- Already-known reference numbers (full bucket population, NOT restricted
    # to the last_obs-matched subset) -- printed for context, never re-derived. ---
    bucketed_sweep = pd.read_parquet(LEAD_TIME_BUCKETED_SWEEP_PATH)
    a1_row = bucketed_sweep[
        (bucketed_sweep["lead_time_bucket"] == BUCKET_LABEL) & (bucketed_sweep["method"] == "emos_pooled")
        & (bucketed_sweep["n_days"] == "full") & (bucketed_sweep["sampling_arm"] == "contiguous")
    ].iloc[0]
    b_row = bucketed_sweep[
        (bucketed_sweep["lead_time_bucket"] == BUCKET_LABEL) & (bucketed_sweep["method"] == "tsfm3")
    ].iloc[0]
    grouped_emos = pd.read_parquet(LEAD_TIME_GROUPED_EMOS_PATH)
    a2_row = grouped_emos[
        (grouped_emos["lead_time_bucket"] == BUCKET_LABEL) & (grouped_emos["training"] == "bucket_specific")
    ].iloc[0]

    print(f"\n=== {BUCKET_LABEL} bucket: persistence-augmented EMOS vs. references ===")
    print(f"(a1) emos_pooled, lead-pooled training, full N, FULL bucket pop (n={int(a1_row['n_instances']) if 'n_instances' in a1_row else 'n/a'}): "
          f"CRPS={a1_row['crps']:.4f}, coverage={a1_row['coverage_80pct']:.4f}, width={a1_row['interval_width_k']:.4f}")
    print(f"(a2) emos bucket_specific training, full N, FULL bucket pop (n={int(a2_row['n_instances'])}): "
          f"CRPS={a2_row['crps']:.4f}, coverage={a2_row['coverage_80pct']:.4f}, width={a2_row['interval_width_k']:.4f}")
    print(f"(b)  tsfm3, FULL bucket pop: CRPS={b_row['crps']:.4f}, coverage={b_row['coverage_80pct']:.4f}, width={b_row['interval_width_k']:.4f}")
    print(f"--- on the last_obs-matched evaluation subset (n={n_eval}) ---")
    print(f"(a2') plain EMOS bucket-specific, refit on the identical post-join rows: "
          f"CRPS={plain_bucket_metrics['crps']:.4f}, coverage={plain_bucket_metrics['coverage_80pct']:.4f}, width={plain_bucket_metrics['interval_width_k']:.4f}")
    print(f"(persistence) PersistenceAugmentedEMOS: "
          f"CRPS={persistence_metrics['crps']:.4f}, coverage={persistence_metrics['coverage_80pct']:.4f}, width={persistence_metrics['interval_width_k']:.4f}")
    print(f"(b') tsfm3, on the identical matched rows: "
          f"CRPS={tsfm3_metrics_matched['crps']:.4f}, coverage={tsfm3_metrics_matched['coverage_80pct']:.4f}, width={tsfm3_metrics_matched['interval_width_k']:.4f}")

    gap_before = plain_bucket_metrics["crps"] - tsfm3_metrics_matched["crps"]
    gap_after = persistence_metrics["crps"] - tsfm3_metrics_matched["crps"]
    frac_closed = 1.0 - (gap_after / gap_before) if gap_before != 0 else float("nan")
    print(
        f"\nCRPS gap to tsfm3 (matched subset): plain EMOS bucket-specific={gap_before:+.4f}, "
        f"persistence-augmented EMOS={gap_after:+.4f} -- "
        f"{frac_closed:.1%} of the gap closed by adding last_obs "
        f"({'gap fully closed/reversed' if gap_after <= 0 else 'a real gap to tsfm3 remains'})."
    )

    rows = [
        {"method": "emos_pooled_full_bucket_pop", "scope": "full_bucket", "n_instances": int(len(bucket_keys)),
         **{k: float(a1_row[k]) for k in ("crps", "coverage_80pct", "interval_width_k")}},
        {"method": "emos_bucket_specific_full_bucket_pop", "scope": "full_bucket", "n_instances": int(a2_row["n_instances"]),
         **{k: float(a2_row[k]) for k in ("crps", "coverage_80pct", "interval_width_k")}},
        {"method": "tsfm3_full_bucket_pop", "scope": "full_bucket", "n_instances": int(len(bucket_keys)),
         **{k: float(b_row[k]) for k in ("crps", "coverage_80pct", "interval_width_k")}},
        {"method": "emos_bucket_specific_matched_subset", "scope": "last_obs_matched", "n_instances": n_eval, **plain_bucket_metrics},
        {"method": "persistence_augmented_emos", "scope": "last_obs_matched", "n_instances": n_eval, **persistence_metrics},
        {"method": "tsfm3_matched_subset", "scope": "last_obs_matched", "n_instances": n_eval, **tsfm3_metrics_matched},
    ]
    result_df = pd.DataFrame(rows)
    write_result(
        result_df,
        name="phase3_persistence_augmented_emos",
        model_version="phase3-persistence-emos-v1",
        config={
            "lead_time_bucket": BUCKET_LABEL, "quantile_levels": quantile_levels,
            "persistence_params": {"a": float(a), "b": float(b), "c": float(c), "d": float(d), "e": float(e)},
            "n_train_rows": int(len(train_bucket)), "n_eval_instances": int(n_eval),
            "n_full_bucket_instances": int(len(bucket_keys)),
        },
    )


if __name__ == "__main__":
    main()
