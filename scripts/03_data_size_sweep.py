"""Training-data-size breakpoint curve (RQ2): refit EMOS at increasing amounts of
reforecast training data, under two sampling arms and two pooling variants; TimesFM-3
and raw ensemble are zero-shot / N-independent and reused as flat reference lines from
the already-persisted Phase 2 results, restricted to the exact same evaluation
instances EMOS is scored on (see "Instance-set join" below — fix-round-1 finding 1).

"N days" is operationalized as k = round(N / DAYS_PER_CASE) (year_idx, time_idx)
issue-date pairs out of the full 20-year x 209-date reforecast archive, where
DAYS_PER_CASE is MEASURED directly from the archive (see below), not assumed. Two
arms pick WHICH k pairs:
  - contiguous (primary): the k pairs closest in time to the test period, working
    backward — simulates a real newly-deployed station's limited, chronologically
    contiguous history (small N sees roughly one season, not all four).
  - random (secondary, 5 seeds, mean +/- std reported): k pairs drawn uniformly at
    random from the full archive — gives artificial full-season coverage even at small
    N. The gap between the two arms' curves is itself a reported finding about the
    value of seasonal coverage, not a discrepancy to explain away.
See this plan's Task 4 "Before You Begin" for the full rationale.

Step 0 finding, RE-MEASURED directly in fix-round-1 (2026-09-03, SSH altay) after
review flagged the original ISSUE_DATES_PER_YEAR=209 constant as wrong. Measured via
`len(ds.year)`, `len(ds.time)`, `ds.time.values[0]`/`[-1]`, and `np.diff` on the raw
archive: `len(ds.year)=20`, `len(ds.time)=209`, `ds.time` spans
2017-01-02T00:00:00 to 2018-12-31T00:00:00 (728.0 days exactly), with `np.diff`
alternating 3-day/4-day steps (median 3.5 days) — i.e. the archive's 209 "time"
values are ONE ~2-year template cycle of issue-date LABELS, re-used identically
across all 20 analog years (the separate "year" dimension), not 209 dates repeated
once per analog year. MEASURED DAYS_PER_CASE = 728.0 / 209 = 3.4833 days/case (see
the DAYS_PER_CASE constant below) — NOT the previously assumed 365/209 = 1.746. The
original constant silently doubled every "N days" label's real case count and halved
every reported "calendar days" figure; every k and every calendar-day-equivalent
number in this task's report was re-derived from this measured constant.

Ascending time_idx = later in the ~2-year cycle is CONFIRMED real (real datetime64
values, measured above). The separate "year" coordinate (int64, 1..20, bare
analog-year enumeration) has NO confirmed real calendar meaning — its polarity
("ascending year_idx = more recent analog year") was NOT verified. Its impact is
bounded, not "confirmed safe" and not "unbounded": every tested N whose k stays
within a single year_idx block (k <= 209, the archive's per-year_idx pair count) is
SAFE regardless of year_idx polarity, because contiguous sampling within one block
depends only on time_idx ordering, which WAS verified. With the corrected
DAYS_PER_CASE, the tested N values map to k = 9 (N=30), 26 (N=90), 105 (N=365),
314 (N=1095), 4180 ("full", measured directly, independent of any polarity). Only
N=1095 (k=314 > 209) spans multiple year_idx blocks and is therefore the one tested
point whose training-data identity could differ under the opposite year_idx polarity;
"full" uses every pair regardless of ordering, so polarity is irrelevant there
too. See the task report's "Fix round 1 (revised)" section for the full statement.

Instance-set join (fix-round-1 finding 1): `results/phase2_comparison_raw.parquet`'s
raw_ensemble/emos/tsfm3 rows are already restricted, by `02_run_tsfm.py`, to test
instances with sufficient clean lookback context for TimesFM-3 (a strict subset of
`build_test_long_table`'s full instance set). Comparing EMOS metrics computed on the
FULL test set against tsfm3/raw_ensemble metrics computed on that FILTERED subset
silently compares two different evaluation sets. Fixed here by inner-joining EMOS's
test instances onto the parquet's (station_id, valid_time, step_hours) instance keys
BEFORE computing any EMOS metric, so every method in every row of this sweep's output
is scored on the identical instance set. Instance counts for both sides and the
matched intersection are printed and included in the task report.

EMOS is fit two ways at each N: "pooled" (one global model across all stations, as in
Phase 2) and "local" (one model per station, fit only on that station's rows in the
subsample). Local EMOS needs >= LOCAL_EMOS_MIN_ROWS rows per station to fit reliably
(4 free parameters); at small N most stations won't have enough contiguous rows, so the
local curve is reported only over the stations it could actually cover, with that
coverage fraction reported alongside it — never silently filled in or extrapolated.
`fit_predict_local_emos`'s coverage fraction denominator is the number of unique
stations in the TEST set (fix-round-1 finding 5 — it was previously the number of
unique stations in the training subset, which is the wrong population and can hide
dropped test stations if train/test station sets ever differ).

Reforecasts have 11 ensemble members (germany_ensemble_reforecasts_t2m.nc); the test-
period forecasts EMOS is evaluated against have 51. ens_var computed from 11 members is
a noisier (higher-variance) estimator of the true ensemble spread than one from 51 would
be, at every N equally — this is a fixed property of the training archive, not something
that changes with N, and is not corrected here. Documented as a limitation in the task
report, not fixed in code (no clean fix exists without re-simulating the ensemble).

DRN (Rasp & Lerch 2018, Task 5): a third trained method added to the contiguous arm
only, alongside emos_pooled/emos_local, on the identical post-join test_X/obs_values/
test_station_ids. DRN's per-station embedding already models station-specific structure
the way emos_local does, so there is no separate local/pooled DRN split -- one
method="drn" curve is enough. The random arm and multi-seed treatment stay EMOS-only
(out of scope for DRN, same scope note as emos_local's random-arm exclusion). DRN's
training seed is config.data_size_sweep_seeds[0] (a real config value, not a new
hardcoded seed). DRN participates in the crps/coverage_80pct breakpoints (same
tsfm3 reference as emos_pooled/emos_local) but not interval_width_k, for the identical
"sharpness has no inherent direction" reason those two already follow.

Fix round 1 (task 5 review) A: every row for a method that actually fits a model
carries a "fit_seconds" column (wall-clock time.perf_counter() around that
method's fit()+predict_quantiles() call at that N) -- DRN retrains from scratch at
every N up to N=full's ~4.28M rows with a FIXED epoch budget regardless of N (see
DRN's class docstring), so the scaling story (does DRN's wall-clock cost grow with
N the way EMOS's does, or does the fixed-epoch design make it flatter/steeper) is
itself a reportable finding, not just the metrics. "fit_seconds" is None for rows
where nothing was actually fit at that row (N=0 undefined rows, the N=full random-
arm reuse rows since no refit happens there by construction, and the N-independent
raw_ensemble/tsfm3 reference rows, which are zero-shot/already-computed).
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
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"

# --- Fix-round-1 (revised) R1: MEASURED days-per-case constant, not assumed ---
# Measured directly on altay (2026-09-03) via:
#   len(ds.year), len(ds.time)                          -> 20, 209
#   ds.time.values[0], ds.time.values[-1]                -> 2017-01-02, 2018-12-31
#   np.diff(ds.time.values[:10])                          -> alternating 3/4-day steps
#   (ds.time.values[-1] - ds.time.values[0]) / 1-day      -> 728.0 days exactly
# DAYS_PER_CASE = measured_span_days / len(ds.time), matching the reviewer-specified
# derivation `days_per_case = total_calendar_day_span / len(ds.time)`. The prior
# constant (ISSUE_DATES_PER_YEAR = 209, i.e. days_per_case = 365/209 = 1.746) was
# wrong by ~2x versus this measured value; every "N days" -> case-count (k)
# conversion, and its inverse, used it.
REFORECAST_TOTAL_ISSUE_DATES = 209  # measured: len(ds.time)
REFORECAST_ISSUE_DATE_SPAN_DAYS = 728.0  # measured: (ds.time.max() - ds.time.min()) in days
DAYS_PER_CASE = REFORECAST_ISSUE_DATE_SPAN_DAYS / REFORECAST_TOTAL_ISSUE_DATES  # measured ~3.4833
LOCAL_EMOS_MIN_ROWS = 5

# --- Task 6 E3: low-N grid, k=1..7. Driven by k DIRECTLY (via the exact round-trip
# below), not by n_days, since n_days_to_k's round() rounds coarsely at small N (e.g.
# a human-friendly n_days=10 doesn't necessarily map back to k=3 exactly). See
# n_days_for_exact_k() and the "E3 low-N grid" block in main().
LOW_N_K_GRID = [1, 2, 3, 5, 7]

# --- Task 6 E1(b): a genuinely zero-shot fixed-multiplier variance-inflation
# baseline, cited from the published ensemble spread-skill literature rather than
# fit to any of this project's own data (fitting it here would defeat the point of
# E1(b) -- a baseline with NO fitting step at all, matching TimesFM-3's own
# zero-shot status). ECMWF EPS spread-skill ratios (SSR = ensemble spread / RMSE of
# the ensemble mean; SSR=1 means perfectly calibrated spread) for surface
# temperature at medium range are widely reported in the literature as sub-1
# (under-dispersed), commonly cited in the ~0.6-0.8 range (Buizza, R., 1997,
# "Potential Forecast Skill of Ensemble Prediction and Spread and Skill
# Distributions of the ECMWF Ensemble Prediction System", Mon. Wea. Rev. 125,
# 99-119; Fortin, V. et al., 2014, "Why Should Ensemble Spread Match the RMSE of
# the Ensemble Mean?", J. Hydrometeor. 15, 1708-1713). No single precise scalar is
# universally cited for EUPPBench's specific station/lead-time subset, so
# LAMBDA_CLIM = 1/SSR uses a representative round point (SSR=0.67) from that
# reported range rather than one paper's exact figure re-derived for this exact
# archive -- documented as a deliberate approximation, not a load-bearing citation.
# See the Task 6 report's E1 section for the full judgment-call writeup.
LAMBDA_CLIM = 1.5

# Set from Step 0's verified finding (time_idx ordering is real; year_idx ordering is
# assumed — see module docstring). CHRONOLOGICAL_DESCENDING = True: sort descending by
# (year_idx, time_idx) so the largest values (assumed/confirmed most recent) come first.
CHRONOLOGICAL_DESCENDING = True


def n_days_to_k(n_days) -> int | str:
    """Convert an 'N days' label to a case count k, or pass through 'full'. Uses the
    MEASURED DAYS_PER_CASE constant (fix-round-1 revised R1) — round(N / DAYS_PER_CASE)."""
    if n_days == "full":
        return "full"
    return round(n_days / DAYS_PER_CASE)


def k_to_calendar_days(k: int) -> int:
    """Inverse of n_days_to_k's ratio, for reporting 'N cases (M calendar days)'."""
    return round(k * DAYS_PER_CASE)


def bp_to_calendar_days(bp: float | None) -> int | None:
    """Fix-round-1 finding 3: convert an interpolated (non-integer) breakpoint k
    directly to calendar days via a SINGLE rounding at the very end
    (round(bp * DAYS_PER_CASE)), instead of rounding bp to an integer k FIRST and
    THEN converting via k_to_calendar_days (round(k) then round(k * DAYS_PER_CASE)
    again). The double-rounding version inflates the reported label: e.g.
    bp=1.64 previously rounded to k=2 first, then 2 * 3.4833 ~= 7 days -- overstating
    the true 1.64 * 3.4833 ~= 5.7 days by ~23%. Returns None if bp is None (no
    crossing observed)."""
    if bp is None:
        return None
    return round(bp * DAYS_PER_CASE)


def n_days_for_exact_k(k: int) -> float:
    """Task 6 E3: the inverse of n_days_to_k's ratio, chosen so that
    n_days_to_k(n_days_for_exact_k(k)) == k EXACTLY (round(k) == k for any integer
    k, no rounding error) -- this is how sample_contiguous/sample_random get
    "driven by k directly" per the task brief while still being called with their
    existing n_days-shaped signature, verbatim, with no change to either function.
    Contrast with picking a human-friendly integer n_days (e.g. 10) and hoping it
    happens to round-trip back to the desired k via n_days_to_k -- that's exactly
    the coarse rounding at small N the brief calls out."""
    return k * DAYS_PER_CASE


def k_and_calendar_days(n_days, n_pairs_full: int) -> tuple[int, int | None]:
    """Single source of truth for the (k, calendar_days) pair used to label every row
    for a given n_days — fix-round-1 minor item: this was previously duplicated
    (identically, but separately) three times in main(), which is a drift risk if one
    copy is ever edited without the others. "full" always resolves k to the actual
    measured pair count (n_pairs_full), never through the N/DAYS_PER_YEAR ratio, and
    always reports calendar_days as None (undefined — "full" already means "all of
    it", there is no equivalent calendar-day count to report)."""
    if n_days == "full":
        return n_pairs_full, None
    k = n_days_to_k(n_days)
    return k, k_to_calendar_days(k)


def sample_contiguous(full_train: pd.DataFrame, n_days) -> pd.DataFrame:
    """The k pairs closest in time to the test period, working backward (deterministic,
    no seed) — see module docstring and this plan's Task 4 Step 0 for the verified
    chronological-ordering rationale behind CHRONOLOGICAL_DESCENDING."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    ordered = unique_pairs.sort_values(
        ["year_idx", "time_idx"], ascending=not CHRONOLOGICAL_DESCENDING
    )
    sampled_pairs = ordered.iloc[:k]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def sample_random(full_train: pd.DataFrame, n_days, seed: int) -> pd.DataFrame:
    """k pairs drawn uniformly at random from the full archive (fixed seed) — the
    prior single-arm design, kept as the secondary arm."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(unique_pairs), size=k, replace=False)
    sampled_pairs = unique_pairs.iloc[sampled_idx]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def compute_metrics(y: np.ndarray, preds: np.ndarray, quantile_levels: list[float]) -> dict:
    """y: (n,1), preds: (n,1,n_quantiles). Returns crps, twcrps (tail-weighted CRPS
    at the upper two quantile levels 0.8/0.9, per Methods Section 2.3's definition
    -- D1, 04_results.tex review: added here, the one function every method/variant
    in this script routes through, so twCRPS is now computed identically for all of
    them rather than only the 3 methods results/phase3_summary_metrics.parquet
    happened to have it for), coverage_80pct (nominal 80% band from the q0.1/q0.9
    pair, per this plan's Task 1 coverage-label fix), and interval_width_k (mean
    p10-p90 width in Kelvin, i.e. sharpness)."""
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
    """One EMOS per station, fit only on that station's rows in train_subset. A test
    row whose station has < min_rows training rows (or none at all) is EXCLUDED, not
    filled in with a pooled fallback — the covered_mask and coverage_fraction make that
    exclusion explicit and reportable rather than silent.

    coverage_fraction's denominator is the number of unique TEST stations (fix-round-1
    finding 5) — NOT the number of unique stations in train_subset, which is the wrong
    population: a test station entirely absent from a small training subsample would
    previously be excluded from both numerator and denominator, letting
    coverage_fraction read 100% while test rows for that station were silently
    dropped from the metrics. The fix makes the denominator fixed (= all test
    stations this call is asked to predict for), so a dropped test station now
    visibly lowers the fraction instead of hiding behind it."""
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
    """Linear-interpolated N (in cases) where method_values crosses reference_value,
    walking n_cases_axis in ascending order. better='lower': method starts worse
    (higher) than reference and crosses to better (lower) — e.g. CRPS. better='closer_to_ref':
    for coverage, 'better' means the gap |value - reference| shrinks past the reference's
    own gap to nominal (not used here since TimesFM-3 IS the reference; instead this mode
    finds where method's coverage crosses the reference's coverage value directly, same as
    'lower'/'higher' but the caller passes the coverage direction it wants). Returns None if
    no crossing occurs within the tested range."""
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
    """(n_cases, value) pairs with real (non-NaN) data, sorted ascending by n_cases.
    Used instead of raw method_values[0] anywhere a 'starting value' or 'trend' needs
    to be inferred — index [0] of the raw list is always the n_days=0 'no data to fit'
    NaN row once arm_df is built in main(), and NaN compares False against everything
    in Python, so using it directly silently defaults any direction decision to the
    wrong branch (a bug caught once already on the coverage breakpoint)."""
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
    """Wraps find_breakpoint with a per-metric direction rule AND a self-documenting
    'crossing_direction' reason string (fix-round-1 findings 6/7, revised per a second
    review pass — see module docstring "Fix round 1 (revised)" in the task report).

    Restructured as a single if/elif/else over `metric` with an explicit `raise` on an
    unrecognized metric — each metric's rule is fully self-contained, no `better`
    variable that can leak a stale value between loop iterations if a 4th metric is
    added later.

    `interval_width_k` is DELIBERATELY NOT handled here (revised finding R3b): sharpness
    has no inherent "better" direction on its own — per Gneiting's "maximize sharpness
    subject to calibration," it is only meaningful jointly with calibration, so a
    computed "crossing point" for width alone would look interpretable but mean
    nothing. main() reports interval_width_k as a curve only (in the main sweep
    results table and the figure), never in the breakpoints table; calling this
    function with metric="interval_width_k" is a caller error.

    `coverage_80pct`'s direction is now inferred from the OVERALL TREND across every
    real data point (an ordinary-least-squares slope sign), not from a single point
    (revised finding R3) — inferring from one point is fragile to that one point being
    noisy or (as happened before this fix) accidentally NaN.

    Returns (breakpoint_n_cases_or_None, crossing_direction reason string). The reason
    string is written into results/phase3_data_size_sweep_breakpoints.parquet so a
    downstream reader can tell "already better at smallest N" (no crossing needed)
    apart from "no crossing observed" (never catches up) apart from an actual crossing
    and its direction, without needing this conversation's context (finding 7)."""
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
        raise ValueError(
            f"breakpoint_and_direction has no direction rule for metric {metric!r} — "
            "add one (explicitly) before calling find_breakpoint for a new metric; "
            "this is a deliberate fail-fast instead of silently reusing a stale rule. "
            "Note: interval_width_k is intentionally excluded — see this function's docstring."
        )


def _catch_overflow_warnings(fn, *args, **kwargs):
    """Cheap insurance (fix-round-1 minor item): EMOS.fit's L-BFGS-B optimization was
    observed to occasionally trigger `RuntimeWarning: overflow encountered in exp`
    during finite-difference gradient probing. A single warning does not necessarily
    mean non-convergence (L-BFGS-B usually steps back into a valid region and still
    converges), but with up to 49 per-station fits per N in fit_predict_local_emos, a
    genuinely problematic station fit would otherwise be invisible in the aggregate
    metrics. This wraps a call and returns (result, n_overflow_warnings) — a per-N,
    per-call COUNT of triggered warnings, not a definitive convergence verdict. Does
    not modify EMOS.fit itself (out of this task's declared file scope) or either
    fit_predict_*_emos function's signature (both are reusable interfaces Task 5
    imports per the plan)."""
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

    # Single pass: aggregate per (station_id, valid_time, step_hours) instance AND
    # capture the key columns row-aligned with the aggregated arrays, needed for the
    # instance-set join below (fix-round-1 finding 1). Previously this used a second,
    # separate call to `grouped.groups.keys()` to recover station_id alone; capturing
    # all three key columns inside the same loop that builds the aggregates is both
    # necessary for the join and incidentally removes any reliance on two separate
    # GroupBy iterations agreeing on row order.
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

    # --- Instance-set join (fix-round-1 finding 1) ---
    key_cols = ["station_id", "valid_time", "step_hours"]
    raw = pd.read_parquet(RAW_RESULTS_PATH)  # loaded early: needed for the join, not just the reference lines
    per_method_keys = {m: raw.loc[raw["method"] == m, key_cols].drop_duplicates() for m in raw["method"].unique()}
    canonical_key_set = per_method_keys["tsfm3"]
    methods_share_identical_keys = all(
        len(canonical_key_set.merge(per_method_keys[m], on=key_cols, how="inner")) == len(canonical_key_set) == len(per_method_keys[m])
        for m in per_method_keys
    )
    print(f"instance-set join: raw_ensemble/emos/tsfm3 share an identical instance-key set: {methods_share_identical_keys}")
    assert methods_share_identical_keys, (
        "raw_ensemble/emos/tsfm3 instance keys diverge in results/phase2_comparison_raw.parquet — "
        "filtering every method to tsfm3's key set would silently score raw_ensemble/emos on the "
        "wrong instances. This must not be silently tolerated (see fix-round-1 finding 1)."
    )

    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    n_test_instances = len(test_keys_df)
    n_parquet_instances = len(canonical_key_set)
    n_matched = len(matched_keys)
    subset_note = (
        "parquet is a strict subset of the EMOS test set (expected)"
        if n_matched == n_parquet_instances
        else "PARTIAL OVERLAP -- parquet has instances outside the EMOS test set, investigate before trusting this run"
    )
    print(
        f"instance-set join: EMOS build_test_long_table instances={n_test_instances}, "
        f"parquet reference instances={n_parquet_instances}, matched intersection={n_matched} ({subset_note})"
    )
    assert n_matched == n_parquet_instances, (
        f"instance-set join did not fully cover the parquet reference instances "
        f"({n_matched} matched vs {n_parquet_instances} in parquet) — refusing to proceed with "
        "a partial join, since that would silently score references and EMOS on different sets again."
    )

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
    e2_cache = {}  # Task 6 E2: populated at n_days in (30, 90), see inside the loop
    for n_days in data_size_days:
        n_overflow_this_ndays = 0

        # --- contiguous arm (primary) ---
        train_contig = sample_contiguous(full_train, n_days)
        pooled_metrics = None  # reused below to skip redundant refitting at n_days == "full"
        if len(train_contig) == 0:
            # n_days == "full" always yields the full non-empty archive, so this
            # branch is unreachable for "full" in practice — but k must still be
            # strictly numeric-or-None (never the literal string "full") for
            # write_result's parquet write.
            k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)
            print(f"N={n_days} days, arm=contiguous (0 training rows): EMOS undefined, no data to fit")
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

            # --- Task 6 E2: cache per-instance CRPS/coverage-indicator arrays for
            # emos_pooled at exactly n_days=30 (k=9) and n_days=90 (k=26) -- the two
            # N's Task 4's headline sentence names -- for the significance test
            # against tsfm3 run once after this loop. Kept keyed by n_days (not k)
            # since that's the loop variable already in scope; e2_cache stays empty
            # for every other n_days (cheap: two small arrays, no extra model fit).
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

            # --- DRN (Rasp & Lerch 2018): contiguous arm only, one pooled-with-
            # per-station-embedding model per N (see this task's report for the
            # random-arm/multi-seed out-of-scope note, same rationale as
            # emos_local). Trained seed comes from config.data_size_sweep_seeds[0]
            # (a real config value, not a new hardcoded seed) scored on the exact
            # same post-join test_X/obs_values/test_station_ids emos_local uses.
            t0 = time.perf_counter()
            drn_train = train_contig[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
            drn_model = DRN(quantile_levels=quantile_levels, seed=sweep_seeds[0]).fit(drn_train)
            drn_preds = drn_model.predict_quantiles({
                "ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"], "station_id": test_station_ids,
            })
            drn_fit_seconds = time.perf_counter() - t0
            drn_metrics = compute_metrics(obs_values, drn_preds, quantile_levels)
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "drn",
                **drn_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None, "fit_seconds": drn_fit_seconds,
            })
            print(f"N={n_days} days ({len(train_contig)} training rows, {drn_fit_seconds:.2f}s): DRN CRPS={drn_metrics['crps']:.4f}")

            # Fix round 1 (task 5 review) B: DRN uses a FIXED epoch budget (50 by
            # default) regardless of N, with no early stopping (see class
            # docstring's disclosed deviation from Rasp & Lerch 2018). At the
            # largest N ("full", ~4.28M rows) this is the training regime most at
            # risk of not having converged within the fixed budget, so print the
            # loss trajectory here specifically (first/mid/last epochs) as the
            # convergence check the task report cites, rather than silently
            # trusting "loss decreased" without ever looking at the curve.
            if n_days == "full":
                lh = drn_model.loss_history_
                mid = len(lh) // 2
                print(
                    f"N=full DRN loss_history_ ({len(lh)} epochs): "
                    f"epoch0={lh[0]:.4f}, epoch{mid}={lh[mid]:.4f}, "
                    f"epoch{len(lh) - 2}={lh[-2]:.4f}, epoch{len(lh) - 1}={lh[-1]:.4f}, "
                    f"last-10-epoch range=[{min(lh[-10:]):.4f}, {max(lh[-10:]):.4f}]"
                )

            # --- Task 6 E1(a): variance-inflation baseline, multiplier estimated
            # from the SAME k-sized contiguous training subset EMOS/DRN just used
            # ("1 parameter / k cases" vs. EMOS's "4 parameters / k cases") -- see
            # VarianceInflationBaseline's docstring. Never fit on test data.
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

        # --- random arm (secondary, 5 seeds) ---
        k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)

        if n_days == "full":
            # sample_random(full_train, "full", seed) returns full_train unchanged
            # for every seed — identical to the contiguous fit above by
            # construction, since there is nothing left to subsample from. Refitting
            # pooled EMOS 5 more times (plus the contiguous fit = 6x) on the full
            # archive was the single most expensive repeated computation in the
            # original run. Reuse the contiguous fit's pooled_metrics instead; std is
            # exactly 0.0 here BY CONSTRUCTION, not an empirical finding (there is no
            # seed-to-seed variation possible at N=full).
            assert pooled_metrics is not None, "full-N contiguous fit must have run before reuse"
            for seed in sweep_seeds:
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "random", "seed": seed, "method": "emos_pooled",
                    **pooled_metrics,
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None, "fit_seconds": None,  # reused, no refit — see module docstring
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
                # nanmean over an all-NaN list (every seed had 0 training rows, e.g.
                # N=0) would raise "RuntimeWarning: Mean of empty slice" and return
                # NaN anyway -- short-circuit to explicit NaN instead of letting the
                # warning fire for a value that's NaN either way.
                "n_stations_covered": None,
                "fit_seconds": float(np.nanmean(seed_fit_seconds)) if not all(np.isnan(seed_fit_seconds)) else float("nan"),
            }
        rows.append(mean_row)
        print(f"N={n_days} days, arm=random_mean, method=emos_pooled: {mean_row}")

        n_overflow_by_ndays[str(n_days)] = n_overflow_this_ndays
        if n_overflow_this_ndays:
            print(
                f"N={n_days} days: {n_overflow_this_ndays} EMOS.fit call(s) triggered an "
                "'overflow encountered in exp' RuntimeWarning during optimization (proxy for "
                "potential optimizer instability, NOT a definitive non-convergence signal — "
                "see module docstring)."
            )

    # --- N-independent reference lines: raw_ensemble, tsfm3, restricted to the same
    #     matched instance set as EMOS (fix-round-1 finding 1) ---
    quantile_cols = [f"q{q}" for q in quantile_levels]
    tsfm3_ref_metrics = None  # captured for Fix-round-1 Blocking Fix 3's coverage target below
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
                "n_stations_covered": None, "fit_seconds": None,  # zero-shot/already-computed
            })
        print(f"method={method} (N-independent, {len(df_method)} matched instances): {ref_metrics}")
    assert tsfm3_ref_metrics is not None, "tsfm3 reference metrics were not captured"

    # --- Task 6 E1(b): fixed-climatological-multiplier variance-inflation baseline.
    # Genuinely zero-shot -- no training data, no fit() call, matching TimesFM-3's
    # own zero-shot status (see LAMBDA_CLIM's citation/caveat comment above). One
    # N-independent reference row, exactly like raw_ensemble/tsfm3 above.
    vi_fixed_model = VarianceInflationBaseline.from_fixed_multiplier(LAMBDA_CLIM, quantile_levels=quantile_levels)
    vi_fixed_preds = vi_fixed_model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})
    vi_fixed_metrics = compute_metrics(obs_values, vi_fixed_preds, quantile_levels)
    for n_days in data_size_days:
        rows.append({
            "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
            "sampling_arm": "n_independent", "seed": None, "method": "var_inflation_fixed",
            **vi_fixed_metrics,
            "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
            "n_stations_covered": None, "fit_seconds": None,  # zero-shot, no fit at all
        })
    print(f"method=var_inflation_fixed (N-independent, multiplier={LAMBDA_CLIM}): {vi_fixed_metrics}")

    # --- Fix-round-1 Blocking Fix 3: coverage-matched variance-inflation baseline
    # (variant (c)). target_coverage is TimesFM-3's REAL, measured coverage@[0.1,0.9]
    # captured just above (tsfm3_ref_metrics["coverage_80pct"]) -- never hardcoded.
    # Calibrated on the FULL training reforecast archive (full_train, every (year_idx,
    # time_idx) pair), not the k=9 subset variant (a) (var_inflation_trainfit) uses:
    # this baseline is meant to be compared fairly against TimesFM-3's full-data test
    # performance, and var_inflation_trainfit_full (scripts/06_lead_time_grouped_analysis.py)
    # already established the "fit on the full archive" convention for exactly this
    # kind of full-data variance-inflation comparison -- kept consistent with that
    # precedent rather than picking a third, unrelated training-subset convention.
    # No test-set coverage is EVER looked at while finding lambda_c (see
    # VarianceInflationBaseline.from_coverage_target's docstring) -- only full_train's
    # own t2m_obs/ens_mean/ens_var are used, exactly the same leakage discipline as
    # variant (a)'s .fit(train_contig).
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
            "n_stations_covered": None, "fit_seconds": None,  # zero-shot application; lambda_c fit on train only
        })
    print(
        f"method=var_inflation_coverage_matched (N-independent, lambda_c={vi_covmatch_model.multiplier:.4f}, "
        f"target_coverage={tsfm3_ref_metrics['coverage_80pct']:.4f} [tsfm3's real coverage], "
        f"fit on full training archive): {vi_covmatch_metrics}"
    )
    print(
        "Blocking Fix 3 headline: at matched coverage "
        f"({vi_covmatch_metrics['coverage_80pct']:.4f} vs tsfm3's {tsfm3_ref_metrics['coverage_80pct']:.4f}), "
        f"coverage-matched interval width={vi_covmatch_metrics['interval_width_k']:.4f}K vs. "
        f"tsfm3's {tsfm3_ref_metrics['interval_width_k']:.4f}K "
        f"({'NARROWER' if vi_covmatch_metrics['interval_width_k'] < tsfm3_ref_metrics['interval_width_k'] else 'WIDER'})."
    )

    # --- B2(ii) (04_results.tex review, 2026-09-07): leakage-optimistic upper
    # bound for the trivial-rescaling baseline. Variant (c) above is leakage-free
    # by construction (lambda fit on TRAIN, applied unchanged to TEST). This
    # variant is the deliberately leakage-CONTAMINATED counterpart: lambda is fit
    # directly on the TEST set's own coverage via the same
    # from_coverage_target root-finder, just handed the test set instead of
    # full_train. It answers a different question than (c) does -- not "what can
    # a real, leakage-free trivial baseline achieve" but "what is the ceiling on
    # what ANY one-parameter rescaling could achieve here, even with an unfair
    # advantage." Never report this as a legitimate baseline result on its own;
    # it exists to bound how much of the TimesFM-3-vs-coverage-matched width gap
    # (variant c: 6.7224K vs TimesFM-3's 5.1379K, full data) could in principle
    # close if leakage were the only thing holding variant (c) back. If this
    # variant's width still exceeds TimesFM-3's, the width gap is not a leakage
    # artifact of (c)'s train/test split -- it is real. Labeled
    # "leakage-optimistic" (never "leakage-free" or unqualified) everywhere it
    # is reported, per the explicit instruction that motivated adding it.
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
            "n_stations_covered": None, "fit_seconds": None,  # leakage-optimistic upper bound, not a real baseline
        })
    print(
        f"method=var_inflation_test_leakage (N-independent, LEAKAGE-OPTIMISTIC -- "
        f"lambda={vi_leaky_model.multiplier:.4f} fit DIRECTLY on test coverage, "
        f"target_coverage={tsfm3_ref_metrics['coverage_80pct']:.4f} -- upper bound only, "
        f"never a legitimate baseline): {vi_leaky_metrics}"
    )
    print(
        "B2(ii) headline: leakage-optimistic interval width="
        f"{vi_leaky_metrics['interval_width_k']:.4f}K vs. tsfm3's "
        f"{tsfm3_ref_metrics['interval_width_k']:.4f}K "
        f"({'NARROWER -- TimesFM-3 advantage may be a leakage artifact of variant (c), investigate' if vi_leaky_metrics['interval_width_k'] < tsfm3_ref_metrics['interval_width_k'] else 'WIDER -- TimesFM-3 advantage over trivial rescaling holds even at this optimistic upper bound'})."
    )

    sweep_df = pd.DataFrame(rows)

    # Dtype audit: n_days is the ONLY string-valued column. n_cases and
    # n_calendar_days_equiv must be strictly numeric-or-None (never the literal
    # string "full") or pyarrow's Table.from_pandas raises ArrowInvalid trying to
    # unify a mixed int/str column — exactly what crashed the first real run here.
    def _numeric_or_none(v):
        return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, float) and np.isnan(v))

    assert sweep_df["n_cases"].map(_numeric_or_none).all(), "n_cases contains a non-numeric value (e.g. literal 'full')"
    assert sweep_df["n_calendar_days_equiv"].map(_numeric_or_none).all(), "n_calendar_days_equiv contains a non-numeric value"

    write_result(
        sweep_df,
        name="phase3_data_size_sweep",
        model_version="phase3-sweep-v6",
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

    # --- Breakpoints (fix-round-1 findings 6/7, revised R3b) ---
    # interval_width_k is DELIBERATELY excluded: sharpness has no inherent "better"
    # direction on its own (Gneiting: "maximize sharpness subject to calibration" —
    # it is only meaningful jointly with calibration), so a computed crossing point
    # for width alone would look interpretable but mean nothing. It is reported as a
    # curve only, in sweep_df / results/phase3_data_size_sweep.parquet and the
    # figure, never in this breakpoints table.
    tsfm3_row = sweep_df[(sweep_df["method"] == "tsfm3")].iloc[0]
    breakpoint_rows = []
    for metric in ["crps", "coverage_80pct"]:
        reference_value = tsfm3_row[metric]
        # "emos_variant" here also carries "drn" (Task 5) and "var_inflation_trainfit"
        # (Task 6 E1) — kept as the same column name/loop variable rather than
        # renamed, since "emos_variant" is already the persisted schema of
        # results/phase3_data_size_sweep_breakpoints.parquet and both get the
        # identical treatment (crps/coverage_80pct only, same tsfm3 reference, no
        # interval_width_k breakpoint — see module docstring).
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
    assert breakpoints_df["breakpoint_n_cases"].map(_numeric_or_none).all(), "breakpoint_n_cases contains a non-numeric value"
    assert breakpoints_df["breakpoint_calendar_days"].map(_numeric_or_none).all(), "breakpoint_calendar_days contains a non-numeric value"

    write_result(
        breakpoints_df,
        name="phase3_data_size_sweep_breakpoints",
        model_version="phase3-sweep-v6",
        config={
            "data_size_days": data_size_days,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "chronological_descending": CHRONOLOGICAL_DESCENDING,
            "days_per_case_measured": DAYS_PER_CASE,
        },
    )

    # --- Task 6 E2: apply Task 2's station-blocked significance test to Task 4's
    # headline coverage/CRPS gaps at k=9 (n_days=30) and k=26 (n_days=90), for
    # emos_pooled vs tsfm3. Uses the e2_cache captured inside the main loop above
    # (emos_pooled's per-instance CRPS/coverage-indicator arrays, in matched_keys'
    # row order) against tsfm3's per-instance arrays built HERE in that exact same
    # row order via a left-merge on matched_keys (not `raw`'s own incidental row
    # order) -- station_blocked_paired_test's `diff = loss_a - loss_b` is a
    # per-INSTANCE paired differential, so misaligned row order between the two
    # arrays would silently pair the wrong instances together.
    tsfm3_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all(), (
        "left-merging tsfm3 rows onto matched_keys produced a row with no match — "
        "matched_keys should be a subset of tsfm3's own instance keys by construction "
        "(see the instance-set join above), so this would mean that invariant broke."
    )
    # Fix-round-1 minor item: an explicit row-COUNT assertion alongside the notna
    # check above -- a left-merge that (e.g. via a duplicate key on the right side)
    # fans out to MORE rows than matched_keys would still pass the notna check
    # (every row would still have a real "obs" value) while silently breaking the
    # 1:1 row alignment station_blocked_paired_test's paired differential depends on.
    assert len(tsfm3_ordered) == n_matched, (
        f"tsfm3_ordered has {len(tsfm3_ordered)} rows but matched_keys (and therefore "
        f"emos_pooled's e2_cache arrays) has {n_matched} -- the left-merge must be exactly "
        "1:1, or the per-instance pairing used by station_blocked_paired_test is broken."
    )
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
                f"E2 significance: N={n_days_e2} days (k={k_e2}), metric={metric_name}, "
                f"emos_pooled vs tsfm3: mean diff(a-b)={test_result['block_mean_diff']:.4f}, "
                f"t p={test_result['t_pvalue']:.4f}, wilcoxon p={test_result['wilcoxon_pvalue']:.4f}"
            )

    significance_df = pd.DataFrame(significance_rows)
    write_result(
        significance_df,
        name="phase3_data_size_sweep_significance",
        model_version="phase3-sweep-v6",
        config={"quantile_levels": quantile_levels, "compared_n_days": sorted(e2_cache.keys())},
    )

    # --- Task 6 E3: low-N grid, k=1,2,3,5,7 (roughly N~=3,7,10,17,24 calendar days
    # per DAYS_PER_CASE), driven by k DIRECTLY via n_days_for_exact_k (see its
    # docstring) rather than by a human-friendly n_days label -- avoids the coarse
    # rounding n_days_to_k would otherwise introduce at these very small N. Reuses
    # sample_contiguous/fit_predict_pooled_emos/fit_predict_local_emos/
    # compute_metrics verbatim on the SAME already-joined full_train/test_X/
    # obs_values/test_station_ids as the main sweep loop above -- no new data
    # loading, milliseconds per fit.
    #
    # Column note (fix-round-1 minor item): this table's case-count column is named
    # "k" (not "n_cases", as in the main sweep's phase3_data_size_sweep.parquet).
    # Documented rather than unified: this table is driven BY k directly (see
    # n_days_for_exact_k above) with no "n_days" label at all, so "k" is the more
    # honest name here -- unifying to "n_cases" would suggest it was derived FROM an
    # n_days axis the way the main sweep's column is, which it is not.
    #
    # Fix-round-1 finding 2: k=1 and k=2's contiguous draws are the two points the
    # E3 breakpoint (k~1.64) actually rests on, and a single contiguous draw at each
    # k has no replication. This adds the SAME 5-seed random arm the main sweep
    # already uses (sample_random reused verbatim, identical pattern) at every
    # LOW_N_K_GRID k, so those two points get a random-arm mean+std counterpart to
    # compare against -- real new compute, but cheap (milliseconds per fit, same as
    # the contiguous arm above).
    low_n_rows = []
    for k_low in LOW_N_K_GRID:
        synthetic_n_days = n_days_for_exact_k(k_low)
        n_calendar_low = k_to_calendar_days(k_low)

        train_contig_low = sample_contiguous(full_train, synthetic_n_days)
        actual_k = len(train_contig_low[["year_idx", "time_idx"]].drop_duplicates())
        assert actual_k == k_low, f"n_days_for_exact_k round-trip failed: wanted k={k_low}, got {actual_k}"

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
        print(f"E3 low-N grid: k={k_low} (~{n_calendar_low} days), arm=contiguous, method=emos_pooled: {pooled_metrics_low}")

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
            f"E3 low-N grid: k={k_low} (~{n_calendar_low} days), arm=contiguous, method=emos_local "
            f"(coverage {n_stations_covered_low}/{n_unique_test_stations} test stations): {local_metrics_low}"
        )

        # --- fix-round-1 finding 2: random arm, 5 seeds, emos_pooled only (same
        # scope restriction the main sweep already applies to its random arm --
        # emos_local's per-station fits are out of scope for the random arm there
        # too). sample_random is reused VERBATIM, driven by the same
        # n_days_for_exact_k(k_low) synthetic n_days used for the contiguous arm
        # above, so both arms draw exactly k_low cases. ---
        seed_metrics_low = []
        for seed in sweep_seeds:
            train_rand_low = sample_random(full_train, synthetic_n_days, seed)
            actual_k_rand = len(train_rand_low[["year_idx", "time_idx"]].drop_duplicates())
            assert actual_k_rand == k_low, f"n_days_for_exact_k round-trip failed for the random arm: wanted k={k_low}, got {actual_k_rand}"
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
            f"E3 low-N grid: k={k_low} (~{n_calendar_low} days), arm=random_mean (5 seeds), method=emos_pooled: "
            f"crps={low_n_random_mean['crps']:.4f} (std={low_n_random_mean['crps_std']:.4f}), "
            f"coverage_80pct={low_n_random_mean['coverage_80pct']:.4f} (std={low_n_random_mean['coverage_80pct_std']:.4f})"
        )

    low_n_df = pd.DataFrame(low_n_rows)
    write_result(
        low_n_df,
        name="phase3_low_n_grid",
        model_version="phase3-sweep-v6",
        config={
            "low_n_k_grid": LOW_N_K_GRID,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "days_per_case_measured": DAYS_PER_CASE,
        },
    )

    # Breakpoints over the EXTENDED axis: union of the low-N grid's k=1..7 with the
    # main sweep's existing contiguous-arm k axis (k=9,26,105,314,full) -- "the
    # current k=9 number may not survive as the true breakpoint once k=1..7 are
    # actually measured" (Task 6 E3 brief).
    low_n_breakpoint_rows = []
    for metric in ["crps", "coverage_80pct"]:
        reference_value = tsfm3_row[metric]
        for emos_variant in ["emos_pooled", "emos_local"]:
            # sampling_arm == "contiguous" only (fix-round-1 finding 2 added
            # "random"/"random_mean" rows to low_n_df for emos_pooled -- the
            # breakpoint itself stays defined on the contiguous arm exactly as
            # before, matching the main sweep's own breakpoint convention).
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
                print(f"E3 extended-axis breakpoint: metric={metric}, variant={emos_variant}: no crossing — {reason}")
            else:
                print(f"E3 extended-axis breakpoint: metric={metric}, variant={emos_variant}: {bp:.2f} cases (~{bp_calendar} calendar days) — {reason}")

    low_n_breakpoints_df = pd.DataFrame(low_n_breakpoint_rows)
    write_result(
        low_n_breakpoints_df,
        name="phase3_low_n_grid_breakpoints",
        model_version="phase3-sweep-v6",
        config={"low_n_k_grid": LOW_N_K_GRID, "quantile_levels": quantile_levels},
    )

    # --- Fix-round-1 D5: the ad hoc "figures/data_size_sweep.png" quick-look figure
    # this block used to render here (fix-round-1 finding 5's matplotlib block) has
    # been REMOVED, not just deleted from disk. The style-guide review flagged the
    # shipped PNG as stale (pre-dates docs/figure_style_guide.md, uses the old
    # non-Okabe-Ito tab:* palette, duplicates what scripts/make_figures.py's F1/F3
    # now render properly) and asked for it to be deleted -- leaving this block in
    # place would silently regenerate and re-add that exact file on every future run
    # of this script, undoing the deletion. scripts/make_figures.py's make_f1()/
    # make_f3() are the only figures this project ships for this data now.
    print(
        "Fix-round-1 D5: legacy figures/data_size_sweep.png quick-look plot removed "
        "from this script (style-guide-compliant equivalents are scripts/make_figures.py's "
        "F1/F3) -- not regenerated here."
    )


if __name__ == "__main__":
    main()
