"""Real TimesFM-3 zero-shot comparison against the raw ensemble and EMOS, on
the real Germany t2m test set, at full scale on GPU.

Design history (see task-6-report.md for full detail):

1. Original design grouped by valid_time alone and collapsed all lead times
   and all ensemble members into a single scalar covariate per valid_time --
   a real bug, fixed by grouping by (station_id, issue_time) instead, where
   issue_time = valid_time - step_hours is the true NWP issue time (derived
   here since load_test()'s long table keeps only the correct valid_time, not
   the raw issue time). For each such group we build ONE context and ONE
   future covariate array -- the ensemble-mean t2m_forecast for EACH distinct
   step_hours (lead time) in that group, averaged over members, in step_hours
   order (up to 21 lead times for EUPPBench) -- and call TimesFM3 ONCE per
   group with horizon = number of lead times present. This produces one real
   quantile forecast per (station, valid_time, step_hours) instance -- the
   same unit of analysis scripts/01_run_baselines.py uses.

2. Final-review C1 finding (fixed here): the PAST portion of the
   past-future covariate was, in the previous version of this script, the
   station's own raw past observations (`context`) -- i.e. the same values
   as the model's `context` (target) input -- while the FUTURE portion was
   the ensemble mean. A past-future covariate channel must be ONE coherent
   variable known over past+future; splicing "target" (past) with "ensemble
   mean" (future) lets TimesFM-3 learn a trivial "covariate=truth" identity
   from the context and then apply it to a future covariate that is actually
   just the ensemble mean, which plausibly explains an artificially
   overconfident, mean-pinned forecast (and thus a spuriously bad CRPS).

   Fix: the past portion of the covariate is now ALSO the ensemble mean --
   specifically, for each past valid_time, the ensemble-mean t2m_forecast at
   the SHORTEST available step_hours for that valid_time (the freshest NWP
   guidance available for it; on the real Germany t2m test set the shortest
   available step_hours is 0.0 for every station, essentially an
   analysis/nowcast-like value -- confirmed via
   `test_df["step_hours"].min() == 0.0`). This makes the whole
   past-future-covariate channel one coherent "ensemble mean" variable
   throughout, while the model's separate `context` input remains the
   station's real past observations (the target series), unchanged. Groups
   without full ensemble-mean coverage for all CONTEXT_LENGTH past
   valid_times are skipped (same insufficient-history discipline used
   elsewhere in this project).

For every (station, valid_time, step_hours) instance TimesFM-3 is evaluated
on, this script ALSO computes the raw-ensemble CRPS/MAE/twCRPS/coverage/
reliability and the EMOS versions of the same, on that EXACT SAME instance,
so all printed numbers are a genuine apples-to-apples comparison computed in
one place. Per-instance raw predictions and observations for all three
methods are saved to results/phase2_comparison_raw.parquet BEFORE summary
metrics are computed, so re-analysis (bootstrap CIs, per-lead breakdowns,
etc.) does not require re-running the ~1.5 hour GPU job.

`past_future_ens_spread` is deliberately NOT passed to
`TimesFM3.predict_quantiles` -- per zeropp.models.tsfm_timesfm.TimesFM3's own
docstring/task-5 finding, the real API only exposes one past-future covariate
slot (filled here by ensemble mean), and passing that key only triggers a
no-op UserWarning.

DEVICE: prefers GPU ("cuda") when available -- a single predict() call with
horizon=21 was benchmarked on an altay A100 node at ~0.09-0.29s (vs ~1.8s/call
on CPU for a single-lead-time call), making the full, non-subsampled test set
tractable. Falls back to "cpu" automatically if no GPU is visible (per
CLAUDE.md: "always keep the CPU path working"), so this script does not
hard-fail in a future CPU-only environment -- it will just be much slower.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test, load_train
from zeropp.eval.calibration import empirical_coverage, pit_values, reliability_index
from zeropp.eval.scores import crps_from_quantiles, mae_from_quantiles, twcrps_from_quantiles
from zeropp.models.emos import EMOS
from zeropp.models.tsfm_timesfm import TimesFM3

CONTEXT_LENGTH = 40  # number of past observations/covariate points fed as context, per station
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = Path("results/phase2_comparison_raw.parquet")


def _freshest_ens_mean_lookup(station_df: pd.DataFrame) -> pd.Series:
    """Per-valid_time ensemble-mean t2m_forecast at the SHORTEST available
    step_hours for that valid_time (the freshest NWP guidance available) --
    used for the past portion of the past-future covariate so it is the same
    "ensemble mean" variable throughout, not raw observations."""
    min_step = station_df.groupby("valid_time")["step_hours"].transform("min")
    freshest_rows = station_df[station_df["step_hours"] == min_step]
    return freshest_rows.groupby("valid_time")["t2m_forecast"].mean()


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels

    train_df = load_train()
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")

    emos = EMOS(quantile_levels=quantile_levels).fit(train_df)
    model = TimesFM3(quantile_levels=quantile_levels, device=DEVICE)
    print(f"Running TimesFM-3 on device={DEVICE!r}")

    tsfm_preds, raw_preds, emos_preds, obs_values = [], [], [], []
    meta_station, meta_valid_time, meta_step_hours = [], [], []
    n_groups = 0
    n_skipped_short_context = 0
    n_skipped_missing_covariate = 0

    for station_id, station_df in test_df.groupby("station_id"):
        station_df = station_df.sort_values("valid_time")

        # Real, chronologically sorted, deduplicated-by-valid_time observation
        # series for this station -- the model's `context` (target) input.
        obs_lookup = (
            station_df.drop_duplicates("valid_time")[["valid_time", "t2m_obs"]]
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        obs_times = obs_lookup["valid_time"].to_numpy()
        obs_vals = obs_lookup["t2m_obs"].to_numpy()

        # Coherent "ensemble mean" series for the past-future covariate's
        # past portion, aligned to the same valid_time index as obs_lookup.
        freshest_ens_mean = _freshest_ens_mean_lookup(station_df)
        ens_vals = freshest_ens_mean.reindex(obs_lookup["valid_time"]).to_numpy()

        for issue_time, group in station_df.groupby("issue_time"):
            n_groups += 1
            idx = np.searchsorted(obs_times, np.datetime64(issue_time), side="left")
            if idx < CONTEXT_LENGTH:
                n_skipped_short_context += 1
                continue

            past_ens_covariate = ens_vals[idx - CONTEXT_LENGTH: idx]
            if np.isnan(past_ens_covariate).any():
                n_skipped_missing_covariate += 1
                continue
            context = obs_vals[idx - CONTEXT_LENGTH: idx]

            by_lead = (
                group.groupby("step_hours")
                .agg(
                    ens_mean=("t2m_forecast", "mean"),
                    ens_var=("t2m_forecast", "var"),
                    obs=("t2m_obs", "first"),
                    valid_time=("valid_time", "first"),
                )
                .sort_index()
            )
            future_ens_means = by_lead["ens_mean"].to_numpy()
            horizon = len(future_ens_means)
            if horizon == 0:
                continue

            # Coherent covariate: ensemble mean throughout past AND future.
            covariate = np.concatenate([past_ens_covariate, future_ens_means])

            tsfm_pred = model.predict_quantiles({
                "context": [context],
                "past_future_ens_mean": [covariate],
                "horizon": horizon,
            })  # shape (1, horizon, n_quantiles)

            emos_pred = emos.predict_quantiles({
                "ens_mean": by_lead["ens_mean"].to_numpy().reshape(-1, 1),
                "ens_var": by_lead["ens_var"].to_numpy().reshape(-1, 1),
            })  # shape (horizon, 1, n_quantiles)

            for i, step_hours in enumerate(by_lead.index):
                tsfm_preds.append(tsfm_pred[0, i])
                emos_preds.append(emos_pred[i, 0])
                obs_values.append(by_lead["obs"].iloc[i])
                meta_station.append(station_id)
                meta_valid_time.append(by_lead["valid_time"].iloc[i])
                meta_step_hours.append(step_hours)

                member_vals = group.loc[group["step_hours"] == step_hours, "t2m_forecast"].to_numpy()
                raw_preds.append(np.quantile(member_vals, quantile_levels))

    print(
        f"Processed {n_groups} (station, issue_time) groups "
        f"({n_skipped_short_context} skipped for insufficient prior context, "
        f"{n_skipped_missing_covariate} skipped for missing past ensemble-mean coverage)"
    )

    n = len(obs_values)
    n_q = len(quantile_levels)
    tsfm_flat = np.array(tsfm_preds).reshape(n, n_q)
    raw_flat = np.array(raw_preds).reshape(n, n_q)
    emos_flat = np.array(emos_preds).reshape(n, n_q)
    obs_flat = np.array(obs_values).reshape(n)

    # --- Persist raw per-instance results BEFORE computing summary metrics ---
    quantile_col_names = [f"q{level}" for level in quantile_levels]

    def _as_df(method: str, preds_flat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(preds_flat, columns=quantile_col_names)
        df.insert(0, "obs", obs_flat)
        df.insert(0, "step_hours", meta_step_hours)
        df.insert(0, "valid_time", meta_valid_time)
        df.insert(0, "station_id", meta_station)
        df.insert(0, "method", method)
        return df

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.concat(
        [
            _as_df("raw_ensemble", raw_flat),
            _as_df("emos", emos_flat),
            _as_df("tsfm3", tsfm_flat),
        ],
        ignore_index=True,
    )
    results_df.to_parquet(RESULTS_PATH, index=False)
    print(f"Saved {len(results_df)} raw per-instance rows to {RESULTS_PATH}")

    # --- Summary metrics (CLAUDE.md: never decide from a single metric) ---
    tsfm_preds_3d = tsfm_flat.reshape(n, 1, n_q)
    raw_preds_3d = raw_flat.reshape(n, 1, n_q)
    emos_preds_3d = emos_flat.reshape(n, 1, n_q)
    obs_2d = obs_flat.reshape(n, 1)

    def summarize(name: str, preds_3d: np.ndarray) -> dict:
        crps = crps_from_quantiles(obs_2d, preds_3d, quantile_levels).mean()
        mae = mae_from_quantiles(obs_2d, preds_3d, quantile_levels).mean()
        twcrps = twcrps_from_quantiles(obs_2d, preds_3d, quantile_levels).mean()
        coverage = empirical_coverage(obs_2d, preds_3d, quantile_levels, lower=0.1, upper=0.9)
        pit = pit_values(obs_2d, preds_3d, quantile_levels)
        reliability = reliability_index(pit)
        print(
            f"{name:<14} CRPS={crps:.4f}  MAE={mae:.4f}  twCRPS={twcrps:.4f}  "
            f"coverage@90%={coverage:.4f}  reliability_index={reliability:.4f}"
        )
        return {"crps": crps, "mae": mae, "twcrps": twcrps, "coverage": coverage, "reliability": reliability}

    print(f"\nEvaluated on {n} real (station, valid_time, step_hours) forecast instances")
    print("Nominal target for coverage@90% (the [0.1, 0.9] band) is ~0.80.\n")
    raw_metrics = summarize("Raw ensemble", raw_preds_3d)
    emos_metrics = summarize("EMOS", emos_preds_3d)
    tsfm_metrics = summarize("TimesFM-3", tsfm_preds_3d)

    print(f"\nEMOS improvement over raw ensemble (CRPS, matched instances):      "
          f"{(1 - emos_metrics['crps'] / raw_metrics['crps']) * 100:.1f}%")
    print(f"TimesFM-3 improvement over raw ensemble (CRPS, matched instances): "
          f"{(1 - tsfm_metrics['crps'] / raw_metrics['crps']) * 100:.1f}%")
    print(f"TimesFM-3 improvement over EMOS (CRPS, matched instances):         "
          f"{(1 - tsfm_metrics['crps'] / emos_metrics['crps']) * 100:.1f}%")


if __name__ == "__main__":
    main()
