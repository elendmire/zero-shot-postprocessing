"""TimesFM-3 zero-shot forecasts on the test set, with raw ensemble and EMOS on the same instances.
Per station and issue time, TimesFM-3 gets the last CONTEXT_LENGTH observations as context and the ensemble
mean as covariate (past: ensemble mean at the shortest lead; future: ensemble mean per lead) and forecasts all
leads at once. Options: context length, no covariate, withheld recent observations (gap control).
Output: results/predictions_main.parquet (per-instance quantiles; not in the repository), the input of most
other scripts.
"""
import argparse
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


def _freshest_ens_mean_lookup(station_df: pd.DataFrame) -> pd.Series:
    """Ensemble mean at the shortest available lead of each valid time (past part of the covariate)."""
    min_step = station_df.groupby("valid_time")["step_hours"].transform("min")
    freshest_rows = station_df[station_df["step_hours"] == min_step]
    return freshest_rows.groupby("valid_time")["t2m_forecast"].mean()


def main(context_length: int = CONTEXT_LENGTH, output: str = "predictions_main", tsfm_only: bool = False,
         no_covariate: bool = False, context_gap_steps: int = 0) -> None:
    results_path = Path("results") / f"{output}.parquet"
    quantile_levels = load_experiment_config().quantile_levels

    train_df = load_train()
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")

    emos = None if tsfm_only else EMOS(quantile_levels=quantile_levels).fit(train_df)
    model = TimesFM3(quantile_levels=quantile_levels, device=DEVICE)
    print(f"Running TimesFM-3 on device={DEVICE!r}, context_length={context_length}, tsfm_only={tsfm_only}, no_covariate={no_covariate}, context_gap_steps={context_gap_steps}")

    tsfm_preds, raw_preds, emos_preds, obs_values = [], [], [], []
    meta_station, meta_valid_time, meta_step_hours = [], [], []
    n_groups = 0
    n_skipped_short_context = 0
    n_skipped_missing_covariate = 0

    for station_id, station_df in test_df.groupby("station_id"):
        station_df = station_df.sort_values("valid_time")

        # observed series of the station (model context)
        obs_lookup = (
            station_df.drop_duplicates("valid_time")[["valid_time", "t2m_obs"]]
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        obs_times = obs_lookup["valid_time"].to_numpy()
        obs_vals = obs_lookup["t2m_obs"].to_numpy()

        # past part of the ensemble-mean covariate
        freshest_ens_mean = _freshest_ens_mean_lookup(station_df)
        ens_vals = freshest_ens_mean.reindex(obs_lookup["valid_time"]).to_numpy()

        for issue_time, group in station_df.groupby("issue_time"):
            n_groups += 1
            idx = np.searchsorted(obs_times, np.datetime64(issue_time), side="left")
            gap = context_gap_steps
            if idx < context_length + gap:
                n_skipped_short_context += 1
                continue

            # with a gap the context ends `gap` steps before the issue time; the covariate is kept in full
            past_ens_covariate = ens_vals[idx - gap - context_length: idx]
            if np.isnan(past_ens_covariate).any():
                n_skipped_missing_covariate += 1
                continue
            context = obs_vals[idx - gap - context_length: idx - gap]

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
            lead_pos = np.arange(horizon)
            if gap > 0:
                # lead l maps to horizon step l/6 + gap; needs the complete 6-hourly lead grid
                if not np.array_equal(by_lead.index.to_numpy(), 6.0 * np.arange(horizon)):
                    n_skipped_missing_covariate += 1
                    continue
                lead_pos = lead_pos + gap
                horizon = horizon + gap

            # ensemble mean over past and future
            covariate = np.concatenate([past_ens_covariate, future_ens_means])

            tsfm_input = {"context": [context], "horizon": horizon}
            if not no_covariate:
                tsfm_input["past_future_ens_mean"] = [covariate]
            tsfm_pred = model.predict_quantiles(tsfm_input)  # shape (1, horizon, n_quantiles)

            if not tsfm_only:
                emos_pred = emos.predict_quantiles({
                    "ens_mean": by_lead["ens_mean"].to_numpy().reshape(-1, 1),
                    "ens_var": by_lead["ens_var"].to_numpy().reshape(-1, 1),
                })  # shape (horizon, 1, n_quantiles)

            for i, step_hours in enumerate(by_lead.index):
                tsfm_preds.append(tsfm_pred[0, lead_pos[i]])
                if not tsfm_only:
                    emos_preds.append(emos_pred[i, 0])
                obs_values.append(by_lead["obs"].iloc[i])
                meta_station.append(station_id)
                meta_valid_time.append(by_lead["valid_time"].iloc[i])
                meta_step_hours.append(step_hours)

                if not tsfm_only:
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
    raw_flat = None if tsfm_only else np.array(raw_preds).reshape(n, n_q)
    emos_flat = None if tsfm_only else np.array(emos_preds).reshape(n, n_q)
    obs_flat = np.array(obs_values).reshape(n)

    # per-instance predictions
    quantile_col_names = [f"q{level}" for level in quantile_levels]

    def _as_df(method: str, preds_flat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(preds_flat, columns=quantile_col_names)
        df.insert(0, "obs", obs_flat)
        df.insert(0, "step_hours", meta_step_hours)
        df.insert(0, "valid_time", meta_valid_time)
        df.insert(0, "station_id", meta_station)
        df.insert(0, "method", method)
        return df

    results_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_as_df("tsfm3", tsfm_flat)] if tsfm_only else [
        _as_df("raw_ensemble", raw_flat),
        _as_df("emos", emos_flat),
        _as_df("tsfm3", tsfm_flat),
    ]
    results_df = pd.concat(frames, ignore_index=True)
    results_df.to_parquet(results_path, index=False)
    print(f"Saved {len(results_df)} raw per-instance rows to {results_path}")

    # summary scores
    tsfm_preds_3d = tsfm_flat.reshape(n, 1, n_q)
    if tsfm_only:
        crps = crps_from_quantiles(obs_flat.reshape(n, 1), tsfm_preds_3d, quantile_levels).mean()
        print(f"TimesFM-3 (context_length={context_length}) n={n} CRPS={crps:.4f}")
        return
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
    print("Nominal coverage of the [0.1, 0.9] band is 0.80.\n")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--output", default="predictions_main")
    parser.add_argument("--tsfm-only", action="store_true")
    parser.add_argument("--context-gap-steps", type=int, default=0,
                        help="withhold the last N six-hourly observations before issue time (context ends N steps earlier; "
                             "horizon extended by N, lead l -> step l/6+N); only the original leads are scored")
    parser.add_argument("--no-covariate", action="store_true",
                        help="run TimesFM-3 on the target context only (no past_future_covariates)")
    args = parser.parse_args()
    main(context_length=args.context_length, output=args.output, tsfm_only=args.tsfm_only, no_covariate=args.no_covariate,
         context_gap_steps=args.context_gap_steps)
