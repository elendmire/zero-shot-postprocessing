"""Chronos-2 zero-shot forecasts on the test set.
Same instances as run_timesfm.py: per station and issue time the last CONTEXT_LENGTH observations are the
context and the ensemble mean is the covariate (past: ensemble mean at the shortest lead; future: ensemble
mean per lead). Series are batched per pipeline call.
Usage: python scripts/run_chronos2.py [--context-length 40] [--context-gap-steps 0] [--max-stations N] [--out PATH]
Output: results/chronos2_predictions.parquet and .json (the parquet is not in the repository).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.eval.results import write_result
from zeropp.models.tsfm_chronos import DEFAULT_MODEL_ID, Chronos2

CONTEXT_LENGTH = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 512  # series per pipeline call


def _freshest_ens_mean_lookup(station_df: pd.DataFrame) -> pd.Series:
    min_step = station_df.groupby("valid_time")["step_hours"].transform("min")
    freshest_rows = station_df[station_df["step_hours"] == min_step]
    return freshest_rows.groupby("valid_time")["t2m_forecast"].mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--out", default="results/chronos2_predictions")
    ap.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    ap.add_argument("--context-gap-steps", type=int, default=0,
                    help="withhold the last N six-hourly observations before issue time (context ends N steps "
                         "earlier; horizon extended by N, lead l -> step l/6+N); only the original leads are scored")
    args = ap.parse_args()
    context_length = args.context_length
    gap = args.context_gap_steps

    quantile_levels = load_experiment_config().quantile_levels
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")
    model = Chronos2(quantile_levels=quantile_levels, device=DEVICE)
    print(f"Running Chronos-2 on device={DEVICE!r}", flush=True)

    # one job per (station, issue time), built as in run_timesfm.py
    jobs = []
    n_groups = n_short = n_missing = 0
    station_ids = sorted(test_df["station_id"].unique())
    if args.max_stations:
        station_ids = station_ids[: args.max_stations]
    for station_id, station_df in test_df[test_df["station_id"].isin(station_ids)].groupby("station_id"):
        station_df = station_df.sort_values("valid_time")
        obs_lookup = (
            station_df.drop_duplicates("valid_time")[["valid_time", "t2m_obs"]]
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        obs_times = obs_lookup["valid_time"].to_numpy()
        obs_vals = obs_lookup["t2m_obs"].to_numpy()
        ens_vals = _freshest_ens_mean_lookup(station_df).reindex(obs_lookup["valid_time"]).to_numpy()

        for issue_time, group in station_df.groupby("issue_time"):
            n_groups += 1
            idx = np.searchsorted(obs_times, np.datetime64(issue_time), side="left")
            if idx < context_length + gap:
                n_short += 1
                continue
            # with a gap the context ends `gap` steps early; the covariate is kept in full
            past_cov = ens_vals[idx - gap - context_length: idx]
            if np.isnan(past_cov).any():
                n_missing += 1
                continue
            context = obs_vals[idx - gap - context_length: idx - gap]
            by_lead = (
                group.groupby("step_hours")
                .agg(ens_mean=("t2m_forecast", "mean"), obs=("t2m_obs", "first"),
                     valid_time=("valid_time", "first"))
                .sort_index()
            )
            fut = by_lead["ens_mean"].to_numpy()
            if len(fut) == 0:
                continue
            lead_pos = np.arange(len(fut))
            if gap > 0:
                if not np.array_equal(by_lead.index.to_numpy(), 6.0 * np.arange(len(fut))):
                    n_missing += 1
                    continue
                lead_pos = lead_pos + gap
            jobs.append(dict(
                context=context, covariate=np.concatenate([past_cov, fut]), horizon=len(fut) + gap,
                lead_pos=lead_pos,
                station=station_id, valid_times=by_lead["valid_time"].to_numpy(),
                steps=by_lead.index.to_numpy(), obs=by_lead["obs"].to_numpy(),
            ))
    print(f"{n_groups} groups; {n_short} short context, {n_missing} missing covariate; "
          f"{len(jobs)} series to forecast", flush=True)

    # batched inference, grouped by horizon
    t0 = time.time()
    preds = [None] * len(jobs)
    by_h = {}
    for i, j in enumerate(jobs):
        by_h.setdefault(j["horizon"], []).append(i)
    done = 0
    for h, ids in by_h.items():
        for s in range(0, len(ids), CHUNK):
            chunk = ids[s: s + CHUNK]
            out = model.predict_quantiles({
                "context": [jobs[i]["context"] for i in chunk],
                "past_future_ens_mean": [jobs[i]["covariate"] for i in chunk],
                "horizon": h,
            })
            for k, i in enumerate(chunk):
                preds[i] = out[k]
            done += len(chunk)
            print(f"  {done}/{len(jobs)} series, {time.time() - t0:.0f}s", flush=True)

    # assemble predictions
    qcols = [f"q{level}" for level in quantile_levels]
    frames = []
    for j, p in zip(jobs, preds):
        df = pd.DataFrame(p[j["lead_pos"]], columns=qcols)
        df.insert(0, "obs", j["obs"])
        df.insert(0, "step_hours", j["steps"])
        df.insert(0, "valid_time", j["valid_times"])
        df.insert(0, "station_id", j["station"])
        df.insert(0, "method", "chronos2")
        frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    assert (np.diff(res[qcols].to_numpy(), axis=1) >= 0).all(), "quantiles not sorted"
    runtime = time.time() - t0

    out_path = Path(args.out)
    config = {"model_id": DEFAULT_MODEL_ID, "context_length": context_length,
              "quantile_levels": list(quantile_levels), "device": DEVICE,
              "n_series": len(jobs), "n_rows": len(res), "runtime_s": runtime,
              "max_stations": args.max_stations, "context_gap_steps": gap}
    try:
        import chronos
        version = f"{DEFAULT_MODEL_ID} (chronos-forecasting {chronos.__version__})"
    except Exception:
        version = DEFAULT_MODEL_ID
    write_result(res, name=out_path.name, model_version=version, config=config,
                 out_dir=str(out_path.parent))
    print(f"Saved {len(res)} rows to {out_path}.parquet in {runtime:.0f}s", flush=True)


if __name__ == "__main__":
    main()
