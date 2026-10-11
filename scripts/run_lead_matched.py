"""Lead-matched zero-shot runs.
For each station and lead l a daily series over issue days d is built (target: observation at d + l; covariate:
ensemble mean of the forecast issued on d for lead l). At issue day t the context is the last W known targets
(d <= t - floor(l/24) - 1) with their covariates, the horizon is up to t, and only the step at t is scored
(zeropp.data.perlead). Only instances of the common set (timesfm_context_640) are run.
Usage: python scripts/run_lead_matched.py --model {tsfm3,chronos2} [--window 40] [--issue-stride 1] [--spread]
Output: results/lead_matched_<model>_w<window>.parquet (not in the repository).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zeropp.config import load_experiment_config
from zeropp.data.perlead import build_instances, daily_arrays
from zeropp.eval.results import write_result

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = Path("cache/test_agg.parquet")
COMMON = Path("results/timesfm_context_640.parquet")


def load_agg() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    from zeropp.data.splits import load_test
    d = load_test()
    d["issue_time"] = d["valid_time"] - pd.to_timedelta(d["step_hours"], unit="h")
    g = d.groupby(["station_id", "valid_time", "step_hours", "issue_time"], as_index=False).agg(
        ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"),
        obs=("t2m_obs", "first"), n=("t2m_forecast", "size"))
    CACHE.parent.mkdir(exist_ok=True)
    g.to_parquet(CACHE)
    return g


def build_jobs(agg, common, window, stride, spread, max_stations):
    common_keys = set(zip(common.station_id, common.valid_time, common.step_hours))
    jobs = []
    stations = sorted(agg.station_id.unique())[:max_stations]
    agg = agg[agg.station_id.isin(stations)]
    for (sid, lead), g in agg.groupby(["station_id", "step_hours"]):
        g = g.assign(day=g.issue_time.dt.floor("D")).drop_duplicates("day").sort_values("day")
        cal, y, x, sd = daily_arrays(g.day.to_numpy().astype("datetime64[D]"), g.obs.to_numpy(),
                                     g.ens_mean.to_numpy(), np.sqrt(g.ens_var.to_numpy()))
        vt = (cal.astype("datetime64[ns]") + np.timedelta64(int(lead * 3600), "s"))
        keep = np.array([(sid, pd.Timestamp(v), lead) in common_keys for v in vt])
        idx = np.where(keep)[0]
        idx = idx[::stride] if stride > 1 else idx
        for inst in build_instances(y, x, lead, window, score_idx=idx, x2=sd if spread else None):
            inst.update(station=sid, lead=lead, valid_time=vt[inst["idx"]])
            jobs.append(inst)
    return jobs, len(common_keys)


def run_tsfm3(jobs, levels, bs):
    from zeropp.models.tsfm_timesfm import TimesFM3
    m = TimesFM3(quantile_levels=levels, device=DEVICE)
    m.predict_quantiles({"context": [jobs[0]["context"]], "past_future_ens_mean": [jobs[0]["covariate"]],
                         "horizon": jobs[0]["horizon"]})  # loads the model
    fc = m._model
    try:
        fc.config.per_core_batch_size = bs
    except Exception as e:
        print("could not set per_core_batch_size:", e, flush=True)
    print("per_core_batch_size =", fc.config.per_core_batch_size, flush=True)
    preds = [None] * len(jobs)
    by_h = {}
    for i, j in enumerate(jobs):
        by_h.setdefault(j["horizon"], []).append(i)
    t0, done, checked = time.time(), 0, False
    for h, ids in sorted(by_h.items()):
        for s in range(0, len(ids), 2048):
            ch = ids[s: s + 2048]
            outs = list(fc.predict_batch(contexts=[jobs[i]["context"] for i in ch], horizon=h,
                                         past_future_covariates=[jobs[i]["covariate"] for i in ch],
                                         return_quantiles=True))
            for i, o in zip(ch, outs):
                preds[i] = np.asarray(o.quantiles)[h - 1]  # last step only
            if not checked:  # compare batched with single-series prediction
                for i in ch[:3]:
                    ref = m.predict_quantiles({"context": [jobs[i]["context"]],
                                               "past_future_ens_mean": [jobs[i]["covariate"]],
                                               "horizon": h})[0, h - 1]
                    print("batch-vs-single max abs diff:", float(np.abs(ref - preds[i]).max()), flush=True)
                checked = True
            done += len(ch)
            print(f"  {done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return np.array(preds)


def run_chronos2(jobs, levels):
    from zeropp.models.tsfm_chronos import Chronos2
    m = Chronos2(quantile_levels=levels, device=DEVICE)
    preds = [None] * len(jobs)
    by_h = {}
    for i, j in enumerate(jobs):
        by_h.setdefault(j["horizon"], []).append(i)
    t0, done = time.time(), 0
    for h, ids in sorted(by_h.items()):
        for s in range(0, len(ids), 512):
            ch = ids[s: s + 512]
            out = m.predict_quantiles({"context": [jobs[i]["context"] for i in ch],
                                       "past_future_ens_mean": [jobs[i]["covariate"] for i in ch],
                                       "horizon": h})
            for k, i in enumerate(ch):
                preds[i] = out[k][h - 1]
            done += len(ch)
            print(f"  {done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tsfm3", "chronos2"])
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--issue-stride", type=int, default=1)
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    levels = load_experiment_config().quantile_levels
    agg = load_agg()
    common = pd.read_parquet(COMMON, columns=["station_id", "valid_time", "step_hours"])
    t0 = time.time()
    jobs, n_common = build_jobs(agg, common, a.window, a.issue_stride, a.spread, a.max_stations)
    print(f"{len(jobs)} series (common set {n_common}); {DEVICE}", flush=True)
    if a.model == "tsfm3":
        preds = run_tsfm3(jobs, levels, a.batch_size)
    else:
        assert not a.spread, "spread covariate only for tsfm3"
        preds = run_chronos2(jobs, levels)
    preds = np.sort(preds, axis=-1)
    qcols = [f"q{l}" for l in levels]
    res = pd.DataFrame(preds, columns=qcols)
    res.insert(0, "obs", [j["obs"] for j in jobs])
    res.insert(0, "step_hours", [j["lead"] for j in jobs])
    res.insert(0, "valid_time", [j["valid_time"] for j in jobs])
    res.insert(0, "station_id", [j["station"] for j in jobs])
    res.insert(0, "method", f"{a.model}_perlead")
    name = a.out or f"lead_matched_{a.model}_w{a.window}"
    cfg = {"window": a.window, "issue_stride": a.issue_stride, "spread": a.spread, "device": DEVICE,
           "n_rows": len(res), "n_common": n_common,
           "coverage_of_common_set": len(res) / n_common, "runtime_s": time.time() - t0}
    write_result(res, name=name, model_version=a.model, config=cfg, out_dir="results")
    print(f"Saved {len(res)} rows ({len(res)/n_common:.1%} of common set) -> results/{name}.parquet", flush=True)


if __name__ == "__main__":
    main()
