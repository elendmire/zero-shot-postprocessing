"""Scores of TimesFM-3 at each context length (4 to 640 steps) on the instances common to all lengths.

Reads predictions_main.parquet (40 steps; TimesFM-3 and raw ensemble rows) and timesfm_context_<N>.parquet.
Reports CRPS, 80 % coverage, mean interval width, CRPS by lead bucket and the raw-ensemble CRPS on the common
instances, and each run's CRPS on its own instance set.

Output: results/context_length_summary.parquet and .json (instance counts used in numbers.tex).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.scores import crps_from_quantiles

RESULTS = Path("results")
CONTEXT_LENGTHS = [4, 8, 16, 40, 160, 640]
KEY = ["station_id", "valid_time", "step_hours"]
BUCKETS = [("0-24h", 0, 24), ("24-72h", 24, 72), ("72-120h", 72, 120)]


def _path(n: int) -> Path:
    return RESULTS / ("predictions_main.parquet" if n == 40 else f"timesfm_context_{n}.parquet")


def _crps(df: pd.DataFrame, qcols: list[str], levels: list[float]) -> np.ndarray:
    return crps_from_quantiles(
        df["obs"].to_numpy()[:, None], df[qcols].to_numpy()[:, None, :], levels
    )[:, 0]


def _bucket_mask(step: pd.Series, lo: float, hi: float) -> np.ndarray:
    # (lo, hi], lead 0 included in the first bucket
    s = step.to_numpy()
    return ((s > lo) | (lo == 0)) & (s <= hi)


def _stats(df, qcols, levels, with_buckets=False):
    c = _crps(df, qcols, levels)
    lo, hi = df["q0.1"].to_numpy(), df["q0.9"].to_numpy()
    out = {
        "n_instances": int(len(df)),
        "crps": float(c.mean()),
        "coverage_80": float(((df["obs"].to_numpy() >= lo) & (df["obs"].to_numpy() <= hi)).mean()),
        "mean_width_q10_q90": float((hi - lo).mean()),
    }
    if with_buckets:
        for name, a, b in BUCKETS:
            m = _bucket_mask(df["step_hours"], a, b)
            out[f"crps_{name}"] = float(c[m].mean()) if m.any() else float("nan")
            out[f"n_{name}"] = int(m.sum())
    return out


def main() -> None:
    levels = load_experiment_config().quantile_levels
    qcols = [f"q{q}" for q in levels]

    base = pd.read_parquet(_path(40))
    tsfm = {40: base[base["method"] == "tsfm3"].copy()}
    raw = base[base["method"] == "raw_ensemble"].copy()
    for n in CONTEXT_LENGTHS:
        if n != 40:
            d = pd.read_parquet(_path(n))
            tsfm[n] = d[d["method"] == "tsfm3"].copy()
    for n, d in tsfm.items():
        d["valid_time"] = pd.to_datetime(d["valid_time"])
        d["step_hours"] = d["step_hours"].astype(float)
    raw["valid_time"] = pd.to_datetime(raw["valid_time"])
    raw["step_hours"] = raw["step_hours"].astype(float)

    def keyidx(d):
        return pd.MultiIndex.from_frame(d[KEY])

    common = None
    for n, d in tsfm.items():
        idx = keyidx(d)
        common = idx if common is None else common.intersection(idx)
    common_df = common.to_frame(index=False)
    issue = common_df["valid_time"] - pd.to_timedelta(common_df["step_hours"], unit="h")
    print(f"Common intersection: {len(common)} instances, issue dates {issue.min()} .. {issue.max()}, "
          f"valid {common_df['valid_time'].min()} .. {common_df['valid_time'].max()}")

    rows = []
    for n in CONTEXT_LENGTHS:
        d = tsfm[n]
        own = d.assign(_i=list(zip(*[d[k] for k in KEY])))
        sub = d.set_index(KEY).loc[common].reset_index()
        r = {"context_length": n, "method": "tsfm3", "set": "common"}
        r.update(_stats(sub, qcols, levels, with_buckets=True))
        rows.append(r)
        o = {"context_length": n, "method": "tsfm3", "set": "own"}
        o.update(_stats(d, qcols, levels, with_buckets=True))
        o["first_issue_date"] = str((d["valid_time"] - pd.to_timedelta(d["step_hours"], unit="h")).min())
        o["first_valid_time"] = str(d["valid_time"].min())
        rows.append(o)
    rsub = raw.set_index(KEY).loc[common].reset_index()
    r = {"context_length": -1, "method": "raw_ensemble", "set": "common"}
    r.update(_stats(rsub, qcols, levels, with_buckets=True))
    rows.append(r)

    out = pd.DataFrame(rows)
    out["first_issue_date"] = out.get("first_issue_date")
    RESULTS.mkdir(exist_ok=True)
    out.to_parquet(RESULTS / "context_length_summary.parquet", index=False)
    meta = {
        "context_lengths": CONTEXT_LENGTHS,
        "quantile_levels": levels,
        "crps": "2 * mean pinball loss over all quantile levels (zeropp.eval.scores.crps_from_quantiles)",
        "coverage": "fraction of obs in [q0.1, q0.9] (nominal 0.80)",
        "lead_buckets_h": {name: [a, b] for name, a, b in BUCKETS},
        "n_common_instances": int(len(common)),
        "common_issue_date_min": str(issue.min()),
        "common_issue_date_max": str(issue.max()),
        "common_valid_time_min": str(common_df["valid_time"].min()),
        "common_valid_time_max": str(common_df["valid_time"].max()),
        "inputs": {str(n): _path(n).name for n in CONTEXT_LENGTHS},
    }
    (RESULTS / "context_length_summary.json").write_text(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.drop(columns=["first_issue_date"]).round(4).to_string(index=False))
        print(out[out.set == "own"][["context_length", "first_issue_date", "first_valid_time"]].to_string(index=False))


if __name__ == "__main__":
    main()
