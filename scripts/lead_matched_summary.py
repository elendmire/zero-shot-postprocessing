"""Scores of the lead-matched zero-shot runs against the 6-hourly runs with 160-step context.

Compares the runs of run_lead_matched.py with timesfm_context_160 and chronos2_context_160 on the
instances scored by all of them and contained in the common set (timesfm_context_640).

Usage: python scripts/lead_matched_summary.py [--windows 40 160]
Output: results/lead_matched_summary.parquet (lead-matched numbers in numbers.tex).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.scores import crps_from_quantiles

KEY = ["station_id", "valid_time", "step_hours"]
BUCKETS = [("0-24", 0, 24), ("30-72", 30, 72), ("78-120", 78, 120)]


def metrics(df, levels, qcols):
    q = df[qcols].to_numpy()[:, None, :]
    obs = df["obs"].to_numpy()[:, None]
    crps = crps_from_quantiles(obs, q, levels)[:, 0]
    out = {"n": len(df), "crps": crps.mean(),
           "coverage80": float(((df["obs"] >= df[qcols[0]]) & (df["obs"] <= df[qcols[-1]])).mean()),
           "width80": float((df[qcols[-1]] - df[qcols[0]]).mean())}
    for name, lo, hi in BUCKETS:
        m = ((df.step_hours >= lo) & (df.step_hours <= hi)).to_numpy()
        out[f"crps_{name}"] = crps[m].mean() if m.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[40])
    a = ap.parse_args()
    levels = load_experiment_config().quantile_levels
    qcols = [f"q{l}" for l in levels]
    r = Path("results")
    common = pd.read_parquet(r / "timesfm_context_640.parquet", columns=KEY)
    runs = {}
    for w in a.windows:
        for m in ("tsfm3", "chronos2"):
            p = r / f"lead_matched_{m}_w{w}.parquet"
            if p.exists():
                runs[f"{m}_perlead_w{w}"] = pd.read_parquet(p)
    runs["tsfm3_ctx160_6h"] = pd.read_parquet(r / "timesfm_context_160.parquet")
    runs["chronos2_ctx160_6h"] = pd.read_parquet(r / "chronos2_context_160.parquet")

    def keys(df):
        return df[KEY].drop_duplicates().set_index(KEY).index

    def table(names):
        idx = keys(common)
        for n in names:
            idx = idx.intersection(keys(runs[n]))
        rows = []
        for n in names:
            d = runs[n].drop_duplicates(KEY).set_index(KEY).loc[idx].reset_index()
            rows.append({"run": n, "set": "+".join(names), **metrics(d, levels, qcols)})
        return rows

    base = ["tsfm3_ctx160_6h", "chronos2_ctx160_6h"]
    rows = []
    for w in a.windows:
        names = [n for n in (f"tsfm3_perlead_w{w}", f"chronos2_perlead_w{w}") if n in runs] + base
        rows += [{**x, "window": w} for x in table(names)]
    res = pd.DataFrame(rows)
    res["n_common"] = len(keys(common))
    res["coverage_of_common"] = res["n"] / res["n_common"]
    res.to_parquet(r / "lead_matched_summary.parquet", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.drop(columns=["set"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
