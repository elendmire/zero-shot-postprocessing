"""Scores of every zero-shot run (TimesFM-3 and Chronos-2 at each context length) and of the raw ensemble.
Each run is scored on the matched instances of predictions_main.parquet and on the instances common to all
context lengths (those of the 640-step run): CRPS, qwCRPS, 80 % coverage, width, bias/MAE/RMSE of the median,
CRPS by lead bucket and season, rank histogram.
Output: results/reference_scores.parquet; used for Tables 4-6 and Figs. 2, 3, 6.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles, twcrps_from_quantiles
from seasonal_training import season_of_month

KEY = ["station_id", "valid_time", "step_hours"]
RUNS = {
    ("raw_ensemble", 0): ("results/predictions_main.parquet", "raw_ensemble"),
    **{("tsfm3", n): (f"results/timesfm_context_{n}.parquet", "tsfm3") for n in (4, 8, 16, 40, 160, 640)},
    **{("chronos2", n): (f"results/chronos2_context_{n}.parquet", None) for n in (40, 160, 640)},
}
BUCKETS = {"0-24h": (0, 24), "24-72h": (30, 72), "72-120h": (78, 120)}


def score(df: pd.DataFrame, levels, qcols) -> dict:
    y = df["obs"].to_numpy().reshape(-1, 1)
    q = df[qcols].to_numpy()[:, None, :]
    crps = crps_from_quantiles(y, q, levels).ravel()
    med = df[f"q{0.5}"].to_numpy() - y[:, 0]
    rank = (df[qcols].to_numpy() < y).sum(axis=1)
    out = {"n_instances": len(df), "crps": crps.mean(),
           "upper_qcrps": twcrps_from_quantiles(y, q, levels).mean(),
           "coverage_80pct": float(((y[:, 0] >= q[:, 0, 0]) & (y[:, 0] <= q[:, 0, -1])).mean()),
           "interval_width_k": float((q[:, 0, -1] - q[:, 0, 0]).mean()),
           "bias_k": med.mean(), "mae_k": np.abs(med).mean(), "rmse_k": np.sqrt((med ** 2).mean())}
    for name, (a, b) in BUCKETS.items():
        sel = (df["step_hours"] >= a).to_numpy() & (df["step_hours"] <= b).to_numpy()
        out[f"crps_{name}"] = crps[sel].mean()
    issue = pd.to_datetime(df["valid_time"]) - pd.to_timedelta(df["step_hours"], unit="h")
    seasons = issue.dt.month.map({m: season_of_month(m) for m in range(1, 13)}).to_numpy()
    for s in ("DJF", "MAM", "JJA", "SON"):
        out[f"crps_{s}"] = crps[seasons == s].mean()
    hist = np.bincount(rank, minlength=len(levels) + 1) / len(rank)
    out.update({f"rank_bin_{i}": float(h) for i, h in enumerate(hist)})
    return out


def main() -> None:
    levels = load_experiment_config().quantile_levels
    qcols = [f"q{q}" for q in levels]
    raw = pd.read_parquet("results/predictions_main.parquet")
    matched = raw.loc[raw["method"] == "tsfm3", KEY]
    common_path = Path("results/timesfm_context_640.parquet")
    common = pd.read_parquet(common_path, columns=KEY) if common_path.exists() else None
    rows = []
    for (model, ctx), (path, method) in RUNS.items():
        if not Path(path).exists():
            print(f"skip {model} ctx={ctx}: {path} missing")
            continue
        d = raw if path.endswith("predictions_main.parquet") else pd.read_parquet(path)
        if method is not None:
            d = d[d["method"] == method]
        d = d.drop_duplicates(KEY)
        for set_name, keys in [("matched", matched), ("common", common)]:
            if keys is None:
                continue
            sub = d.merge(keys, on=KEY, how="inner")
            if len(sub) < len(keys):
                print(f"{model} ctx={ctx} covers {len(sub)}/{len(keys)} of the {set_name} set; skipped")
                continue
            rows.append({"model": model, "context_steps": ctx, "context_days": ctx * 6 / 24,
                         "eval_set": set_name, **score(sub, levels, qcols)})
            print(model, ctx, set_name, round(rows[-1]["crps"], 4), flush=True)
    write_result(pd.DataFrame(rows), name="reference_scores", model_version="reference-scores",
                 config={"runs": {f"{m}_{c}": p for (m, c), (p, _) in RUNS.items()}, "buckets": BUCKETS})


if __name__ == "__main__":
    main()
