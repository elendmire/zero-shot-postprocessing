"""Scores of every TimesFM-3 and Chronos-2 context-length run on the instances common to all runs.
Context lengths are in six-hourly steps (40 steps = 10 days); CRPS is twice the mean pinball loss. Runs read
(missing files are skipped): timesfm_context_*, timesfm_nocov_context_*, chronos2_context_*, predictions_main
(TimesFM-3, 40 steps) and chronos2_predictions (Chronos-2, 40 steps).
Output: results/short_context_summary.parquet (Fig. 2, Table 6).
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
KEYS = ["station_id", "valid_time", "step_hours"]
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{q}" for q in LEVELS]

# (label, model, covariate, context steps, file, method filter)
RUNS = (
    [("timesfm3_cov_%d" % n, "timesfm3", True, n, f"timesfm_context_{n}", None) for n in (4, 8, 16)]
    + [("timesfm3_cov_32", "timesfm3", True, 32, "timesfm_context_32", None),
       ("timesfm3_cov_40", "timesfm3", True, 40, "predictions_main", "tsfm3")]
    + [("timesfm3_cov_%d" % n, "timesfm3", True, n, f"timesfm_context_{n}", None) for n in (160, 640)]
    + [("timesfm3_nocov_%d" % n, "timesfm3", False, n, f"timesfm_nocov_context_{n}", None) for n in (4, 8, 16, 32, 40)]
    + [("chronos2_%d" % n, "chronos2", True, n, f"chronos2_context_{n}", None) for n in (8, 16, 28)]
    + [("chronos2_40", "chronos2", True, 40, "chronos2_predictions", None)]
    + [("chronos2_%d" % n, "chronos2", True, n, f"chronos2_context_{n}", None) for n in (160, 640)]
)


def crps_pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    lv = np.asarray(LEVELS)
    d = y[:, None] - q
    return 2.0 * np.mean(np.maximum(lv * d, (lv - 1.0) * d), axis=1)


def main() -> None:
    loaded = {}
    for label, model, cov, n, fname, method in RUNS:
        path = R / f"{fname}.parquet"
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        df = pd.read_parquet(path)
        if method is not None:
            df = df[df["method"] == method]
        df = df.drop_duplicates(KEYS)
        loaded[label] = (model, cov, n, fname, df)
        print(f"{label:22s} {fname:34s} rows={len(df)}")

    common = None
    for _, _, _, _, df in loaded.values():
        k = pd.MultiIndex.from_frame(df[KEYS])
        common = k if common is None else common.intersection(k)
    common_df = common.to_frame(index=False)
    print(f"common instance set: {len(common_df)} rows across {len(loaded)} runs")

    rows = []
    for label, (model, cov, n, fname, df) in loaded.items():
        d = common_df.merge(df, on=KEYS, how="left").sort_values(KEYS)
        y = d["obs"].to_numpy(dtype=float)
        q = d[QCOLS].to_numpy(dtype=float)
        crps = crps_pinball(y, q)
        rows.append({
            "run": label, "model": model, "covariate": cov, "context_steps": n,
            "context_days": n / 4.0, "n_instances": len(d), "n_own_instances": len(loaded[label][4]),
            "crps": float(crps.mean()),
            "coverage_80pct": float(np.mean((y >= q[:, 0]) & (y <= q[:, -1]))),
            "mean_width_80pct": float(np.mean(q[:, -1] - q[:, 0])),
            "source": fname,
        })
    out = pd.DataFrame(rows).sort_values(["model", "covariate", "context_steps"]).reset_index(drop=True)
    out.to_parquet(R / "short_context_summary.parquet", index=False)
    print(out.drop(columns=["source"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
