"""CRPS by lead time with and without a 24 h gap before the issue time (gap control).
Gap runs: results/gap24_{tsfm3,chronos2}.parquet (160-step context ending 4 steps early); no-gap runs: the
tsfm3 rows of predictions_main.parquet and chronos2_context_160.parquet. Both are restricted to the common set
(timesfm_context_640) and to the instances covered by the pair.
Usage: python scripts/gap_control_summary.py [--main-dir DIR] [--chronos-nogap PATH]
Output: results/gap_control_summary.parquet (gap-control sentences in numbers.tex).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.scores import crps_from_quantiles

KEY = ["station_id", "valid_time", "step_hours"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", default=str(Path.home() / "zsp_m/results"))
    ap.add_argument("--chronos-nogap", default=None)
    a = ap.parse_args()
    main_dir = Path(a.main_dir)
    levels = load_experiment_config().quantile_levels
    qcols = [f"q{l}" for l in levels]
    common = pd.read_parquet("results/timesfm_context_640.parquet", columns=KEY)
    nogap = {
        "tsfm3": pd.read_parquet(main_dir / "predictions_main.parquet").query("method == 'tsfm3'"),
        "chronos2": pd.read_parquet(a.chronos_nogap or main_dir / "chronos2_context_160.parquet"),
    }
    rows = []
    for model in ("tsfm3", "chronos2"):
        if not Path(f"results/gap24_{model}.parquet").exists():
            continue
        gap = pd.read_parquet(f"results/gap24_{model}.parquet")
        both = (common.merge(gap[KEY], on=KEY).merge(nogap[model][KEY].drop_duplicates(), on=KEY)
                .drop_duplicates())
        for variant, df in (("nogap", nogap[model]), ("gap24", gap)):
            d = both.merge(df.drop_duplicates(KEY), on=KEY)
            crps = crps_from_quantiles(d["obs"].to_numpy()[:, None], d[qcols].to_numpy()[:, None, :], levels)[:, 0]
            lo, hi = d[qcols[0]].to_numpy(), d[qcols[-1]].to_numpy()
            d = d.assign(crps=crps, cov80=((d["obs"] >= lo) & (d["obs"] <= hi)).astype(float), width80=hi - lo)
            for lead, g in list(d.groupby("step_hours")) + [("all", d)]:
                rows.append(dict(model=model, variant=variant, lead=lead, n=len(g), crps=g.crps.mean(),
                                 coverage80=g.cov80.mean(), width80=g.width80.mean(), n_common_pair=len(both)))
    res = pd.DataFrame(rows)
    res["lead"] = res["lead"].astype(str)
    res.to_parquet("results/gap_control_summary.parquet", index=False)
    p = res.pivot_table(index=["model", "lead"], columns="variant", values="crps")
    p["delta"] = p["gap24"] - p["nogap"]
    pd.set_option("display.width", 200)
    p = p.reset_index()
    p["_o"] = p["lead"].map(lambda x: 1e9 if x == "all" else float(x))
    print(p.sort_values(["model", "_o"]).drop(columns="_o").round(4).to_string(index=False))
    print(res.groupby(["model", "variant"]).n_common_pair.first())


if __name__ == "__main__":
    main()
