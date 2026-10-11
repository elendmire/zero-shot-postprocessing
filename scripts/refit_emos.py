"""Refit the pooled EMOS rows of results/predictions_main.parquet on the overlap-free training data.

The EMOS rows written by run_timesfm.py were fit before reforecast cases verifying in the test period were
removed (TEST_PERIOD_START in zeropp.data.build). TimesFM-3 and the raw ensemble use no training data.
This script refits the pooled EMOS (same model and quantile levels) and rewrites the file in place.
"""
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test, load_train
from zeropp.eval.results import write_result
from zeropp.models.emos import EMOS

PATH = "results/predictions_main.parquet"
KEY = ["station_id", "valid_time", "step_hours"]


def main() -> None:
    levels = load_experiment_config().quantile_levels
    qcols = [f"q{q}" for q in levels]
    raw = pd.read_parquet(PATH)
    test = (load_test().groupby(KEY)
            .agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var")).reset_index())
    emos_rows = raw[raw["method"] == "emos"].reset_index(drop=True)
    x = emos_rows[KEY].merge(test, on=KEY, how="left")
    assert x["ens_mean"].notna().all()
    model = EMOS(quantile_levels=levels).fit(load_train())
    q = model.predict_quantiles({"ens_mean": x[["ens_mean"]].to_numpy(), "ens_var": x[["ens_var"]].to_numpy()})[:, 0, :]
    before = emos_rows[qcols].to_numpy()
    emos_rows[qcols] = q
    print(f"refit {len(emos_rows)} emos rows; params={model._params}; "
          f"mean |change| of median = {np.abs(q[:, levels.index(0.5)] - before[:, levels.index(0.5)]).mean():.4f} K")
    out = pd.concat([raw[raw["method"] != "emos"], emos_rows], ignore_index=True)
    write_result(out, name="predictions_main", model_version="emos-refit",
                 config={"quantile_levels": levels, "refit": "emos rows only, overlap-free training"})


if __name__ == "__main__":
    main()
