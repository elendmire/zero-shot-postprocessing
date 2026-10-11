"""Coverage-width curve of the rescaled raw ensemble over a grid of multipliers lambda.

For each lambda the forecast is N(ens_mean, lambda^2 ens_var) on the matched test instances; the 80 %
coverage, mean width of the 10-90 % interval and CRPS are recorded. Every (coverage, width) pair reachable
by rescaling the ensemble spread lies on this curve.

Output: results/lambda_curve.parquet (Fig. 6).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles

KEY = ["station_id", "valid_time", "step_hours"]
LAMBDAS = np.round(np.arange(0.50, 3.001, 0.05), 2)


def main() -> None:
    levels = load_experiment_config().quantile_levels
    raw = pd.read_parquet("results/predictions_main.parquet", columns=["method"] + KEY)
    keys = raw.loc[raw["method"] == "tsfm3", KEY]
    test = (load_test().groupby(KEY)
            .agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"), obs=("t2m_obs", "first"))
            .reset_index().merge(keys, on=KEY, how="inner"))
    assert len(test) == len(keys)
    mu = test["ens_mean"].to_numpy()[:, None]
    sd = np.sqrt(test["ens_var"].to_numpy())[:, None]
    y = test[["obs"]].to_numpy()
    z = norm.ppf(levels)[None, :]
    rows = []
    for lam in LAMBDAS:
        q = (mu + lam * sd * z)[:, None, :]
        inside = (y[:, 0] >= q[:, 0, 0]) & (y[:, 0] <= q[:, 0, -1])
        rows.append({"lambda": float(lam),
                     "coverage_80pct": float(inside.mean()),
                     "interval_width_k": float((q[:, 0, -1] - q[:, 0, 0]).mean()),
                     "crps": float(crps_from_quantiles(y, q, levels).mean())})
        print(rows[-1], flush=True)
    write_result(pd.DataFrame(rows), name="lambda_curve", model_version="lambda-curve",
                 config={"lambdas": LAMBDAS.tolist(), "quantile_levels": levels})


if __name__ == "__main__":
    main()
