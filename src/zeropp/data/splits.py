import pandas as pd

from zeropp.data.build import build_test_long_table, build_train_ensemble_stats


def load_train(data_dir: str = "data/raw") -> pd.DataFrame:
    return build_train_ensemble_stats(
        f"{data_dir}/germany_ensemble_reforecasts_t2m.nc",
        f"{data_dir}/germany_reforecasts_observations_t2m.nc",
    )


def load_test(data_dir: str = "data/raw") -> pd.DataFrame:
    return build_test_long_table(
        f"{data_dir}/germany_ensemble_forecasts_t2m.nc",
        f"{data_dir}/germany_forecasts_observations_t2m.nc",
    )
