from zeropp.models.base import Postprocessor


class TimeSeriesEMOS(Postprocessor):
    """Time-series EMOS baseline (Jobst et al. 2024).

    Extends EMOS by modeling the postprocessing coefficients as smooth
    functions across lead time / time using a time-series formulation,
    rather than fitting an independent EMOS per lead time.

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — the CRPS-minimization fit only
    means something against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "TimeSeriesEMOS":
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(time-series EMOS / Jobst 2024 fit needs real ensemble/obs pairs)"
        )

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(time-series EMOS / Jobst 2024 fit needs real ensemble/obs pairs)"
        )
