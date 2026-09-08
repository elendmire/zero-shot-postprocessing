from zeropp.models.base import Postprocessor


class AREMOS(Postprocessor):
    """Autoregressive EMOS baseline (AR-EMOS).

    This is the autoregressive extension of EMOS: it conditions the
    predictive distribution's parameters on lagged observations/errors in
    addition to the raw ensemble, rather than treating each lead time
    independently.

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — the CRPS-minimization fit only
    means something against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "AREMOS":
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(AR-EMOS autoregressive fit needs real ensemble/obs pairs)"
        )

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(AR-EMOS autoregressive fit needs real ensemble/obs pairs)"
        )
