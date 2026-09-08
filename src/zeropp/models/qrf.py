from zeropp.models.base import Postprocessor


class QRF(Postprocessor):
    """Quantile Regression Forest baseline (Taillardat 2016).

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — a random-forest fit only means
    something against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "QRF":
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(QRF forest fit needs real ensemble/obs pairs)"
        )

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(QRF forest fit needs real ensemble/obs pairs)"
        )
