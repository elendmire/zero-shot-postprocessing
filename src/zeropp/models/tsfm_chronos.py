from zeropp.models.base import Postprocessor


class Chronos2(Postprocessor):
    """Frozen Chronos-2 zero-shot postprocessor.

    BLOCKED: needs the SSH server's `chronos` package installed before this
    can be implemented for real.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "Chronos2":
        return self  # zero-shot: no-op fit, but see predict_quantiles

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs SSH server chronos package install"
        )
