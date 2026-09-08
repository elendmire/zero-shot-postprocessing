from zeropp.models.base import Postprocessor


class Moirai2(Postprocessor):
    """Frozen Moirai-2 zero-shot postprocessor.

    BLOCKED: needs the SSH server's `moirai` package installed before this
    can be implemented for real.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "Moirai2":
        return self  # zero-shot: no-op fit, but see predict_quantiles

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs SSH server moirai package install"
        )
