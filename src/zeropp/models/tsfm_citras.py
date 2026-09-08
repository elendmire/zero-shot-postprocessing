from zeropp.models.base import Postprocessor


class CitrasFM(Postprocessor):
    """Frozen CITRAS-FM zero-shot postprocessor.

    BLOCKED: needs the SSH server's `citras-fm` package installed and a
    verified covariate API before this can be implemented for real.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "CitrasFM":
        return self  # zero-shot: no-op fit, but see predict_quantiles

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs SSH server citras-fm package install and "
            "verified covariate API"
        )
