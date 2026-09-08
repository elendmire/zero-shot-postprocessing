import numpy as np

from zeropp.models.base import Postprocessor


class Climatology(Postprocessor):
    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self._empirical_quantiles: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> "Climatology":
        self._empirical_quantiles = np.quantile(train, self.quantile_levels)
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self._empirical_quantiles is None:
            raise RuntimeError("Climatology.predict_quantiles called before fit()")
        n_samples, n_leads = X["n_samples"], X["n_leads"]
        base = self._empirical_quantiles
        return np.tile(base, (n_samples, n_leads, 1))
