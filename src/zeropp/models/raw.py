import numpy as np

from zeropp.models.base import Postprocessor


class RawEnsemble(Postprocessor):
    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "RawEnsemble":
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        return X["ensemble_quantiles"]


class Persistence(Postprocessor):
    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "Persistence":
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        last_observed = X["last_observed"]
        n_leads = X["n_leads"]
        n_quantiles = len(self.quantile_levels)
        return np.tile(last_observed[:, None, None], (1, n_leads, n_quantiles))
