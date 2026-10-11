import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from zeropp.models.base import Postprocessor


def _gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


class EMOS(Postprocessor):
    """Ensemble model output statistics (Gneiting et al. 2005): Gaussian with mu = a + b*ens_mean and
    sigma^2 = exp(c) + exp(d)*ens_var, fit by minimizing the mean CRPS."""

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self._params = None

    def fit(self, train) -> "EMOS":
        ens_mean = train["ens_mean"].to_numpy()
        ens_var = train["ens_var"].to_numpy()
        obs = train["t2m_obs"].to_numpy()

        def negative_mean_crps(params):
            a, b, c, d = params
            mu = a + b * ens_mean
            sigma = np.sqrt(np.exp(c) + np.exp(d) * ens_var)
            return np.mean(_gaussian_crps(mu, sigma, obs))

        result = minimize(negative_mean_crps, x0=[0.0, 1.0, 0.0, 0.0], method="L-BFGS-B")
        self._params = result.x
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("EMOS.predict_quantiles called before fit()")
        a, b, c, d = self._params
        ens_mean = X["ens_mean"]
        ens_var = X["ens_var"]
        mu = a + b * ens_mean
        sigma = np.sqrt(np.exp(c) + np.exp(d) * ens_var)
        quantiles = [mu + sigma * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)


class PersistenceAugmentedEMOS(Postprocessor):
    """EMOS with the most recent observation (last_obs) as an extra predictor of the mean:
    mu = a + b*ens_mean + e*last_obs, sigma^2 = exp(c) + exp(d)*ens_var.

    train and X must contain a last_obs column built without using the observation to be predicted."""

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self._params = None

    def fit(self, train) -> "PersistenceAugmentedEMOS":
        ens_mean = train["ens_mean"].to_numpy()
        ens_var = train["ens_var"].to_numpy()
        last_obs = train["last_obs"].to_numpy()
        obs = train["t2m_obs"].to_numpy()

        def negative_mean_crps(params):
            a, b, c, d, e = params
            mu = a + b * ens_mean + e * last_obs
            sigma = np.sqrt(np.exp(c) + np.exp(d) * ens_var)
            return np.mean(_gaussian_crps(mu, sigma, obs))

        result = minimize(negative_mean_crps, x0=[0.0, 1.0, 0.0, 0.0, 0.0], method="L-BFGS-B")
        self._params = result.x
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("PersistenceAugmentedEMOS.predict_quantiles called before fit()")
        a, b, c, d, e = self._params
        ens_mean = X["ens_mean"]
        ens_var = X["ens_var"]
        last_obs = X["last_obs"]
        mu = a + b * ens_mean + e * last_obs
        sigma = np.sqrt(np.exp(c) + np.exp(d) * ens_var)
        quantiles = [mu + sigma * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
