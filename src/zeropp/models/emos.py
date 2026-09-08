import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from zeropp.models.base import Postprocessor


def _gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


class EMOS(Postprocessor):
    """Ensemble Model Output Statistics: a global Gaussian CRPS-minimization
    fit, mu = a + b*ens_mean, sigma^2 = exp(c) + exp(d)*ens_var (Gneiting et al. 2005)."""

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
    """EMOS extended with exactly one extra predictor -- the most recent
    genuinely available observation ("last_obs") -- added linearly to the
    mean term only: mu = a + b*ens_mean + e*last_obs, sigma^2 = exp(c) +
    exp(d)*ens_var (unchanged from plain EMOS). Built for this project's
    priority-1 investigation (docs/references/priority123_investigation_report.md):
    TimesFM-3 receives ~40 past observations plus NWP covariates at every lead,
    while plain EMOS receives only ens_mean/ens_var -- at very short leads
    (0-24h) the most recent observation is close to a perfect predictor
    (persistence), so part of TimesFM-3's short-lead advantage could be pure
    information asymmetry rather than an architectural advantage. This model
    isolates that by giving EMOS the same single most-informative piece of
    observation history TimesFM-3 has (its most recent point), nothing more.

    `train`/`X` must carry a `last_obs` column/key alongside `ens_mean`/
    `ens_var`/`t2m_obs` (or just `ens_mean`/`ens_var`/`last_obs` for
    predict_quantiles). Constructing a leakage-free `last_obs` (never the
    same physical instant as the observation being predicted, never a future
    observation) is the caller's responsibility -- see
    scripts/09_persistence_augmented_emos.py for how it is built from the raw
    reforecast/forecast archives with a strict issue-time discipline.
    """

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
