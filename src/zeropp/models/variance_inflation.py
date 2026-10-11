import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from zeropp.models.base import Postprocessor


class VarianceInflationBaseline(Postprocessor):
    """Gaussian forecast with the ensemble mean and the ensemble spread scaled by one multiplier:
    mu = ens_mean, sigma = multiplier * sqrt(ens_var).

    The multiplier comes from fit() (sqrt of the mean squared error over the mean ensemble variance of the
    training rows), from from_fixed_multiplier() (no data used) or from from_coverage_target(). Instances
    created by the last two ignore later calls to fit().
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self.multiplier = None
        self._fixed = False

    @classmethod
    def from_fixed_multiplier(cls, multiplier: float, quantile_levels: list[float]) -> "VarianceInflationBaseline":
        instance = cls(quantile_levels)
        instance.multiplier = float(multiplier)
        instance._fixed = True
        return instance

    @classmethod
    def from_coverage_target(
        cls,
        target_coverage: float,
        train_df,
        quantile_levels: list[float],
        lower: float = 0.1,
        upper: float = 0.9,
        bracket: tuple[float, float] = (0.1, 10.0),
    ) -> "VarianceInflationBaseline":
        """Multiplier that gives the target coverage of [lower, upper] on train_df.

        Coverage increases with the multiplier, so the root is found with brentq inside `bracket`, which
        must enclose the target (checked). Only train_df is used; the multiplier is then fixed."""
        obs = np.asarray(train_df["t2m_obs"], dtype=float)
        ens_mean = np.asarray(train_df["ens_mean"], dtype=float)
        ens_var = np.asarray(train_df["ens_var"], dtype=float)
        sqrt_var = np.sqrt(ens_var)
        z_lo = norm.ppf(lower)
        z_hi = norm.ppf(upper)

        def coverage_at(lam: float) -> float:
            sigma = lam * sqrt_var
            lo_bound = ens_mean + sigma * z_lo
            hi_bound = ens_mean + sigma * z_hi
            return float(np.mean((obs >= lo_bound) & (obs <= hi_bound)))

        lam_lo, lam_hi = bracket
        cov_lo, cov_hi = coverage_at(lam_lo), coverage_at(lam_hi)
        assert cov_lo < target_coverage < cov_hi, (
            f"bracket {bracket} does not enclose the target coverage {target_coverage} "
            f"(coverage {cov_lo:.4f} at {lam_lo}, {cov_hi:.4f} at {lam_hi})"
        )

        lam_c = brentq(lambda lam: coverage_at(lam) - target_coverage, lam_lo, lam_hi, xtol=1e-6)

        instance = cls(quantile_levels)
        instance.multiplier = float(lam_c)
        instance._fixed = True
        return instance

    def fit(self, train) -> "VarianceInflationBaseline":
        if self._fixed:
            return self
        obs = np.asarray(train["t2m_obs"])
        ens_mean = np.asarray(train["ens_mean"])
        ens_var = np.asarray(train["ens_var"])
        mse = np.mean((obs - ens_mean) ** 2)
        mean_var = np.mean(ens_var)
        self.multiplier = float(np.sqrt(mse / mean_var))
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self.multiplier is None:
            raise RuntimeError("predict_quantiles called before fit()")
        ens_mean = X["ens_mean"]
        sigma = self.multiplier * np.sqrt(X["ens_var"])
        quantiles = [ens_mean + sigma * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
