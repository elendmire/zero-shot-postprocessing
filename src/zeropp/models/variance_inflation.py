import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from zeropp.models.base import Postprocessor


class VarianceInflationBaseline(Postprocessor):
    """Trivial single-parameter baseline for Task 6 E1: mu = ens_mean (no mean
    adjustment), sigma = multiplier * sqrt(ens_var) -- the same Gaussian parametric
    family EMOS/DRN already use in this project (mu = a + b*ens_mean, sigma =
    sqrt(exp(c) + exp(d)*ens_var)), but with exactly ONE free scalar and the mean
    left untouched, so it is a fair "1 parameter / k cases" contrast against EMOS's
    "4 parameters / k cases".

    The point of this baseline is to check whether TimesFM-3's sharpness/coverage
    can be matched by a trivial rescaling of the raw ensemble's own reported
    spread, with no fitted mean correction and (for the fixed-multiplier variant)
    no fitting step at all.

    Two ways to obtain `multiplier`, per Task 6 E1 -- both are reported, neither is
    picked as "the" baseline:
      - fit(train): the standard second-moment-matching / spread-skill-ratio
        estimator, multiplier = sqrt(mean squared error / mean ensemble variance)
        over the training rows passed in. Callers MUST pass only a training subset
        (e.g. the same k-sized contiguous subset EMOS is fit on at each N in
        scripts/03_data_size_sweep.py) -- fitting this on test data would be
        leakage, exactly as it would be for EMOS.
      - from_fixed_multiplier(value, quantile_levels): a genuinely zero-shot
        constructor -- no fit() call, no training data at all. `value` is meant to
        be a literal constant, e.g. a published climatological ensemble
        under-dispersion inflation factor (see scripts/03_data_size_sweep.py's
        LAMBDA_CLIM constant and comment for the citation and its caveats), matching
        TimesFM-3's own zero-shot status.

    Fix-round-1 finding 4: an instance created via from_fixed_multiplier is
    PERMANENTLY fixed -- calling .fit(train) on it (e.g. by accident, or via generic
    code that calls .fit() on every Postprocessor uniformly) is a no-op that returns
    self unchanged, rather than silently overwriting the fixed multiplier with a
    data-derived one. Without this guard,
    `VarianceInflationBaseline.from_fixed_multiplier(1.5).fit(train)` would quietly
    stop being the zero-shot baseline E1(b) is supposed to be.
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
        """Fix-round-1 Blocking Fix 3, variant (c): find the multiplier lambda_c such
        that the raw ensemble's spread, scaled by lambda_c, achieves `target_coverage`
        empirical coverage@[lower, upper] -- evaluated ONLY on `train_df` (the training
        reforecast archive) against its own observations. This is deliberately the same
        leakage discipline as fit(): the multiplier is derived without ever looking at
        test-set coverage. Callers apply the returned (permanently fixed, like
        from_fixed_multiplier) instance to the TEST set afterward, exactly like variants
        (a) (`fit`) and (b) (`from_fixed_multiplier`) already do.

        `target_coverage` is meant to be a real, persisted number (e.g. TimesFM-3's
        measured coverage_80pct at [0.1, 0.9]) read from a results file by the caller,
        never hardcoded here.

        Coverage as a function of lambda is monotonically increasing (a wider interval
        can only cover more, never less, for a fixed centre) -- bracketed root-finding
        via `scipy.optimize.brentq` is used rather than a custom bisection loop, but the
        monotonicity assumption is verified on THIS data before trusting the single
        root: coverage at the bracket's low end must sit below target_coverage and
        coverage at its high end above it (the classic brentq precondition), which for
        a sensible bracket ([0.1, 10.0] lambda) also serves as the sanity check the task
        asked for (coverage near 0 at lambda=0.1, near 1 at lambda=10 for any
        reasonably-dispersed archive) -- an AssertionError here means the bracket itself
        needs revisiting for this dataset, not that root-finding silently returned a
        wrong answer.
        """
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
            f"from_coverage_target: bracket {bracket} does not contain target_coverage="
            f"{target_coverage} (coverage(lambda={lam_lo})={cov_lo:.4f}, "
            f"coverage(lambda={lam_hi})={cov_hi:.4f}) -- coverage-vs-lambda monotonicity "
            "could not be verified with this bracket on this data; widen `bracket` rather "
            "than trusting brentq outside a checked range."
        )

        # NOTE (Investigation 2, docs/references/priority123_investigation_report.md,
        # verified 2026-09-07 on altay): this brentq solve converges essentially
        # exactly on `train_df` -- independently re-run for target_coverage=0.760328
        # (TimesFM-3's real coverage), it returns lambda_c=2.706424, and re-computing
        # coverage@[0.1,0.9] on `train_df` at that lambda gives 0.7603282 (matching
        # the target to ~1e-7). The persisted TEST-set coverage for this exact
        # lambda_c (results/phase3_data_size_sweep.parquet,
        # var_inflation_coverage_matched, 0.822103) is NOT a root-finding miss --
        # it is the expected consequence of calibrating lambda on the training
        # reforecast archive (11 ensemble members) and applying it, unchanged, to a
        # structurally different test forecast archive (51 members) -- exactly the
        # train/test ensemble-size mismatch scripts/03_data_size_sweep.py's module
        # docstring already documents ("ens_var computed from 11 members is a
        # noisier... estimator... than one from 51 would be"). No amount of
        # root-finding precision can close a gap caused by evaluating a
        # train-calibrated constant on a differently-distributed test set; brentq
        # already converges to the tightest defensible answer to the question this
        # baseline actually asks ("what lambda hits the target on the archive we are
        # allowed to look at").
        lam_c = brentq(lambda lam: coverage_at(lam) - target_coverage, lam_lo, lam_hi, xtol=1e-6)

        instance = cls(quantile_levels)
        instance.multiplier = float(lam_c)
        instance._fixed = True
        return instance

    def fit(self, train) -> "VarianceInflationBaseline":
        if self._fixed:
            # Fix-round-1 finding 4: from_fixed_multiplier's whole point is a
            # multiplier that is NOT derived from any training data -- fit() must
            # not be allowed to silently overwrite it.
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
            raise RuntimeError(
                "VarianceInflationBaseline.predict_quantiles called before fit() or "
                "from_fixed_multiplier()"
            )
        ens_mean = X["ens_mean"]
        sigma = self.multiplier * np.sqrt(X["ens_var"])
        quantiles = [ens_mean + sigma * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
