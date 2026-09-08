from zeropp.models.base import Postprocessor


class LeadTimeContinuous(Postprocessor):
    """Lead-time-continuous postprocessing baseline (Wessel et al. 2024).

    Treats lead time as a continuous covariate instead of fitting separate
    per-lead models, which is intended to generalize better in the
    small-data regime. Per CLAUDE.md's data-size axis
    (N in {0, 30, 90, 365, 1095, full} days), this is the baseline
    specifically designated to represent the small-data-regime comparison
    point against the zero-shot TSFMs.

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — the fit only means something
    against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "LeadTimeContinuous":
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(lead-time-continuous / Wessel 2024 fit needs real ensemble/obs pairs)"
        )

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs EUPPBench training data via splits.py "
            "(lead-time-continuous / Wessel 2024 fit needs real ensemble/obs pairs)"
        )
