import os

import numpy as np
import timesfm

from zeropp.models.base import Postprocessor

DEFAULT_WEIGHTS_PATH = os.environ.get(
    "ZEROPP_TIMESFM_WEIGHTS_PATH", "model_cache/timesfm-3.0-pytorch"
)


class TimesFM3(Postprocessor):
    """Zero-shot TimesFM-3.

    X has "context" (list of 1-D target arrays), "horizon", and optionally "past_future_ens_mean" (list of
    1-D arrays of length context + horizon; omitted for a run without covariate) and
    "past_future_ens_spread" (same layout; stacked with the mean as a second channel). Returns an array of
    shape (n_series, horizon, 9) at the quantile levels 0.1, ..., 0.9 of the model's quantile head.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        device: str = "cpu",
    ):
        self.quantile_levels = quantile_levels
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def fit(self, train) -> "TimesFM3":
        return self  # zero-shot: nothing to fit

    def predict_quantiles(self, X: dict) -> np.ndarray:
        ens_spreads = X.get("past_future_ens_spread")

        if self._model is None:
            self._model = timesfm.TimesFM3Forecaster.from_pretrained(self.weights_path, device=self.device)

        contexts = X["context"]
        # Optional: when absent (e.g. scripts/02 --no-covariate) the model is
        # run on the target context alone (past_future_covariates=None).
        ens_means = X.get("past_future_ens_mean")
        if ens_means is None:
            ens_means = [None] * len(contexts)
        horizon = X["horizon"]

        all_quantiles = []
        covariate_iter = zip(contexts, ens_means, ens_spreads) if ens_spreads is not None else (
            (ctx, ens_mean, None) for ctx, ens_mean in zip(contexts, ens_means)
        )
        for ctx, ens_mean, ens_spread in covariate_iter:
            if ens_spread is not None:
                # shape (2, T): channel 0 ensemble mean, channel 1 ensemble spread
                covariates = np.stack([np.asarray(ens_mean), np.asarray(ens_spread)], axis=0)
            else:
                covariates = ens_mean
            out = self._model.predict(
                context=ctx,
                horizon=horizon,
                past_future_covariates=covariates,
                return_quantiles=True,
            )
            quantiles = np.asarray(out.quantiles)
            assert quantiles.shape[-1] == len(self.quantile_levels), "quantile levels of the model and the config differ"
            all_quantiles.append(quantiles)

        return np.stack(all_quantiles, axis=0)
