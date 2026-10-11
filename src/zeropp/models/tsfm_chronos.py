import numpy as np

from zeropp.models.base import Postprocessor

DEFAULT_MODEL_ID = "amazon/chronos-2"


class Chronos2(Postprocessor):
    """Frozen Chronos-2 zero-shot postprocessor with a known-future covariate.

    Mirrors TimesFM3.predict_quantiles: X has "context" (list of 1-D target
    arrays, length C), "past_future_ens_mean" (list of 1-D arrays of length
    C + horizon: past part then future part of ONE covariate) and "horizon".
    The covariate is split into Chronos-2's `past_covariates` (first C values)
    and `future_covariates` (last `horizon` values), the API's mechanism for a
    known-future covariate (chronos-forecasting >= 2.0, verified from
    `Chronos2Pipeline.predict` source). Series are batched in a single
    pipeline call. Returns (n_series, horizon, n_quantiles), sorted along the
    quantile axis.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 256,
    ):
        self.quantile_levels = quantile_levels
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def fit(self, train) -> "Chronos2":
        return self  # zero-shot: no-op fit

    def _load(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)
        return self._pipeline

    def predict_quantiles(self, X: dict) -> np.ndarray:
        pipe = self._load()
        horizon = int(X["horizon"])
        inputs = []
        for ctx, cov in zip(X["context"], X["past_future_ens_mean"]):
            ctx = np.asarray(ctx, dtype=np.float32)
            cov = np.asarray(cov, dtype=np.float32)
            n = len(ctx)
            assert len(cov) == n + horizon, "covariate must be context_len + horizon long"
            inputs.append({
                "target": ctx,
                "past_covariates": {"ens_mean": cov[:n]},
                "future_covariates": {"ens_mean": cov[n:]},
            })
        quantiles, _ = pipe.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
            batch_size=self.batch_size,
        )
        out = np.stack([q[0].float().cpu().numpy() for q in quantiles], axis=0)
        assert out.shape[-1] == len(self.quantile_levels)
        return np.sort(out, axis=-1)
