import numpy as np


def average_quantile_predictions(
    predictions: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Combine several models' predict_quantiles() outputs into one ensemble
    quantile prediction (quantile averaging / vincentization).

    BLOCKED: needs multiple real model quantile outputs (e.g. EMOS and
    TimesFM3 both fit/run on real EUPPBench data) to average — averaging
    stub or synthetic outputs would not be a meaningful ensemble.
    """
    raise NotImplementedError(
        "blocked: needs multiple real model quantile outputs to average"
    )
