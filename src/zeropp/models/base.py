from abc import ABC, abstractmethod

import numpy as np


class Postprocessor(ABC):
    """Interface of all postprocessing models.

    predict_quantiles returns an array of shape (n_samples, n_leads, n_quantiles) at the quantile levels
    of configs/experiment.yaml. The input X of fit/predict_quantiles is model specific (a dict).
    """

    @abstractmethod
    def fit(self, train) -> "Postprocessor":
        raise NotImplementedError

    @abstractmethod
    def predict_quantiles(self, X) -> np.ndarray:
        raise NotImplementedError
