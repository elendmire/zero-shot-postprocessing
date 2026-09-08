from abc import ABC, abstractmethod

import numpy as np


class Postprocessor(ABC):
    """Common contract for every baseline and TSFM model in ZeroPP.

    predict_quantiles must return an array of shape
    (n_samples, n_leads, n_quantiles) at the quantile levels defined in
    configs/experiment.yaml — never a model-specific shape.

    Note on `X` (the input to fit/predict_quantiles): its exact shape/type
    is deliberately UNSPECIFIED in Phase 1. Each concrete model's
    placeholder `X: dict` schema (e.g. Climatology's
    {"n_samples", "n_leads"}, RawEnsemble's {"ensemble_quantiles"},
    Persistence's {"last_observed", "n_leads"}) is temporary scaffolding
    built only for that model's own unit tests, and these schemas are NOT
    meant to be unified yet. Phase 2 must define one shared `X`
    representation (e.g. a real feature frame or a `TypedDict`) that every
    model accepts identically, before more than a couple of real baselines
    are wired into a shared driver script.
    """

    @abstractmethod
    def fit(self, train) -> "Postprocessor":
        raise NotImplementedError

    @abstractmethod
    def predict_quantiles(self, X) -> np.ndarray:
        raise NotImplementedError
