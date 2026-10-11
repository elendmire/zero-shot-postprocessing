import numpy as np
import pytest

from zeropp.models.base import Postprocessor


class _DummyPostprocessor(Postprocessor):
    def fit(self, train):
        return self

    def predict_quantiles(self, X):
        n_samples, n_leads, n_quantiles = len(X), 2, 9
        return np.zeros((n_samples, n_leads, n_quantiles))


def test_postprocessor_is_abstract():
    with pytest.raises(TypeError):
        Postprocessor()


def test_concrete_subclass_fit_returns_self():
    model = _DummyPostprocessor()
    fitted = model.fit(train=[1, 2, 3])
    assert fitted is model


def test_concrete_subclass_predict_shape():
    model = _DummyPostprocessor().fit(train=None)
    preds = model.predict_quantiles(X=[0, 1, 2, 3])
    assert preds.shape == (4, 2, 9)


def test_missing_predict_quantiles_raises_typeerror():
    class _Incomplete(Postprocessor):
        def fit(self, train):
            return self

    with pytest.raises(TypeError):
        _Incomplete()
