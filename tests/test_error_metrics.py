import numpy as np
import pytest

import error_metrics


def test_point_errors_uses_median_and_forecast_minus_obs_sign():
    levels = [0.1, 0.5, 0.9]
    y = np.array([[0.0], [0.0]])
    preds = np.array([[[-9.0, 1.0, 9.0]], [[-9.0, -3.0, 9.0]]])
    out = error_metrics.point_errors(y, preds, levels)
    assert out["bias_k"] == pytest.approx(-1.0)
    assert out["mae_k"] == pytest.approx(2.0)
    assert out["rmse_k"] == pytest.approx(np.sqrt(5.0))
