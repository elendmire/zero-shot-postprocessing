import numpy as np


def apply_conformal_calibration(
    quantile_preds: np.ndarray,
    calibration_residuals: np.ndarray,
    quantile_levels: list[float],
) -> np.ndarray:
    """Recalibrate a model's raw quantile predictions via split conformal prediction.

    BLOCKED: needs real TSFM quantile output (e.g. from tsfm_timesfm.py) on a
    held-out calibration split to compute nonconformity scores against — a
    conformal adjustment fit on synthetic quantiles would not transfer to
    real EUPPBench data.
    """
    raise NotImplementedError(
        "blocked: needs real TSFM quantile output to calibrate against"
    )
