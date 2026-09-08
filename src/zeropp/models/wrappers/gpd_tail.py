import numpy as np


def apply_gpd_tail_correction(
    quantile_preds: np.ndarray,
    tail_quantile_levels: list[float],
) -> np.ndarray:
    """Replace a model's extreme (tail) quantiles with a Generalized Pareto
    Distribution fit over exceedances, to fix under-dispersion in the tails.

    BLOCKED: needs real TSFM tail quantile output to fit the GPD exceedance
    threshold and shape/scale parameters against — a fit on synthetic tail
    quantiles would not reflect the model's actual tail behavior.
    """
    raise NotImplementedError(
        "blocked: needs real TSFM tail quantile output"
    )
