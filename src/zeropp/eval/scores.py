import numpy as np


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> np.ndarray:
    diff = y_true - q_pred
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


def _quantile_index(
    quantile_levels: list[float], levels: tuple[float, ...], param_name: str = "level"
) -> list[int]:
    indices = []
    for level in levels:
        if level not in quantile_levels:
            raise ValueError(
                f"{param_name} entry {level} is not in quantile_levels {quantile_levels}"
            )
        indices.append(quantile_levels.index(level))
    return indices


def _check_quantile_shape(quantile_preds: np.ndarray, quantile_levels: list[float]) -> None:
    if quantile_preds.shape[-1] != len(quantile_levels):
        raise ValueError(
            f"quantile_preds last-axis length {quantile_preds.shape[-1]} does not "
            f"match len(quantile_levels)={len(quantile_levels)}"
        )


def _mean_pinball_over_levels(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float], indices: list[int]
) -> np.ndarray:
    losses = [
        pinball_loss(y_true, quantile_preds[..., i], tau=quantile_levels[i]) for i in indices
    ]
    return np.mean(np.stack(losses, axis=-1), axis=-1)


def crps_from_quantiles(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> np.ndarray:
    _check_quantile_shape(quantile_preds, quantile_levels)
    indices = list(range(len(quantile_levels)))
    return 2.0 * _mean_pinball_over_levels(y_true, quantile_preds, quantile_levels, indices)


def mae_from_quantiles(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> np.ndarray:
    _check_quantile_shape(quantile_preds, quantile_levels)
    median_index = _quantile_index(quantile_levels, (0.5,), param_name="median (0.5)")[0]
    return np.abs(y_true - quantile_preds[..., median_index])


def twcrps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: list[float],
    tail_levels: tuple[float, ...] = (0.8, 0.9),
) -> np.ndarray:
    _check_quantile_shape(quantile_preds, quantile_levels)
    indices = _quantile_index(quantile_levels, tail_levels, param_name="tail_levels")
    return 2.0 * _mean_pinball_over_levels(y_true, quantile_preds, quantile_levels, indices)
