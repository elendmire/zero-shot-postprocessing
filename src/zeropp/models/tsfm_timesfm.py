import os

import numpy as np
import timesfm

from zeropp.models.base import Postprocessor

DEFAULT_WEIGHTS_PATH = os.environ.get(
    "ZEROPP_TIMESFM_WEIGHTS_PATH", "model_cache/timesfm-3.0-pytorch"
)


class TimesFM3(Postprocessor):
    """Frozen TimesFM-3 zero-shot postprocessor with real past-future covariate injection.

    COVARIATE SHAPE, VERIFIED FROM THE REAL UPSTREAM SOURCE (2026-09-07,
    paper review B9 -- this replaces an earlier version of this docstring
    that only checked our OWN wrapper's `inspect.signature`, not the
    installed `timesfm==3.0.1` package's real internals). The public
    `TimesFM3Forecaster.predict`/`predict_batch` methods take one
    parameter, `past_future_covariates: np.ndarray | None` -- but reading
    `timesfm3/model.py`'s own docstring directly (not the model card)
    shows this parameter's real accepted shape is `(batch,
    num_covariate_channels, context_len+horizon)`: a genuine multi-channel
    axis, confirmed further by `timesfm3/evaluator.py`, which implements
    an explicit "subsample future covariates to at most 31 slots" step --
    multi-channel covariates are a real, exercised capability of this
    package, not a theoretical one. At the single-series convenience-
    wrapper level used here, a bare 1-D covariate array is promoted via
    `np.atleast_2d` to shape `(1, time)`; passing two simultaneous
    channels therefore requires shape `(2, time)` -- channels on axis 0,
    time on the LAST axis. (An earlier version of `predict_quantiles`
    below stacked mean and spread on `axis=-1`, producing `(time, 2)` --
    the wrong axis convention; fixed here.)

    This shape-level finding is strong evidence the API supports what
    this class now attempts, but it is not the same as confirming the
    model produces sensible, genuinely-different output when given two
    channels instead of one -- that empirical question is
    `docs/pending_ssh_runs.md` item 6's smoke test (T1), not something
    reading source code can settle by itself.

    QUANTILE LEVEL VERIFICATION (final-review C2 finding): inspected the real
    installed `timesfm3` package source
    (`timesfm3/model.py`: `quantiles: list[float] | None = None` defaulting to
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`; `timesfm3_forecaster.py`:
    `median_quantile_index: int = 4`) AND the real loaded checkpoint at
    `DEFAULT_WEIGHTS_PATH` on the server (`model.config.quantiles ==
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`, `median_quantile_index ==
    4`) — confirmed the model's real quantile head outputs exactly these 9
    levels, in this ascending order, matching this project's
    `quantile_levels` config column-for-column. `predict()` is also called
    with its default `sort_quantiles=True`, which only fixes any residual
    quantile-crossing (keeps output non-decreasing along the quantile axis)
    and does not reorder columns/levels, so column i of `out.quantiles`
    always corresponds to `self.quantile_levels[i]`. The shape assertion
    below is a lightweight runtime guard for this; the level *values* are
    a fixed model-config property verified once here rather than re-checked
    on every call (the underlying package does not expose them per-call).
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
        return self  # zero-shot: no-op fit, but see predict_quantiles

    def predict_quantiles(self, X: dict) -> np.ndarray:
        ens_spreads = X.get("past_future_ens_spread")

        if self._model is None:
            self._model = timesfm.TimesFM3Forecaster.from_pretrained(self.weights_path, device=self.device)

        contexts = X["context"]
        ens_means = X["past_future_ens_mean"]
        horizon = X["horizon"]

        all_quantiles = []
        covariate_iter = zip(contexts, ens_means, ens_spreads) if ens_spreads is not None else (
            (ctx, ens_mean, None) for ctx, ens_mean in zip(contexts, ens_means)
        )
        for ctx, ens_mean, ens_spread in covariate_iter:
            if ens_spread is not None:
                # Shape (2, T): channels on axis 0, time on axis -1 -- the
                # convention verified directly from timesfm3/model.py's
                # docstring (B9, class docstring above), not guessed. Row 0
                # = ensemble mean, row 1 = ensemble spread. Whether the real
                # model produces genuinely different, sensible output for
                # this (as opposed to merely accepting the shape without
                # erroring) is docs/pending_ssh_runs.md item 6's smoke test.
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
            assert quantiles.shape[-1] == len(self.quantile_levels), (
                f"TimesFM3 returned {quantiles.shape[-1]} quantile columns but "
                f"quantile_levels has {len(self.quantile_levels)} entries — the "
                "model's quantile head configuration no longer matches this "
                "project's quantile_levels (see class docstring for the "
                "verified default: [0.1, ..., 0.9])."
            )
            all_quantiles.append(quantiles)

        return np.stack(all_quantiles, axis=0)
