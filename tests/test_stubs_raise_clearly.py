import pytest

from zeropp.data import download as download_stub
from zeropp.models.ar_emos import AREMOS
from zeropp.models.base import Postprocessor
from zeropp.models.lead_time_continuous import LeadTimeContinuous
from zeropp.models.mos_rf import MOSRandomForest
from zeropp.models.qrf import QRF
from zeropp.models.ts_emos import TimeSeriesEMOS
from zeropp.models.tsfm_chronos import Chronos2
from zeropp.models.tsfm_citras import CitrasFM
from zeropp.models.tsfm_moirai import Moirai2

# Trained baselines: fit() itself raises NotImplementedError("blocked: ...").
# NOTE: EMOS was removed from this list once it became a real implementation
# (Task 4 of the Phase 2 plan); DRN was removed for the same reason (Task 5 of
# the Phase 3 plan) -- it has its own real behavioral tests in
# tests/test_drn.py now.
TRAINED_STUB_CLASSES = [QRF, MOSRandomForest, AREMOS, TimeSeriesEMOS, LeadTimeContinuous]

# Zero-shot TSFMs: fit() is a genuine no-op, predict_quantiles() raises.
# NOTE: TimesFM3 was removed from this list once it became a real
# implementation (Task 5 of the Phase 2 plan) -- it has its own real
# behavioral tests in tests/test_tsfm_timesfm.py now.
ZERO_SHOT_STUB_CLASSES = [Chronos2, Moirai2, CitrasFM]

ALL_STUB_CLASSES = TRAINED_STUB_CLASSES + ZERO_SHOT_STUB_CLASSES


def test_download_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="blocked"):
        download_stub.download_euppbench(target_dir="/tmp/whatever")


@pytest.mark.parametrize("model_cls", ALL_STUB_CLASSES)
def test_stub_model_is_a_postprocessor(model_cls):
    assert issubclass(model_cls, Postprocessor)


@pytest.mark.parametrize("model_cls", TRAINED_STUB_CLASSES)
def test_trained_stub_fit_raises_blocked(model_cls):
    model = model_cls(quantile_levels=[0.1, 0.5, 0.9])
    with pytest.raises(NotImplementedError, match="blocked"):
        model.fit(train=None)


@pytest.mark.parametrize("model_cls", ZERO_SHOT_STUB_CLASSES)
def test_zero_shot_stub_fit_is_noop_and_predict_raises_blocked(model_cls):
    model = model_cls(quantile_levels=[0.1, 0.5, 0.9])
    fitted = model.fit(train=None)
    assert fitted is model
    with pytest.raises(NotImplementedError, match="blocked"):
        model.predict_quantiles(X=None)
