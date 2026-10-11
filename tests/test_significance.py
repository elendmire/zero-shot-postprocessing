import numpy as np
import pytest

from zeropp.eval.significance import block_bootstrap_skill_score_ci, station_blocked_paired_test


@pytest.fixture
def synthetic_panel():
    # 10 stations, 200 instances each. Method B's loss is systematically 0.5 lower
    # than method A's, with station-correlated noise (each station has its own offset).
    rng = np.random.default_rng(0)
    n_stations = 10
    n_per_station = 200
    block_ids = np.repeat(np.arange(n_stations), n_per_station)
    station_offset = np.repeat(rng.normal(0, 0.3, n_stations), n_per_station)
    loss_a = 2.0 + station_offset + rng.normal(0, 0.2, n_stations * n_per_station)
    loss_b = loss_a - 0.5 + rng.normal(0, 0.05, n_stations * n_per_station)  # B is genuinely better
    return loss_a, loss_b, block_ids


def test_block_bootstrap_skill_score_ci_point_estimate_matches_true_effect(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    point, lo, hi = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=500, seed=1)
    true_skill = 1 - loss_b.mean() / loss_a.mean()
    assert point == pytest.approx(true_skill, abs=1e-9)  # point estimate is exact, not resampled
    assert lo < point < hi


def test_block_bootstrap_skill_score_ci_bounds_are_ordered(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    point, lo, hi = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=500, seed=1)
    assert lo <= point <= hi


def test_block_bootstrap_skill_score_ci_is_reproducible_with_same_seed(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    result1 = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=200, seed=42)
    result2 = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=200, seed=42)
    assert result1 == result2


def test_station_blocked_paired_test_detects_real_effect(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    result = station_blocked_paired_test(loss_a, loss_b, block_ids)
    assert result["n_blocks"] == 10
    assert result["block_mean_diff"] == pytest.approx(0.5, abs=0.1)  # A - B ~= 0.5
    assert result["t_pvalue"] < 0.01  # a real, station-consistent 0.5 effect should be significant
    assert result["wilcoxon_pvalue"] < 0.05


def test_station_blocked_paired_test_null_case_not_significant():
    rng = np.random.default_rng(0)  # verified: this seed gives t_pvalue≈0.375, comfortably non-significant
    n_stations = 20
    n_per_station = 100
    block_ids = np.repeat(np.arange(n_stations), n_per_station)
    loss_a = rng.normal(2.0, 0.3, n_stations * n_per_station)
    loss_b = rng.normal(2.0, 0.3, n_stations * n_per_station)  # no real difference
    result = station_blocked_paired_test(loss_a, loss_b, block_ids)
    assert result["t_pvalue"] > 0.05


def test_station_blocked_paired_test_aggregates_to_one_row_per_block(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    result = station_blocked_paired_test(loss_a, loss_b, block_ids)
    # the diff should be far more precisely estimated than raw per-instance noise would suggest,
    # because it's a mean of 10 already-averaged block means, not raw std/sqrt(2000)
    assert result["n_blocks"] == len(np.unique(block_ids))
