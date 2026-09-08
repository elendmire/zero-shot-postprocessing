"""Statistical significance testing for per-instance loss comparisons on a
station x time x lead-time panel (e.g. Phase 2's 737,809 CRPS instances).

The 737,809 instances are NOT independent draws: they come from 49 stations,
overlapping issue times, and 21 correlated lead times per issue. Treating them
as i.i.d. for a plain paired t-test / textbook Diebold-Mariano test would
understate the true variance and overstate significance.

This module deliberately does NOT implement a textbook single-series
Diebold-Mariano (DM) test. Classical DM (and its HAC/Newey-West variance
correction) assumes one autocorrelated loss-differential series over time;
this data is a station x time x lead panel, not one series, so those
assumptions don't cleanly apply here. Instead:

- `block_bootstrap_skill_score_ci` resamples whole stations (blocks) with
  replacement, which preserves within-station (time/lead) correlation while
  still capturing across-station sampling uncertainty.
- `station_blocked_paired_test` aggregates the per-instance loss differential
  to one mean value per station first (49 independent-ish blocks), then runs
  a paired t-test and Wilcoxon signed-rank test on those block means.

Call this a "station-blocked paired test," not "Diebold-Mariano."
"""
import numpy as np
from scipy import stats


def block_bootstrap_skill_score_ci(
    loss_method: np.ndarray,
    loss_reference: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Block bootstrap confidence interval for the CRPS skill score
    1 - mean(loss_method) / mean(loss_reference), resampling whole stations
    (blocks) with replacement so within-station correlation is preserved.

    Returns (point_estimate, ci_low, ci_high). The point estimate is computed
    once on the full data (not resampled); only the CI bounds come from the
    bootstrap distribution.
    """
    loss_method = np.asarray(loss_method)
    loss_reference = np.asarray(loss_reference)
    block_ids = np.asarray(block_ids)
    unique_blocks = np.unique(block_ids)

    point = 1 - loss_method.mean() / loss_reference.mean()

    rng = np.random.default_rng(seed)
    boot_skills = np.empty(n_boot)
    block_index = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    for i in range(n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_index[b] for b in sampled_blocks])
        boot_skills[i] = 1 - loss_method[idx].mean() / loss_reference[idx].mean()

    alpha = 1 - ci
    lo, hi = np.quantile(boot_skills, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def station_blocked_paired_test(loss_a: np.ndarray, loss_b: np.ndarray, block_ids: np.ndarray) -> dict:
    """Station-blocked paired significance test (NOT a textbook Diebold-Mariano
    test): aggregate the per-instance loss differential (a - b) to one mean
    value per station/block first, then run a paired t-test and Wilcoxon
    signed-rank test on those block-level means.

    This avoids treating correlated station x time x lead instances as
    independent, at the cost of testing on ~n_stations block means rather than
    the full instance count.
    """
    loss_a = np.asarray(loss_a)
    loss_b = np.asarray(loss_b)
    block_ids = np.asarray(block_ids)
    unique_blocks = np.unique(block_ids)

    diff = loss_a - loss_b
    block_means = np.array([diff[block_ids == b].mean() for b in unique_blocks])

    t_stat, t_pvalue = stats.ttest_1samp(block_means, popmean=0.0)
    w_stat, w_pvalue = stats.wilcoxon(block_means)

    return {
        "n_blocks": len(unique_blocks),
        "block_mean_diff": float(block_means.mean()),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
    }
