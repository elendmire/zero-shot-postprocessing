"""Station-blocked tests for per-instance loss comparisons.

The instances (station x issue time x lead) are correlated, so a plain paired t-test would overstate
significance. block_bootstrap_skill_score_ci resamples whole stations; station_blocked_paired_test averages
the loss difference within each block (station or day) and tests the block means with a paired t-test and a
Wilcoxon signed-rank test.
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
    """Block bootstrap CI of the skill score 1 - mean(loss_method) / mean(loss_reference).

    Whole blocks are resampled; returns (point estimate on the full data, ci_low, ci_high)."""
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
    """Paired t-test and Wilcoxon test on the block means of the loss difference a - b."""
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
