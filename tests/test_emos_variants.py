import numpy as np
import pandas as pd

import emos_variants as ev

LEVELS = [0.1, 0.5, 0.9]


def _ops(n_days=12, stations=(1, 2), steps=(0.0, 6.0, 24.0), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        issue = pd.Timestamp("2017-01-01") + pd.Timedelta(days=d)
        for s in stations:
            for st in steps:
                m = 280 + rng.normal()
                rows.append({"station_id": s, "step_hours": st, "issue_time": issue,
                             "valid_time": issue + pd.Timedelta(hours=st), "ens_mean": m,
                             "ens_var": 1.0 + rng.random(), "t2m_obs": m + rng.normal()})
    return pd.DataFrame(rows).sort_values("issue_time").reset_index(drop=True)


def test_rolling_emos_never_uses_observations_at_or_after_issue_time():
    ops = _ops()
    t0 = pd.Timestamp("2017-01-10")
    test = ops[ops["issue_time"] == t0].reset_index(drop=True)
    _, pooled_a, local_a = ev.rolling_one_window((5, ops, test, LEVELS))
    perturbed = ops.copy()
    perturbed.loc[perturbed["valid_time"] >= t0, "t2m_obs"] += 50.0
    _, pooled_b, local_b = ev.rolling_one_window((5, perturbed, test, LEVELS))
    np.testing.assert_allclose(pooled_a, pooled_b)
    np.testing.assert_allclose(local_a, local_b)


def test_lead_specific_local_skips_station_lead_without_enough_rows():
    ops = _ops(n_days=3)
    test = ops.iloc[:4].reset_index(drop=True)
    out = ev.lead_specific_local(ops.iloc[:2], test, LEVELS)
    assert np.isnan(out).all()


def test_loso_excludes_target_station_from_training():
    ops = _ops(n_days=6)
    test = ops[ops["station_id"] == 1].reset_index(drop=True)
    shifted = ops.copy()
    shifted.loc[shifted["station_id"] == 1, "t2m_obs"] += 100.0
    a = ev.loso(ops, test, LEVELS, lead_specific=False)
    b = ev.loso(shifted, test, LEVELS, lead_specific=False)
    np.testing.assert_allclose(a, b)


def test_chunked_rolling_local_lead_matches_single_window():
    ops = _ops(n_days=14)
    test = ops[ops["issue_time"] >= pd.Timestamp("2017-01-09")].reset_index(drop=True)
    _, ref = ev.rolling_local_lead_one_window((5, ops, test, LEVELS))
    ev._SHARED.update(ops=ops, test=test, levels=LEVELS)
    times = list(test.groupby("issue_time").indices.keys())
    got = np.full_like(ref, np.nan)
    for chunk in (times[0::2], times[1::2]):
        _, idxs, qs = ev.rolling_local_lead_chunk((5, chunk))
        for idx, q in zip(idxs, qs):
            got[idx] = q
    np.testing.assert_allclose(got, ref)


def test_rolling_all_chunk_matches_individual_variants_and_respects_cutoff():
    ops = _ops(n_days=30)
    test = ops[ops["issue_time"] >= pd.Timestamp("2017-01-25")].reset_index(drop=True)
    ev._SHARED.update(ops=ops, test=test, levels=LEVELS)
    times = list(test.groupby("issue_time").indices.keys())
    _, res = ev.rolling_all_chunk((20, times))
    assert set(res) == set(ev.ROLLING_VARIANTS)
    _, ref_pooled, ref_local = ev.rolling_one_window((20, ops, test, LEVELS))
    got = {v: np.full_like(ref_pooled, np.nan) for v in res}
    for v, (idxs, qs) in res.items():
        for idx, q in zip(idxs, qs):
            got[v][idx] = q
    np.testing.assert_allclose(got["rolling_emos_pooled_leadspec"], ref_pooled)
    np.testing.assert_allclose(got["rolling_emos_local"], ref_local)
    assert not np.isnan(got["rolling_emos_local_adaptive"]).any()
    # future observations must not change any variant
    perturbed = ops.copy()
    perturbed.loc[perturbed["valid_time"] >= pd.Timestamp("2017-01-25"), "t2m_obs"] += 50.0
    ev._SHARED.update(ops=perturbed, test=test, levels=LEVELS)
    _, res2 = ev.rolling_all_chunk((20, times[:1]))
    for v in res2:
        np.testing.assert_allclose(res2[v][1][0], res[v][1][0])


def test_parallel_loso_worker_matches_serial():
    ops = _ops(n_days=6)
    test = ops.reset_index(drop=True)
    ev._SHARED.update(train=ops, test=test, levels=LEVELS)
    ref = ev.loso(ops, test, LEVELS, lead_specific=False)
    got = np.full_like(ref, np.nan)
    for sid in test["station_id"].unique():
        _, _, idx, q = ev.loso_one_station((sid, False))
        got[idx] = q
    np.testing.assert_allclose(got, ref)
