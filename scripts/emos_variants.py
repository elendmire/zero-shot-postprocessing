"""EMOS variants: lead-specific, rolling and leave-one-station-out EMOS.
Scored on the matched TimesFM-3 instances (training cases verifying in the test period removed); each variant
is tested against TimesFM-3 and Chronos-2 with a HAC Diebold-Mariano test. Parts (--parts):
sweep (archive subsets of k cases, four EMOS variants), loso, rolling_all (four rolling variants), and the
earlier rolling and rolling_local_lead.
Output (results/): emos_variants, rolling_emos (rolling_all), rolling_emos_basic and rolling_emos_local_lead
(other rolling parts), lead_time_curves. Supports Figs. 2, 3, 5, 6 and Tables 4, 6, A1.
"""
import argparse
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_lead
from zeropp.eval.results import write_result
from zeropp.eval.robust_significance import robust_comparison
from zeropp.eval.scores import crps_from_quantiles
from zeropp.eval.significance import station_blocked_paired_test
from zeropp.models.emos import EMOS

import training_size as sweep

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/predictions_main.parquet"
CHRONOS_PATH = "results/chronos2_predictions.parquet"

K_GRID = [1, 2, 3, 5, 7, 17, 50, 204, 611]  # training cases
ROLLING_WINDOWS_DAYS = [2, 3, 5, 7, 10, 14, 30, 40, 160]
ROLLING_LOCAL_LEAD_WINDOWS = [7, 10, 14, 30, 40, 160]
ROLLING_ALL_WINDOWS = [2, 3, 5, 7, 10, 14, 20, 30, 40, 160]
MIN_ROWS_LEAD = 20
MIN_ROWS_LOCAL = 5
BUCKETS = {"0-24h": (0, 24), "24-72h": (30, 72), "72-120h": (78, 120)}
KEY = ["station_id", "valid_time", "step_hours"]


def fit_emos(df: pd.DataFrame, levels) -> EMOS | None:
    if len(df) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EMOS(quantile_levels=levels).fit(df[["ens_mean", "ens_var", "t2m_obs"]])


def predict(model: EMOS, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return model.predict_quantiles({"ens_mean": mean.reshape(-1, 1), "ens_var": var.reshape(-1, 1)})[:, 0, :]


def lead_specific_pooled(train, test, levels):
    out = np.full((len(test), len(levels)), np.nan)
    fallback = fit_emos(train, levels)
    for step, idx in test.groupby("step_hours").indices.items():
        sub = train[train["step_hours"] == step]
        model = fit_emos(sub, levels) if len(sub) >= MIN_ROWS_LEAD else fallback
        out[idx] = predict(model, test["ens_mean"].to_numpy()[idx], test["ens_var"].to_numpy()[idx])
    return out


def lead_specific_local(train, test, levels):
    out = np.full((len(test), len(levels)), np.nan)
    groups = train.groupby(["station_id", "step_hours"]).indices
    for (sid, step), idx in test.groupby(["station_id", "step_hours"]).indices.items():
        rows = groups.get((sid, step))
        if rows is None or len(rows) < MIN_ROWS_LOCAL:
            continue
        model = fit_emos(train.iloc[rows], levels)
        out[idx] = predict(model, test["ens_mean"].to_numpy()[idx], test["ens_var"].to_numpy()[idx])
    return out


def lead_pooled_local(train, test, levels):
    out = np.full((len(test), len(levels)), np.nan)
    groups = train.groupby("station_id").indices
    for sid, idx in test.groupby("station_id").indices.items():
        rows = groups.get(sid)
        if rows is None or len(rows) < MIN_ROWS_LOCAL:
            continue
        model = fit_emos(train.iloc[rows], levels)
        out[idx] = predict(model, test["ens_mean"].to_numpy()[idx], test["ens_var"].to_numpy()[idx])
    return out


def lead_pooled_pooled(train, test, levels):
    return predict(fit_emos(train, levels), test["ens_mean"].to_numpy(), test["ens_var"].to_numpy())


def loso(train, test, levels, lead_specific: bool):
    out = np.full((len(test), len(levels)), np.nan)
    for sid, idx in test.groupby("station_id").indices.items():
        others = train[train["station_id"] != sid]
        sub = test.iloc[idx]
        out[idx] = lead_specific_pooled(others, sub, levels) if lead_specific else lead_pooled_pooled(others, sub, levels)
    return out


_SHARED = {}  # data shared with forked worker processes


def rolling_local_lead_chunk(args):
    """rolling_local_lead_one_window for a chunk of issue times (data from _SHARED)."""
    window_days, issue_times = args
    ops, test, levels = _SHARED["ops"], _SHARED["test"], _SHARED["levels"]
    w = pd.Timedelta(days=window_days)
    ops_issue = ops["issue_time"].to_numpy()
    groups = test.groupby("issue_time").indices
    out_idx, out_q = [], []
    for t in issue_times:
        idx = groups[t]
        t = pd.Timestamp(t)
        lo, hi = np.searchsorted(ops_issue, np.datetime64(t - w)), np.searchsorted(ops_issue, np.datetime64(t))
        hist = ops.iloc[lo:hi]
        hist = hist[hist["valid_time"] < t]
        sub = test.iloc[idx]
        if len(hist) < MIN_ROWS_LEAD:
            continue
        q = lead_specific_local(hist, sub, levels)
        miss = np.isnan(q[:, 0])
        if miss.any():
            q[miss] = lead_pooled_local(hist, sub[miss], levels)
            miss = np.isnan(q[:, 0])
            if miss.any():
                q[miss] = lead_pooled_pooled(hist, sub[miss], levels)
        out_idx.append(idx)
        out_q.append(q)
    return window_days, out_idx, out_q


def rolling_local_lead_one_window(args):
    """Rolling EMOS, local per station and lead time; cells with fewer than MIN_ROWS_LOCAL rows fall back
    to the station's lead-pooled fit, then to the pooled fit."""
    window_days, ops, test, levels = args
    w = pd.Timedelta(days=window_days)
    out = np.full((len(test), len(levels)), np.nan)
    ops_issue = ops["issue_time"].to_numpy()
    for t, idx in test.groupby("issue_time").indices.items():
        t = pd.Timestamp(t)
        lo, hi = np.searchsorted(ops_issue, np.datetime64(t - w)), np.searchsorted(ops_issue, np.datetime64(t))
        hist = ops.iloc[lo:hi]
        hist = hist[hist["valid_time"] < t]
        sub = test.iloc[idx]
        if len(hist) < MIN_ROWS_LEAD:
            continue
        q = lead_specific_local(hist, sub, levels)
        miss = np.isnan(q[:, 0])
        if miss.any():
            q[miss] = lead_pooled_local(hist, sub[miss], levels)
            miss = np.isnan(q[:, 0])
            if miss.any():
                q[miss] = lead_pooled_pooled(hist, sub[miss], levels)
        out[idx] = q
    return window_days, out


MIN_ROWS_LEAD_LOCAL = 20


def local_adaptive(train, test, levels):
    """Single-rule rolling EMOS: local lead-specific EMOS for every station and lead with at least
    MIN_ROWS_LEAD_LOCAL pairs, otherwise local lead-pooled EMOS, otherwise pooled lead-pooled EMOS."""
    out = np.full((len(test), len(levels)), np.nan)
    groups = train.groupby(["station_id", "step_hours"]).indices
    for (sid, step), idx in test.groupby(["station_id", "step_hours"]).indices.items():
        rows = groups.get((sid, step))
        if rows is not None and len(rows) >= MIN_ROWS_LEAD_LOCAL:
            out[idx] = predict(fit_emos(train.iloc[rows], levels),
                               test["ens_mean"].to_numpy()[idx], test["ens_var"].to_numpy()[idx])
    miss = np.isnan(out[:, 0])
    if miss.any():
        out[miss] = lead_pooled_local(train, test[miss], levels)
        miss = np.isnan(out[:, 0])
        if miss.any():
            out[miss] = lead_pooled_pooled(train, test[miss], levels)
    return out


ROLLING_VARIANTS = ("rolling_emos_pooled_leadspec", "rolling_emos_local", "rolling_emos_local_leadspec",
                    "rolling_emos_local_adaptive")


def rolling_all_chunk(args):
    """All rolling variants for one window length and a chunk of issue times (data from _SHARED)."""
    window_days, issue_times = args
    ops, test, levels = _SHARED["ops"], _SHARED["test"], _SHARED["levels"]
    w = pd.Timedelta(days=window_days)
    ops_issue = ops["issue_time"].to_numpy()
    groups = test.groupby("issue_time").indices
    res = {v: ([], []) for v in ROLLING_VARIANTS}
    for t in issue_times:
        idx = groups[t]
        t = pd.Timestamp(t)
        lo, hi = np.searchsorted(ops_issue, np.datetime64(t - w)), np.searchsorted(ops_issue, np.datetime64(t))
        hist = ops.iloc[lo:hi]
        hist = hist[hist["valid_time"] < t]
        sub = test.iloc[idx]
        if len(hist) < MIN_ROWS_LEAD:
            continue
        loc = lead_pooled_local(hist, sub, levels)
        miss = np.isnan(loc[:, 0])
        if miss.any():
            loc[miss] = lead_pooled_pooled(hist, sub[miss], levels)
        ll = lead_specific_local(hist, sub, levels)
        miss = np.isnan(ll[:, 0])
        if miss.any():
            ll[miss] = loc[miss]
        for v, q in [("rolling_emos_pooled_leadspec", lead_specific_pooled(hist, sub, levels)),
                     ("rolling_emos_local", loc), ("rolling_emos_local_leadspec", ll),
                     ("rolling_emos_local_adaptive", local_adaptive(hist, sub, levels))]:
            res[v][0].append(idx)
            res[v][1].append(q)
    return window_days, res


def loso_one_station(args):
    sid, lead_specific = args
    train, test, levels = _SHARED["train"], _SHARED["test"], _SHARED["levels"]
    idx = test.groupby("station_id").indices[sid]
    others = train[train["station_id"] != sid]
    sub = test.iloc[idx]
    q = lead_specific_pooled(others, sub, levels) if lead_specific else lead_pooled_pooled(others, sub, levels)
    return sid, lead_specific, idx, q


def rolling_one_window(args):
    """Rolling EMOS for one window length W over all test issue times."""
    window_days, ops, test, levels = args
    w = pd.Timedelta(days=window_days)
    pooled = np.full((len(test), len(levels)), np.nan)
    local = np.full((len(test), len(levels)), np.nan)
    ops_issue = ops["issue_time"].to_numpy()
    for t, idx in test.groupby("issue_time").indices.items():
        t = pd.Timestamp(t)
        lo, hi = np.searchsorted(ops_issue, np.datetime64(t - w)), np.searchsorted(ops_issue, np.datetime64(t))
        hist = ops.iloc[lo:hi]
        hist = hist[hist["valid_time"] < t]
        sub = test.iloc[idx]
        if len(hist) < MIN_ROWS_LEAD:
            continue
        pooled[idx] = lead_specific_pooled(hist, sub, levels)
        loc = lead_pooled_local(hist, sub, levels)
        missing = np.isnan(loc[:, 0])
        if missing.any():
            loc[missing] = lead_pooled_pooled(hist, sub[missing], levels)
        local[idx] = loc
    return window_days, pooled, local


COMMON_REFS = {}  # name -> (mask over matched instances, per-instance CRPS on that mask)


def scores(y, q, levels, valid_time, step, tsfm_crps, sweep, chronos_crps=None) -> dict:
    ok = ~np.isnan(q[:, 0])
    if ok.sum() < 1000:
        return {"n_instances": int(ok.sum()), "covered_fraction": float(ok.mean()), "crps": float("nan")}, None, ok
    m = sweep.compute_metrics(y[ok].reshape(-1, 1), q[ok][:, None, :], levels)
    crps = crps_from_quantiles(y[ok].reshape(-1, 1), q[ok][:, None, :], levels).ravel()
    out = {**m, "n_instances": int(ok.sum()), "covered_fraction": float(ok.mean())}
    for name, (a, b) in BUCKETS.items():
        sel = (step[ok] >= a) & (step[ok] <= b)
        out[f"crps_{name}"] = float(crps[sel].mean())
        out[f"crps_tsfm3_{name}"] = float(tsfm_crps[ok][sel].mean())
    rob = robust_comparison(crps, tsfm_crps[ok], valid_time[ok])
    day = station_blocked_paired_test(crps, tsfm_crps[ok], pd.to_datetime(pd.Series(valid_time[ok])).dt.normalize().to_numpy())
    out.update({f"vs_tsfm3_{k}": v for k, v in rob.items()})
    if chronos_crps is not None:
        rob_c = robust_comparison(crps, chronos_crps[ok], valid_time[ok])
        out.update({f"vs_chronos2_{k}": v for k, v in rob_c.items()})
        for name, (a, b) in BUCKETS.items():
            sel = (step[ok] >= a) & (step[ok] <= b)
            out[f"crps_chronos2_{name}"] = float(chronos_crps[ok][sel].mean())
    out["vs_tsfm3_day_t_pvalue"] = day["t_pvalue"]
    # zero-shot references on the instances common to all context lengths
    full_crps = np.full(len(y), np.nan)
    full_crps[ok] = crps
    for ref_name, (mask, ref_crps) in COMMON_REFS.items():
        sel = mask & ok
        if sel.sum() < 1000:
            continue
        out[f"crps_common"] = float(full_crps[sel].mean())
        out[f"crps_{ref_name}_common"] = float(ref_crps[sel[mask]].mean())
        rob_r = robust_comparison(full_crps[sel], ref_crps[sel[mask]], valid_time[sel])
        out[f"vs_{ref_name}_dm_pvalue"] = rob_r["dm_pvalue"]
        out[f"vs_{ref_name}_mean_diff"] = rob_r["mean_diff"]
    return out, crps, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--parts", default="sweep,loso,rolling", help="comma-separated subset of sweep,loso,rolling")
    args = ap.parse_args()
    cfg = load_experiment_config()
    levels = cfg.quantile_levels
    qcols = [f"q{q}" for q in levels]

    print("loading data ...", flush=True)
    train = build_train_ensemble_stats_with_lead(REFORECAST_PATH, REFORECAST_OBS_PATH)
    long = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)
    ops = (long.groupby(KEY).agg(ens_mean=("t2m_forecast", "mean"), ens_var=("t2m_forecast", "var"),
                                 t2m_obs=("t2m_obs", "first")).reset_index())
    ops["issue_time"] = ops["valid_time"] - pd.to_timedelta(ops["step_hours"], unit="h")
    ops = ops.sort_values("issue_time").reset_index(drop=True)

    raw = pd.read_parquet(RAW_RESULTS_PATH)
    tsfm = raw[raw["method"] == "tsfm3"]
    test = ops.merge(tsfm[KEY], on=KEY, how="inner").reset_index(drop=True)
    assert len(test) == len(tsfm), "instance-set join incomplete"
    y = test["t2m_obs"].to_numpy()
    vt = test["valid_time"].to_numpy()
    step = test["step_hours"].to_numpy()
    ref = {}
    for name in ["tsfm3", "raw_ensemble"]:
        m = test[KEY].merge(raw[raw["method"] == name], on=KEY, how="left")
        ref[name] = m[qcols].to_numpy()
    tsfm_crps = crps_from_quantiles(y.reshape(-1, 1), ref["tsfm3"][:, None, :], levels).ravel()
    chronos_crps = None
    if Path(CHRONOS_PATH).exists():
        ch = test[KEY].merge(pd.read_parquet(CHRONOS_PATH), on=KEY, how="left")
        assert ch[qcols].notna().all().all(), "chronos2 rows missing for matched instances"
        ref["chronos2"] = ch[qcols].to_numpy()
        chronos_crps = crps_from_quantiles(y.reshape(-1, 1), ref["chronos2"][:, None, :], levels).ravel()

    for ref_name, path in [("raw_ensemble", RAW_RESULTS_PATH)] + [
            (f"{m}_ctx{c}", f"results/{'timesfm' if m == 'tsfm3' else 'chronos2'}_context_{c}.parquet")
            for m in ("tsfm3", "chronos2") for c in (40, 160, 640)]:
        if not Path(path).exists():
            continue
        d = pd.read_parquet(path)
        if "method" in d.columns:
            want = "raw_ensemble" if ref_name == "raw_ensemble" else ("tsfm3" if ref_name.startswith("tsfm3") else None)
            d = d[d["method"] == want] if want else d[d["method"] != "raw_ensemble"]
        m = test[KEY].merge(d[KEY + qcols].drop_duplicates(KEY), on=KEY, how="left")
        mask = m[qcols[0]].notna().to_numpy().copy()
        common_path = Path("results/timesfm_context_640.parquet")
        common = pd.read_parquet(common_path, columns=KEY) if common_path.exists() else None
        if common is not None:
            mask = mask & test[KEY].merge(common.assign(_c=1).drop_duplicates(KEY), on=KEY, how="left")["_c"].notna().to_numpy()
        qr = m.loc[mask, qcols].to_numpy()
        COMMON_REFS[ref_name] = (mask, crps_from_quantiles(y[mask].reshape(-1, 1), qr[:, None, :], levels).ravel())
        print(f"common-set reference {ref_name}: n={mask.sum()}, crps={COMMON_REFS[ref_name][1].mean():.4f}", flush=True)

    rows, lead_rows = [], []

    def add_lead_curve(method, q, ok):
        c = crps_from_quantiles(y[ok].reshape(-1, 1), q[ok][:, None, :], levels).ravel()
        for s, v in pd.Series(c).groupby(step[ok]).mean().items():
            lead_rows.append({"method": method, "step_hours": float(s), "crps": float(v)})

    for name in ref:
        add_lead_curve(name, ref[name], np.ones(len(y), dtype=bool))

    variants = {
        "emos_pooled": lead_pooled_pooled,
        "emos_local": lead_pooled_local,
        "emos_pooled_leadspec": lead_specific_pooled,
        "emos_local_leadspec": lead_specific_local,
    }
    case_cols = ["year_idx", "time_idx", "case_date"]
    n_cases = len(train[case_cols].drop_duplicates())
    parts = set(args.parts.split(","))
    for k in (K_GRID + [n_cases]) if "sweep" in parts else []:
        sub = sweep.sample_contiguous(train, sweep.n_days_for_exact_k(k)) if k < n_cases else train
        for method, fn in variants.items():
            q = fn(sub, test, levels)
            res, _, ok = scores(y, q, levels, vt, step, tsfm_crps, sweep, chronos_crps)
            rows.append({"method": method, "arm": "contiguous", "k_cases": k, **res})
            print(f"k={k} {method}: crps={res['crps']:.4f} cov={res['covered_fraction']:.3f} "
                  f"dm_p={res.get('vs_tsfm3_dm_pvalue', float('nan')):.3g}", flush=True)
            if k == n_cases and ok.any():
                add_lead_curve(method, q, ok)

    if "loso" in parts:
        _SHARED.update(train=train, test=test, levels=levels)
        tasks = [(sid, ls) for ls in (False, True) for sid in test["station_id"].unique()]
        preds = {ls: np.full((len(test), len(levels)), np.nan) for ls in (False, True)}
        with mp.get_context("fork").Pool(min(args.processes, len(tasks))) as pool:
            for sid, ls, idx, q in pool.imap_unordered(loso_one_station, tasks):
                preds[ls][idx] = q
        for ls, name in [(False, "emos_loso"), (True, "emos_loso_leadspec")]:
            res, _, ok = scores(y, preds[ls], levels, vt, step, tsfm_crps, sweep, chronos_crps)
            rows.append({"method": name, "arm": "full_other_stations", "k_cases": n_cases, **res})
            add_lead_curve(name, preds[ls], ok)
            print(f"{name}: crps={res['crps']:.4f}", flush=True)

    sweep_df = pd.DataFrame(rows)
    if rows:
        write_result(sweep_df, name="emos_variants", model_version="emos-variants",
                     config={"k_grid": K_GRID, "n_cases_full": n_cases, "min_rows_lead": MIN_ROWS_LEAD,
                             "min_rows_local": MIN_ROWS_LOCAL, "buckets": BUCKETS})

    if "rolling" in parts:
        print("rolling EMOS ...", flush=True)
        jobs = [(w, ops, test, levels) for w in ROLLING_WINDOWS_DAYS]
        roll_rows = []
        with mp.get_context("fork").Pool(min(args.processes, len(jobs))) as pool:
            for w, pooled, local in pool.imap_unordered(rolling_one_window, jobs):
                for name, q in [("rolling_emos_pooled_leadspec", pooled), ("rolling_emos_local", local)]:
                    res, _, ok = scores(y, q, levels, vt, step, tsfm_crps, sweep, chronos_crps)
                    roll_rows.append({"method": name, "window_days": w, **res})
                    if w == 10:
                        add_lead_curve(f"{name}_w10", q, ok)
                    print(f"W={w} {name}: crps={res['crps']:.4f} covered={res['covered_fraction']:.3f}", flush=True)
        write_result(pd.DataFrame(roll_rows).sort_values(["method", "window_days"]), name="rolling_emos_basic",
                     model_version="rolling-emos-basic",
                     config={"windows_days": ROLLING_WINDOWS_DAYS, "min_rows_lead": MIN_ROWS_LEAD,
                             "min_rows_local": MIN_ROWS_LOCAL, "verified_pairs_only": True})
    if "rolling_all" in parts:
        _SHARED.update(ops=ops, test=test, levels=levels)
        issue_times = list(test.groupby("issue_time").indices.keys())
        n_chunks = max(1, (args.processes * 4) // len(ROLLING_ALL_WINDOWS))
        chunks = [issue_times[i::n_chunks] for i in range(n_chunks)]
        tasks = [(w, c) for w in ROLLING_ALL_WINDOWS for c in chunks]
        preds = {(v, w): np.full((len(test), len(levels)), np.nan) for v in ROLLING_VARIANTS for w in ROLLING_ALL_WINDOWS}
        done = 0
        with mp.get_context("fork").Pool(args.processes) as pool:
            for w, res in pool.imap_unordered(rolling_all_chunk, tasks):
                for v, (idxs, qs) in res.items():
                    for idx, q in zip(idxs, qs):
                        preds[(v, w)][idx] = q
                done += 1
                if done % 50 == 0:
                    print(f"rolling_all: {done}/{len(tasks)} chunks", flush=True)
        roll_rows = []
        for (v, w), q in preds.items():
            res, _, ok = scores(y, q, levels, vt, step, tsfm_crps, sweep, chronos_crps)
            roll_rows.append({"method": v, "window_days": w, **res})
            if w in (10, 40):
                add_lead_curve(f"{v}_w{w}", q, ok)
            print(f"W={w} {v}: crps={res['crps']:.4f}", flush=True)
        write_result(pd.DataFrame(roll_rows).sort_values(["method", "window_days"]), name="rolling_emos",
                     model_version="rolling-emos",
                     config={"windows_days": ROLLING_ALL_WINDOWS, "variants": list(ROLLING_VARIANTS),
                             "min_rows_lead": MIN_ROWS_LEAD, "min_rows_local": MIN_ROWS_LOCAL,
                             "min_rows_lead_local_adaptive": MIN_ROWS_LEAD_LOCAL, "verified_pairs_only": True})

    if "rolling_local_lead" in parts:
        _SHARED.update(ops=ops, test=test, levels=levels)
        issue_times = list(test.groupby("issue_time").indices.keys())
        n_chunks = max(1, (args.processes * 4) // len(ROLLING_LOCAL_LEAD_WINDOWS))
        chunks = [issue_times[i::n_chunks] for i in range(n_chunks)]
        tasks = [(w, c) for w in ROLLING_LOCAL_LEAD_WINDOWS for c in chunks]
        preds = {w: np.full((len(test), len(levels)), np.nan) for w in ROLLING_LOCAL_LEAD_WINDOWS}
        done = 0
        with mp.get_context("fork").Pool(args.processes) as pool:
            for w, idxs, qs in pool.imap_unordered(rolling_local_lead_chunk, tasks):
                for idx, q in zip(idxs, qs):
                    preds[w][idx] = q
                done += 1
                if done % 50 == 0:
                    print(f"rolling_local_lead: {done}/{len(tasks)} chunks", flush=True)
        rows_ll = []
        for w in ROLLING_LOCAL_LEAD_WINDOWS:
            q = preds[w]
            res, _, ok = scores(y, q, levels, vt, step, tsfm_crps, sweep, chronos_crps)
            rows_ll.append({"method": "rolling_emos_local_leadspec", "window_days": w, **res})
            if w == 40:
                add_lead_curve("rolling_emos_local_leadspec_w40", q, ok)
            print(f"W={w} rolling_emos_local_leadspec: crps={res['crps']:.4f}", flush=True)
        write_result(pd.DataFrame(rows_ll).sort_values("window_days"), name="rolling_emos_local_lead",
                     model_version="rolling-emos-local-lead",
                     config={"windows_days": ROLLING_LOCAL_LEAD_WINDOWS, "min_rows_local": MIN_ROWS_LOCAL,
                             "verified_pairs_only": True})
    write_result(pd.DataFrame(lead_rows), name="lead_time_curves", model_version="lead-time-curves",
                 config={"note": "CRPS per lead time on the matched set"})


if __name__ == "__main__":
    main()
