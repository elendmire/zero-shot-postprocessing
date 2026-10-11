"""Figures 2-7 of the paper and the plotting style shared with fig01_study_area.py.
Reads results/ and writes figures/<name>.pdf and .svg (and a .png preview, not tracked).
fig02_recent_data (CRPS against days of recent data), fig03_training_size, fig04_seasonal, fig05_lead_time,
fig06_calibration (coverage-width plane), fig07_rank_histograms.
Usage: python scripts/make_figures.py [fig02 fig03 ...]
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
R = RESULTS_DIR

STYLE = {
    "font": {
        "family": "sans-serif",
        "sans_serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "panel_title_pt": 10,
        "axis_label_pt": 9,
        "tick_label_pt": 8,
        "annotation_pt": 8,
        "legend_pt": 9,
    },
    "linewidth": {"data": 1.4, "reference": 1.0, "gridline": 0.5},
    "mm_per_column": {"single": 90.0, "double": 190.0},
    "dpi_png": 300,
    "alpha": {"std_band": 0.20},
}
COLORS = {
    "raw_ensemble": "#999999",
    "var_inflation_fixed": "#E69F00",
    "var_inflation_trainfit": "#E69F00",
    "tsfm3": "#0072B2",
    "emos_pooled": "#D55E00",
    "emos_local": "#009E73",
    "drn": "#CC79A7",
    "reference": "#666666",
    "gridline": "#DDDDDD",
    "chronos2": "#56B4E9",
    "emos_local_leadspec": "#004D40",
    "emos_pooled_leadspec": "#8C3B00",
    "rolling_emos_local": "#009E73",
    "rolling_emos_pooled_leadspec": "#D55E00",
    "rolling_emos_local_leadspec": "#004D40",
    "emos_loso_leadspec": "#7F7F7F",
    "rolling_emos_local_adaptive": "#000000",
}
LABELS = {
    "raw_ensemble": "Raw ensemble",
    "tsfm3": "TimesFM-3 (zero-shot)",
    "emos_pooled": "EMOS pooled",
    "emos_local": "EMOS local",
    "drn": "DRN",
    "chronos2": "Chronos-2 (zero-shot)",
    "emos_local_leadspec": "EMOS local, lead-specific",
    "emos_pooled_leadspec": "EMOS pooled, lead-specific",
    "rolling_emos_local": "Rolling EMOS, local",
    "rolling_emos_pooled_leadspec": "Rolling EMOS, pooled lead-specific",
    "rolling_emos_local_leadspec": "Rolling EMOS, local lead-specific",
    "rolling_emos_local_adaptive": "Rolling EMOS, single rule (40 days)",
    "emos_loso_leadspec": "EMOS leave-one-station-out",
    "var_inflation_trainfit": "Var.-inflated, training-fit",
    "var_inflation_fixed": "Var.-inflated, fixed λ = 1.5",
}


def _mm_to_in(mm):
    return mm / 25.4


def _figsize_in(kind, height_mm):
    return (_mm_to_in(STYLE["mm_per_column"][kind]), _mm_to_in(height_mm))


def _apply_rcparams():
    plt.rcParams.update({
        "font.family": STYLE["font"]["family"],
        "font.sans-serif": STYLE["font"]["sans_serif"],
        "font.size": STYLE["font"]["tick_label_pt"],
        "axes.labelsize": STYLE["font"]["axis_label_pt"],
        "axes.titlesize": STYLE["font"]["panel_title_pt"],
        "xtick.labelsize": STYLE["font"]["tick_label_pt"],
        "ytick.labelsize": STYLE["font"]["tick_label_pt"],
        "legend.fontsize": STYLE["font"]["legend_pt"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _style_grid(ax):
    ax.set_facecolor("white")
    ax.grid(True, axis="both", color=COLORS["gridline"], linewidth=STYLE["linewidth"]["gridline"])
    ax.set_axisbelow(True)


def _save_figure(fig, name, source_parquets):
    """Write PDF (source files and git SHA in the metadata), SVG and PNG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {
        "Subject": "source: " + "; ".join(source_parquets),
        "Keywords": f"git_sha:{_git_sha()}",
    }
    fig.savefig(FIGURES_DIR / f"{name}.pdf", metadata=pdf_metadata)
    fig.savefig(FIGURES_DIR / f"{name}.svg")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=STYLE["dpi_png"])


def _load(name):
    return pd.read_parquet(R / f"{name}.parquet")


MAIN = 160  # main zero-shot configuration: 40-day (160 six-hourly steps) context


def _ref(model, ctx, eval_set="matched", col="crps"):
    d = _load("reference_scores")
    hit = d[(d["model"] == model) & (d["context_steps"] == ctx) & (d["eval_set"] == eval_set)]
    return float(hit[col].iloc[0]) if len(hit) else None


def _days_per_case():
    return float(json.loads((R / "seasonal_breakpoints.json").read_text())["config"]["days_per_case"])


def _panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".rstrip(), loc="left")


def _legend(fig, handles, ncol=3, y=0.0):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
               fontsize=STYLE["font"]["legend_pt"] - 1)


def add_short_context(refs, summary):
    """Append the runs of short_context_summary that are missing from the reference scores."""
    s = summary.rename(columns={"mean_width_80pct": "interval_width_k"}).copy()
    s.loc[~s["covariate"], "model"] = s.loc[~s["covariate"], "model"] + "_nocov"
    s["model"] = s["model"].replace({"timesfm3": "tsfm3", "timesfm3_nocov": "tsfm3_nocov"})
    have = set(zip(refs["model"], refs["context_steps"]))
    s = s[[(m, c) not in have for m, c in zip(s["model"], s["context_steps"])]]
    cols = ["model", "context_steps", "context_days", "crps", "coverage_80pct", "interval_width_k"]
    rows = [s[cols].assign(eval_set=e) for e in ("matched", "common")]
    return pd.concat([refs, *rows], ignore_index=True)


# ---------------------------------------------------------------- CRPSS figures
SKILL_LIM = (-0.17, 0.30)  # axis; values below SKILL_EDGE are drawn as triangles in the strip below it
SKILL_EDGE = -0.15
LABEL_PT = 8  # direct line labels at tick-label size
OFFSET = {"emos_pooled": 0.86, "emos_local": 0.95, "emos_local_leadspec": 1.05, "drn": 1.16,
          "tsfm3": 0.93, "chronos2": 1.07, "rolling": 1.0}  # fixed horizontal offsets of edge triangles
MSTYLE = {  # one colour/marker per method across all figures
    "tsfm3": dict(color=COLORS["tsfm3"], marker="D", ls="-", lw=1.6),
    "chronos2": dict(color=COLORS["chronos2"], marker="s", ls="-", lw=1.6),
    "rolling": dict(color="#000000", marker="o", ls="-", lw=2.4),
    "emos_pooled": dict(color=COLORS["emos_pooled"], marker="o", ls="-", lw=1.4),
    "emos_local": dict(color=COLORS["emos_local"], marker="^", ls="-", lw=1.4),
    "emos_local_leadspec": dict(color=COLORS["emos_local_leadspec"], marker="v", ls="-", lw=1.4),
    "drn": dict(color=COLORS["drn"], marker="X", ls="-", lw=1.4),
    "var_inflation_trainfit": dict(color=COLORS["var_inflation_trainfit"], marker="P", ls="-", lw=1.2),
}
MLABEL = {
    "tsfm3": "TimesFM-3", "chronos2": "Chronos-2", "rolling": "Rolling EMOS, single rule",
    "emos_pooled": "EMOS pooled", "emos_local": "EMOS local", "emos_local_leadspec": "EMOS local, lead-specific",
    "drn": "DRN", "var_inflation_trainfit": "Var.-inflated, training-fit",
}


def _skill(crps, ref):
    return 1.0 - np.asarray(crps, float) / ref


def _skill_axis(ax):
    ax.set_ylim(*SKILL_LIM)
    ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ax.axhline(SKILL_EDGE, color="0.6", linewidth=0.6, zorder=1)


def _in_range(s, lo=SKILL_EDGE):
    """Drop values below the skill axis so that a reference line ends at the frame instead of leaving it."""
    s = np.asarray(s, float).copy()
    s[s < lo] = np.nan
    return s


def _mhandle(key, label=None, **kw):
    st = MSTYLE[key]
    return Line2D([], [], color=st["color"], marker=st["marker"], linestyle=st["ls"], linewidth=st["lw"],
                  markersize=4, label=label or MLABEL[key], **kw)


def _skill_line(ax, x, s, key, jitter=None, lo=SKILL_EDGE):
    """Plot in-range points joined by a line; points below the axis become open
    triangles on the lower edge, not joined to the line."""
    st = MSTYLE[key]
    jitter = OFFSET.get(key, 1.0) if jitter is None else jitter
    x, s = np.asarray(x, float), np.asarray(s, float)
    inr = s >= lo
    seg_x, seg_s = [], []
    for xi, si, ok in zip(x, s, inr):
        if ok:
            seg_x.append(xi); seg_s.append(si)
        elif seg_x:
            ax.plot(seg_x, seg_s, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"], markersize=4)
            seg_x, seg_s = [], []
    if seg_x:
        ax.plot(seg_x, seg_s, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"], markersize=4)
    if (~inr).any():
        ax.plot(x[~inr] * jitter, np.full((~inr).sum(), (SKILL_LIM[0] + lo) / 2), linestyle="none", marker="v",
                markersize=4.5,
                markerfacecolor="white", markeredgecolor=st["color"], markeredgewidth=1.3, clip_on=False, zorder=5)


def _hline(ax, y, label, color, ls, x_text=0.99, ha="right", va="bottom"):
    ax.axhline(y, color=color, linestyle=ls, linewidth=1.2, zorder=1)
    ax.text(x_text, y, label, transform=ax.get_yaxis_transform(), ha=ha, va=va, fontsize=LABEL_PT, color=color)


def _crps_axis(ax, ref):
    """Right-hand axis in K for a CRPSS axis relative to ``ref``."""
    sec = ax.secondary_yaxis("right", functions=(lambda s: ref * (1 - s), lambda c: 1 - c / ref))
    sec.set_ylabel("CRPS (K)")
    return sec


BELOW_HANDLE = Line2D([], [], linestyle="none", marker="v", markersize=6, markerfacecolor="white",
                      markeredgecolor="0.3", label="Below axis: >15 % worse than raw ensemble")


def fig02_recent_data():
    refs = _load("reference_scores")
    if (R / "short_context_summary.parquet").exists():
        refs = add_short_context(refs, _load("short_context_summary"))
    var = _load("emos_variants")
    roll = _load("rolling_emos")
    raw = float(refs[(refs["model"] == "raw_ensemble") & (refs["eval_set"] == "common")]["crps"].iloc[0])
    full = var[(var["method"] == "emos_local_leadspec") & (var["arm"] == "contiguous")].sort_values("k_cases").iloc[-1]
    loso_v = float(var[var["method"] == "emos_loso_leadspec"]["crps"].iloc[0])
    zs = refs[refs["model"].isin(["tsfm3", "chronos2"]) & (refs["eval_set"] == "common") & (refs["context_steps"] > 0)]
    _apply_rcparams()
    fig, ax = plt.subplots(figsize=_figsize_in("double", 105))
    fig.subplots_adjust(left=0.09, right=0.91, top=0.97, bottom=0.27)
    _style_grid(ax)
    ax.set_xscale("log")
    ax.axvspan(0.8, 8, color="0.94", zorder=0, linewidth=0)
    ax.text(2.6, 0.17, "TimesFM-3 context shorter\nthan one input patch (8 days)", ha="center", va="center",
            fontsize=LABEL_PT, color="0.35")
    _hline(ax, _skill(full["crps"], raw), "Archive-trained local lead-specific EMOS (20 years)",
           MSTYLE["emos_local_leadspec"]["color"], "--", x_text=0.01, ha="left")
    _hline(ax, _skill(loso_v, raw), "Lead-specific EMOS on other stations (LOSO)", COLORS["emos_loso_leadspec"], "-.")
    _hline(ax, 0.0, "Raw ensemble", "0.45", ":")
    d = roll[roll["method"] == "rolling_emos_local_adaptive"].sort_values("window_days")
    _skill_line(ax, d["window_days"], _skill(d["crps"], raw), "rolling")
    for model in ("tsfm3", "chronos2"):
        d = zs[zs["model"] == model].sort_values("context_days")
        _skill_line(ax, d["context_days"], _skill(d["crps"], raw), model)
    _skill_axis(ax)
    ax.set_xlim(0.8, 200)
    ax.set_xticks([1, 2, 5, 10, 20, 40, 160])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "40", "160"])
    ax.minorticks_off()
    ax.set_xlabel("Days of recent station observations (context length or training window, log scale)")
    ax.set_ylabel("CRPSS relative to the raw ensemble")
    _crps_axis(ax, raw)
    _legend(fig, [_mhandle("rolling"), _mhandle("tsfm3", "TimesFM-3, context length"),
                  _mhandle("chronos2", "Chronos-2, context length"), BELOW_HANDLE], ncol=2)
    _save_figure(fig, "fig02_recent_data", ["results/rolling_emos.parquet", "results/reference_scores.parquet"])
    plt.close(fig)


def fig03_training_size():
    sweep = _load("training_size")
    low = _load("training_size_small").rename(columns={"k": "n_cases"})
    var = _load("emos_variants")
    lead = var[(var["arm"] == "contiguous") & (var["method"] == "emos_local_leadspec")].dropna(subset=["crps"])
    lead = lead.sort_values("k_cases")
    dpc = _days_per_case()

    def curve(method, metric, arm="contiguous"):
        d = pd.concat([sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == arm)],
                       low[(low["method"] == method) & (low["sampling_arm"] == arm)]])
        d = d.dropna(subset=[metric]).drop_duplicates("n_cases").sort_values("n_cases")
        d = d[d["n_cases"] > 0]
        std = d[f"{metric}_std"].fillna(0).to_numpy() if f"{metric}_std" in d else np.zeros(len(d))
        return d["n_cases"].to_numpy(float), d[metric].to_numpy(float), std

    def flat(method, metric):
        return float(sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "n_independent")][metric].iloc[0])

    raw = _ref("raw_ensemble", 0)
    zs = {m: {"crps": _ref(m, MAIN), "coverage_80pct": _ref(m, MAIN, col="coverage_80pct")} for m in ("tsfm3", "chronos2")}
    _apply_rcparams()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=_figsize_in("double", 175), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.11, right=0.91, top=0.965, bottom=0.25, hspace=0.18)
    for ax in (ax_a, ax_b):
        _style_grid(ax)
        ax.set_xscale("log")
    # (a) CRPSS
    _hline(ax_a, _skill(zs["tsfm3"]["crps"], raw), "TimesFM-3, 40-day context", MSTYLE["tsfm3"]["color"], "-", x_text=0.01, ha="left")
    _hline(ax_a, _skill(zs["chronos2"]["crps"], raw), "Chronos-2, 40-day context", MSTYLE["chronos2"]["color"], "-",
           x_text=0.01, ha="left", va="top")
    _hline(ax_a, _skill(flat("var_inflation_fixed", "crps"), raw), "Var.-inflated, fixed λ = 1.5", COLORS["var_inflation_fixed"], "-.")
    _hline(ax_a, 0.0, "Raw ensemble", "0.45", ":")
    x, y, s = curve("emos_pooled", "crps", "random_mean")
    lo_band = np.maximum(_skill(y + s, raw), SKILL_EDGE)
    ax_a.fill_between(x, lo_band, _skill(y - s, raw), color=COLORS["emos_pooled"], alpha=STYLE["alpha"]["std_band"], linewidth=0)
    ax_a.plot(x, _in_range(_skill(y, raw)), color=COLORS["emos_pooled"], linestyle="--", linewidth=1.2)
    for m in ("emos_pooled", "emos_local", "drn"):
        x, y, _ = curve(m, "crps")
        _skill_line(ax_a, x, _skill(y, raw), m)
    _skill_line(ax_a, lead["k_cases"], _skill(lead["crps"], raw), "emos_local_leadspec")
    _skill_axis(ax_a)
    ax_a.set_xlim(0.7, 6500)
    ax_a.set_ylabel("CRPSS relative to\nthe raw ensemble")
    _crps_axis(ax_a, raw)
    _panel(ax_a, "a")
    # (b) coverage
    ax_b.axhline(0.8, color=COLORS["reference"], linestyle="--", linewidth=STYLE["linewidth"]["reference"])
    ax_b.text(0.99, 0.8, "Nominal", transform=ax_b.get_yaxis_transform(), ha="right", va="bottom", fontsize=LABEL_PT, color="0.35")
    for m, lab in [("tsfm3", "TimesFM-3"), ("chronos2", "Chronos-2")]:
        ax_b.axhline(zs[m]["coverage_80pct"], color=MSTYLE[m]["color"], linewidth=1.2)
    ax_b.text(0.99, zs["tsfm3"]["coverage_80pct"], "TimesFM-3", transform=ax_b.get_yaxis_transform(), ha="right",
              va="bottom", fontsize=LABEL_PT, color=MSTYLE["tsfm3"]["color"])
    ax_b.text(0.99, zs["chronos2"]["coverage_80pct"], "Chronos-2", transform=ax_b.get_yaxis_transform(), ha="right",
              va="top", fontsize=LABEL_PT, color=MSTYLE["chronos2"]["color"])
    _hline(ax_b, flat("raw_ensemble", "coverage_80pct"), "Raw ensemble", "0.45", ":")
    x, y, s = curve("emos_pooled", "coverage_80pct", "random_mean")
    ax_b.fill_between(x, y - s, y + s, color=COLORS["emos_pooled"], alpha=STYLE["alpha"]["std_band"], linewidth=0)
    ax_b.plot(x, y, color=COLORS["emos_pooled"], linestyle="--", linewidth=1.2)
    for m in ["emos_pooled", "emos_local", "drn"]:
        x, y, _ = curve(m, "coverage_80pct")
        st = MSTYLE[m]
        ax_b.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"], markersize=4)
    st = MSTYLE["emos_local_leadspec"]
    ax_b.plot(lead["k_cases"], lead["coverage_80pct"], color=st["color"], marker=st["marker"], lw=st["lw"], markersize=4)
    ax_b.set_ylabel("80 % coverage")
    _panel(ax_b, "b")
    ax_b.set_xlabel("Archive training cases (log scale)")
    sec = ax_b.secondary_xaxis(-0.30, functions=(lambda k: k * dpc, lambda d: d / dpc))
    sec.set_xlabel("Equivalent span of calendar days")
    _legend(fig, [_mhandle("emos_pooled"), _mhandle("emos_local"), _mhandle("emos_local_leadspec"), _mhandle("drn"),
                  Line2D([], [], color=COLORS["emos_pooled"], linestyle="--", label="EMOS pooled, random cases (± 1 SD)"),
                  BELOW_HANDLE], ncol=3)
    _save_figure(fig, "fig03_training_size", ["results/training_size.parquet", "results/training_size_small.parquet",
                                                 "results/emos_variants.parquet"])
    plt.close(fig)


def fig04_seasonal():
    sweep = _load("seasonal_sweep")
    dpc = _days_per_case()
    _apply_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=_figsize_in("double", 150), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.95, bottom=0.25, hspace=0.24, wspace=0.08)
    names = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)", "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}
    for ax, (season, title), letter in zip(axes.flat, names.items(), "abcd"):
        cur = sweep[sweep["season"] == season]
        _style_grid(ax)
        ax.set_xscale("log")
        _panel(ax, letter, title)
        raw = float(cur[(cur["method"] == "raw_ensemble") & (cur["sampling_arm"] == "n_independent")]["crps"].iloc[0])
        ax.axhline(0, color="0.45", linestyle=":", linewidth=1.2)
        for m in ("tsfm3", "chronos2"):
            ref = cur[(cur["method"] == m) & (cur["sampling_arm"] == "n_independent")]
            if len(ref):
                ax.axhline(_skill(float(ref["crps"].iloc[0]), raw), color=MSTYLE[m]["color"], linewidth=1.4)
        rnd = cur[(cur["method"] == "emos_pooled") & (cur["sampling_arm"] == "random")].sort_values("k_cases")
        ax.fill_between(rnd["k_cases"], np.maximum(_skill(rnd["crps"] + rnd["crps_std"], raw), SKILL_EDGE),
                        _skill(rnd["crps"] - rnd["crps_std"], raw), color=COLORS["emos_pooled"],
                        alpha=STYLE["alpha"]["std_band"], linewidth=0)
        ax.plot(rnd["k_cases"], _in_range(_skill(rnd["crps"], raw)), color=COLORS["emos_pooled"], linestyle="--",
                linewidth=1.0)
        for m in ["emos_pooled", "emos_local", "emos_local_leadspec"]:
            c = cur[(cur["method"] == m) & (cur["sampling_arm"] == "contiguous")].sort_values("k_cases")
            if len(c):
                _skill_line(ax, c["k_cases"], _skill(c["crps"], raw), m)
        _skill_axis(ax)
        ax.set_xlim(0.7, 1500)
    for ax in axes[1]:
        ax.set_xlabel("Season-matched training cases (log scale)")
        sec = ax.secondary_xaxis(-0.30, functions=(lambda k: k * dpc, lambda d: d / dpc))
        sec.set_xlabel("Equivalent span of calendar days")
    for ax in axes[:, 0]:
        ax.set_ylabel("CRPSS relative to the\nseason's raw ensemble")
    handles = [_mhandle("emos_pooled"), _mhandle("emos_local")]
    if (sweep["method"] == "emos_local_leadspec").any():
        handles.append(_mhandle("emos_local_leadspec"))
    handles += [Line2D([], [], color=MSTYLE["tsfm3"]["color"], lw=1.4, label="TimesFM-3, 40-day context"),
                Line2D([], [], color=MSTYLE["chronos2"]["color"], lw=1.4, label="Chronos-2, 40-day context"),
                BELOW_HANDLE]
    _legend(fig, handles, ncol=3)
    _save_figure(fig, "fig04_seasonal", ["results/seasonal_sweep.parquet"])
    plt.close(fig)


def fig05_lead_time():
    lc = _load("lead_time_curves").drop_duplicates(["method", "step_hours"], keep="last")
    base = lc[lc["method"] == "rolling_emos_local_adaptive_w40"].set_index("step_hours")["crps"]
    _apply_rcparams()
    fig, ax = plt.subplots(figsize=_figsize_in("double", 100))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.27)
    _style_grid(ax)
    for h in range(0, 121, 24):
        ax.axvspan(h - 3, h + 3, color="0.93", zorder=0, linewidth=0)
    ax.axhline(0, color="#000000", linewidth=1.6)
    handles = []
    for m, key in [("tsfm3", "tsfm3"), ("chronos2", "chronos2"), ("emos_local_leadspec", "emos_local_leadspec")]:
        d = lc[lc["method"] == m].set_index("step_hours")["crps"].reindex(base.index)
        st = MSTYLE[key]
        ax.plot(base.index, d - base, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"], markersize=3.5)
        lab = {"tsfm3": "TimesFM-3 (40-day context)", "chronos2": "Chronos-2 (40-day context)",
               "emos_local_leadspec": "EMOS local, lead-specific (archive)"}[m]
        handles.append(_mhandle(key, lab))
    ax.set_xticks(np.arange(0, 121, 12))
    ax.set_xlim(-4, 124)
    ax.set_ylim(-0.06, 0.31)
    ax.set_xlabel("Lead time (h); shaded: 00 UTC valid times")
    ax.set_ylabel("CRPS difference to rolling EMOS (K)")
    _legend(fig, handles, ncol=3)
    _save_figure(fig, "fig05_lead_time", ["results/lead_time_curves.parquet"])
    plt.close(fig)


def fig06_calibration():
    lam = _load("lambda_curve")
    refs = _load("reference_scores")
    var = _load("emos_variants")
    sweep = _load("training_size")
    _apply_rcparams()
    fig, ax = plt.subplots(figsize=_figsize_in("double", 120))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.27)
    _style_grid(ax)
    ax.plot(lam["coverage_80pct"], lam["interval_width_k"], color=COLORS["var_inflation_fixed"], linewidth=STYLE["linewidth"]["data"])
    for lv in (1.0, 1.5, 2.0, 2.5):
        r = lam.iloc[(lam["lambda"] - lv).abs().argmin()]
        ax.annotate(f"λ = {lv:g}", (r["coverage_80pct"], r["interval_width_k"]), xytext=(4, -10), textcoords="offset points",
                    fontsize=STYLE["font"]["annotation_pt"], color=COLORS["var_inflation_fixed"])
    pts = []
    for model, key, mk in [("tsfm3", "tsfm3", "D"), ("chronos2", "chronos2", "s"), ("raw_ensemble", "raw_ensemble", "X")]:
        r = refs[(refs["model"] == model) & (refs["eval_set"] == "matched") & (refs["context_steps"] == (0 if model == "raw_ensemble" else MAIN))]
        if len(r):
            pts.append((key, mk, float(r["coverage_80pct"].iloc[0]), float(r["interval_width_k"].iloc[0])))
    full = var[(var["arm"] == "contiguous")].sort_values("k_cases").groupby("method").tail(1).set_index("method")
    for m, mk in [("emos_pooled", "o"), ("emos_local", "^"), ("emos_local_leadspec", "^")]:
        if m in full.index:
            pts.append((m, mk, float(full.loc[m, "coverage_80pct"]), float(full.loc[m, "interval_width_k"])))
    drn = sweep[(sweep["method"] == "drn") & (sweep["sampling_arm"] == "contiguous")].sort_values("n_cases").iloc[-1]
    pts.append(("drn", "s", float(drn["coverage_80pct"]), float(drn["interval_width_k"])))
    ra = _load("rolling_emos")
    ra = ra[(ra["method"] == "rolling_emos_local_adaptive") & (ra["window_days"] == 40)].iloc[0]
    pts.append(("rolling_emos_local_adaptive", "o", float(ra["coverage_80pct"]), float(ra["interval_width_k"])))
    handles = [Line2D([], [], color=COLORS["var_inflation_fixed"], label="Ensemble rescaled by λ (0.5–3)")]
    for key, mk, cov, wid in pts:
        ax.plot(cov, wid, marker=mk, color=COLORS[key], markersize=7, linestyle="none", markeredgecolor="black", markeredgewidth=0.4)
        handles.append(Line2D([], [], marker=mk, color=COLORS[key], linestyle="none", markeredgecolor="black",
                              markeredgewidth=0.4, label=LABELS.get(key, key).replace("(zero-shot)", "(40-day context)")))
    ax.axvline(0.8, color=COLORS["reference"], linestyle="--", linewidth=STYLE["linewidth"]["reference"])
    ax.set_xlabel("80 % coverage")
    ax.set_ylabel("Mean width of the 10–90 % interval (K)")
    _legend(fig, handles, ncol=3)
    _save_figure(fig, "fig06_calibration", ["results/lambda_curve.parquet", "results/reference_scores.parquet"])
    plt.close(fig)


def fig07_rank_histograms():
    h = _load("rank_histograms")
    methods = [m for m in ["raw_ensemble", "tsfm3", "chronos2", "emos_pooled", "emos_local"] if m in set(h["method"])]
    _apply_rcparams()
    fig, axes = plt.subplots(1, len(methods), figsize=_figsize_in("double", 60), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.2, wspace=0.08)
    for ax, m, letter in zip(axes, methods, "abcdefg"):
        d = h[h["method"] == m].sort_values("bin_index")
        ax.bar(d["bin_index"], d["fraction"], color=COLORS[m], alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axhline(0.1, color=COLORS["reference"], linestyle="--", linewidth=STYLE["linewidth"]["reference"])
        ax.set_xticks([0, 9])
        ax.set_xticklabels(["< q0.1", "> q0.9"], fontsize=STYLE["font"]["tick_label_pt"] - 1)
        _panel(ax, letter, LABELS.get(m, m).replace(" (zero-shot)", ""))
        ax.title.set_fontsize(STYLE["font"]["annotation_pt"])
    axes[0].set_ylabel("Relative frequency")
    fig.supxlabel("Rank of the observation among the nine predicted quantiles", fontsize=STYLE["font"]["axis_label_pt"])
    _save_figure(fig, "fig07_rank_histograms", ["results/rank_histograms.parquet"])
    plt.close(fig)


FIGS = {"fig02": fig02_recent_data, "fig03": fig03_training_size, "fig04": fig04_seasonal,
        "fig05": fig05_lead_time, "fig06": fig06_calibration, "fig07": fig07_rank_histograms}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(FIGS):
        FIGS[name]()
        print("wrote", name)
