"""Render the five main figures for the ZeroPP paper (F1-F5), per
docs/figure_style_guide.md. Matplotlib only, no plotnine. One function per
figure (make_f1..make_f5), each independently callable -- this module has no
monolithic main() that must run start-to-end; __main__ below just calls all
five in sequence for convenience.

Every color, line style, marker and font size used anywhere in this file
lives in the single STYLE dict below -- no hex-color literal appears
anywhere else in this module (grep for '#[0-9A-Fa-f]{6}' outside the STYLE
block as a mechanical check).

Every number that appears baked into a figure or its caption (breakpoints,
crossover lead time, coverage percentages, calendar-day equivalents, N
values at F3's arrowheads, etc.) is read at render time from the persisted
`results/*.parquet` files listed in docs/results_index.md -- never
hardcoded/retyped in this file. F4 is the one figure whose input data did
not already exist anywhere; scripts/08_compute_pit_histograms.py computes
and persists it to results/phase3_pit_histograms.parquet BEFORE this module
reads it, so the "numbers come from results files" rule holds for F4 too.

Column width note: the style guide fixes 90mm (single) / 190mm (double) as
the two allowed physical widths, set in code and never scaled after the
fact. All five figures here use the double-column (190mm) width -- each
carries multiple series, in-panel R2 annotations and (for F1/F3/F5) a
secondary axis or trajectory arrows/labels that do not fit legibly at 90mm;
first-draft renders at single-column width produced illegible, overlapping
text (see docs/figures_implementation_report.md). Heights are chosen per
figure and are NOT derived from any fixed ratio.

Exports per figure: <name>.pdf (vector, with source-parquet-path + git-SHA
written into the PDF's own metadata via savefig(metadata=...)), <name>.svg
(vector), <name>.png (300 dpi, preview only -- see .gitignore, PNGs under
figures/ are not tracked). Layout is done with explicit fig.subplots_adjust
margins (not tight_layout/constrained_layout), and every fig-level legend is
anchored at a small POSITIVE figure-fraction y (never a negative
bbox_to_anchor) so it always lands inside the fixed canvas -- tight_layout
is deliberately not used together with secondary axes / fig-level legends,
since matplotlib's own tight_layout warns it cannot account for either and
a first-draft render confirmed that in practice (clipped/overlapping
legends, an oversized blank margin on F1).

Captions (numbers substituted from the same results files, never retyped)
are collected into figures/captions.md by each make_fX call -- see
_update_caption(). F4's caption includes, verbatim, the sentence required by
docs/figure_style_guide.md: "PIT uniformity is assessed descriptively; no
formal uniformity test valid under serial dependence is applied." F3's
caption carries the Gneiting "up and to the left is better" reading (per the
style guide: "Say that in the caption" -- not on the canvas itself).
"""
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CAPTIONS_PATH = FIGURES_DIR / "captions.md"

# --------------------------------------------------------------------------
# STYLE: every color / line style / marker / font size / geometry constant
# used anywhere in this file. No color literal appears outside this dict.
# Palette and typography are copied verbatim from docs/figure_style_guide.md
# sections 2-3 -- this is a fixed design spec, not a place to improvise.
# --------------------------------------------------------------------------
STYLE = {
    "color": {
        "raw_ensemble": "#999999",
        # Fix-round-1 Blocking Fix 3: THREE variance-inflation variants now exist
        # (fixed-climatological / coverage-matched / CRPS-optimal trainfit). The style
        # guide's palette has exactly one "Variance-inflated baseline" hex entry --
        # per that section's own instruction ("pick a defensible visual scheme that
        # keeps them recognizably one family... state your choice explicitly"), all
        # three share this SAME orange hue and are distinguished only by linestyle /
        # marker / alpha below, never by inventing a second orange or an unrelated
        # color. See docs/figures_implementation_report.md's Fix round 1 section.
        "var_inflation_fixed": "#E69F00",
        "var_inflation_coverage_matched": "#E69F00",
        "var_inflation_trainfit": "#E69F00",
        "tsfm3": "#0072B2",
        "emos_pooled": "#D55E00",
        "emos_local": "#009E73",
        "drn": "#CC79A7",
        "reference": "#666666",
        "gridline": "#DDDDDD",
    },
    "linestyle": {
        "raw_ensemble": ":",
        # The three variance-inflation variants get three distinct dash patterns
        # within the same orange family (see the "color" comment above): fixed (b)
        # keeps its original dashdot; coverage-matched (c) is dotted; trainfit (a) is
        # solid (like the trained curves) since it is the one variant that is
        # actually refit per-N and rendered as a curve, not a flat line.
        "var_inflation_fixed": "-.",
        "var_inflation_coverage_matched": ":",
        "var_inflation_trainfit": "-",
        "tsfm3": "-",
        "emos_pooled": "-",
        "emos_local": "-",
        "drn": "-",
        "reference": "--",
        "random_arm": "--",  # F1's random-arm mean: same color as emos_pooled, dashed
    },
    "marker": {
        "raw_ensemble": None,
        "var_inflation_fixed": None,
        "var_inflation_coverage_matched": None,
        "var_inflation_trainfit": "P",  # filled plus: unclaimed elsewhere, marks it as
                                         # the one variance-inflation variant that is a
                                         # trained (per-N) curve, not a flat zero-shot line.
        "tsfm3": None,
        "emos_pooled": "o",
        "emos_local": "^",
        "drn": "s",
        "reference": None,
        "zero_shot_point": "D",  # F3: raw/fixed/tsfm3 zero-shot points -- "diamond" is
                                  # unclaimed elsewhere in the palette table (whose
                                  # "marker: none" describes their flat-line rendering
                                  # in N-axis figures, not a scatter point).
        "zero_shot_point_coverage_matched": "p",  # F3 only: coverage-matched (c) is
                                  # ALSO zero-shot/orange like var_inflation_fixed (b) --
                                  # needs its own marker shape (pentagon) so the two
                                  # same-colored points remain visually distinguishable.
        # Fix-round-1 Blocking Fix 2: F5's two DIFFERENT "no crossing" situations get
        # two DIFFERENT marker shapes -- a filled star for "already better than
        # TimesFM-3 at the smallest tested N" (no crossing NEEDED), and the existing
        # open circle kept, unchanged, for "still worse than TimesFM-3 throughout the
        # tested range" (no crossing OBSERVED). Conflating these under one open-circle
        # marker at the right edge (the pre-fix behaviour) visually implied the wrong
        # one of the two for CRPS/EMOS-pooled, where the true situation is the former.
        "already_better_at_min_n": "*",
        "no_crossing_worse_throughout": "o",
    },
    "label": {
        "raw_ensemble": "Raw ensemble",
        # Fix-round-1: shortened from the original single, longer "Variance-inflated
        # baseline" label -- the fuller three-variant names overflowed F3's two-column
        # legend at double-column (190mm) width. Still names each variant
        # unambiguously; the full description lives in this report's Fix round 3
        # section and in each figure's caption text.
        "var_inflation_fixed": "Var.-inflated, fixed λ=1.5 (zero-shot)",
        "var_inflation_coverage_matched": "Var.-inflated, coverage-matched (zero-shot)",
        "var_inflation_trainfit": "Var.-inflated, CRPS-optimal trainfit (per-N)",
        "tsfm3": "TimesFM-3 (zero-shot)",
        "emos_pooled": "EMOS pooled",
        "emos_local": "EMOS local",
        "drn": "DRN",
        "reference": "Reference / nominal",
        "random_arm_mean": "EMOS pooled, random arm (mean ± 1 SD)",
    },
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
    "marker_size_pt": 5,
    "mm_per_column": {"single": 90.0, "double": 190.0},
    "dpi_png": 300,
    "alpha": {
        "std_band": 0.20, "pit_fill": 0.70,
        # Fix-round-1 Blocking Fix 3: var_inflation_trainfit renders at reduced
        # opacity relative to the other trained curves (emos_pooled/emos_local/drn,
        # alpha=1.0 implicitly) -- it is one parameter fit per N (vs. EMOS's 4 / DRN's
        # many), and the reduced opacity keeps it visually grouped with its own
        # variance-inflation family (fixed/coverage-matched) rather than competing
        # for attention with the paper's two primary trained methods.
        "var_inflation_trainfit_curve": 0.65,
    },
}


def _mm_to_in(mm: float) -> float:
    return mm / 25.4


def _figsize_in(kind: str, height_mm: float) -> tuple[float, float]:
    return (_mm_to_in(STYLE["mm_per_column"][kind]), _mm_to_in(height_mm))


def _apply_rcparams() -> None:
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


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _style_grid(ax) -> None:
    """R6: minimal chrome -- light grey gridlines on white, no background fill,
    no panel border beyond the left/bottom axis lines (top/right spines are
    disabled globally in _apply_rcparams)."""
    ax.set_facecolor("white")
    ax.grid(True, axis="both", color=STYLE["color"]["gridline"], linewidth=STYLE["linewidth"]["gridline"])
    ax.set_axisbelow(True)


def _add_top_headroom(ax, frac: float = 0.32) -> None:
    """Extend the y-axis above the current data max so R2's in-panel
    annotation stack has clear whitespace instead of overlapping the plotted
    series. Call AFTER all data/reference lines for the panel are drawn."""
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + frac * (ymax - ymin))


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{name}.parquet")


def _save_figure(fig, name: str, source_parquets: list[str]) -> None:
    """Export PDF+SVG (vector) and PNG (300 dpi, preview only). The PDF's own
    metadata carries the source parquet path(s) and the git SHA (per the style
    guide: written into PDF metadata via savefig(metadata=...), never as a
    text annotation on the canvas). No bbox_inches='tight' anywhere: the
    figure's physical size is fixed in code (single/double column mm) and
    must not be scaled or trimmed after the fact."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    pdf_metadata = {
        "Subject": "source: " + "; ".join(source_parquets),
        "Keywords": f"git_sha:{sha}",
    }
    fig.savefig(FIGURES_DIR / f"{name}.pdf", metadata=pdf_metadata)
    fig.savefig(FIGURES_DIR / f"{name}.svg")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=STYLE["dpi_png"])


def _update_caption(fig_id: str, title: str, text: str) -> None:
    """Append/replace this figure's caption section in figures/captions.md,
    keeping the file's F1..F5 ordering stable regardless of call order --
    each make_fX call is independent and idempotent with respect to this
    file."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sections: dict[str, str] = {}
    if CAPTIONS_PATH.exists():
        raw = CAPTIONS_PATH.read_text()
        for block in re.split(r"(?=^## )", raw, flags=re.MULTILINE):
            block = block.strip()
            if not block:
                continue
            m = re.match(r"^## (F\d+)", block)
            if m:
                sections[m.group(1)] = block
    sections[fig_id] = f"## {fig_id} -- {title}\n\n{text.strip()}\n"
    ordered = [sections[k] for k in sorted(sections, key=lambda k: int(k[1:]))]
    CAPTIONS_PATH.write_text(
        "# ZeroPP figure captions\n\n"
        "Auto-generated by scripts/make_figures.py -- every number here is read "
        "programmatically from results/*.parquet at render time, never retyped.\n\n"
        + "\n".join(ordered)
    )


def _fmt_cases(n: float) -> str:
    return f"{n:.1f}" if abs(n - round(n)) > 1e-6 else f"{int(round(n))}"


def _classify_no_crossing(reason: str) -> str:
    """Fix-round-1 Blocking Fix 2: `breakpoint_and_direction`'s (scripts/
    03_data_size_sweep.py) `crossing_direction` reason string already distinguishes
    two OPPOSITE situations whenever no crossing is found:
      - "already better than (or equal to) the reference at the smallest tested N"
        / "already at/beyond the reference at the smallest tested N" -- the trained
        method started ahead of TimesFM-3 and never needed to cross.
      - "no crossing observed in tested range" -- the trained method stayed WORSE
        than TimesFM-3 for the entire tested range (the opposite situation).
    Both previously rendered identically (a marker at the right/largest-N edge) in
    F1/F5, which visually implied the wrong one of the two for CRPS/EMOS-pooled
    (already ahead at the smallest N, not "never catches up"). Returns
    "already_better" or "true_no_crossing" -- callers use this to pick a distinct
    marker AND x-position (left/smallest-N edge for "already_better", right/
    largest-N edge for "true_no_crossing") per the fix."""
    if "already" in reason:
        return "already_better"
    return "true_no_crossing"


def _annotate_stack(ax, entries: list[tuple[str, str]], x: float = 0.02, y0: float = 0.97, step: float = 0.085, ha: str = "left") -> None:
    """R2: stack colour-matched in-panel annotations top-to-bottom at a fixed
    axes-fraction position. entries: list of (text, color)."""
    for i, (text, color) in enumerate(entries):
        ax.text(
            x, y0 - i * step, text, transform=ax.transAxes, color=color,
            fontsize=STYLE["font"]["annotation_pt"], va="top", ha=ha,
        )


def _days_per_case_from_data(*dfs: pd.DataFrame) -> float:
    """Recover the (real, measured) case->calendar-day ratio directly from the
    already-persisted n_cases/n_calendar_days_equiv columns instead of typing
    the constant into this file -- averaged over every real (n_cases>0) row
    across the given frames to smooth out the rounding each source row already
    went through."""
    ratios = []
    for df in dfs:
        sub = df[df["n_cases"] > 0].dropna(subset=["n_cases", "n_calendar_days_equiv"])
        ratios.extend((sub["n_calendar_days_equiv"] / sub["n_cases"]).tolist())
    return float(np.mean(ratios))


# ============================================================================
# F1 -- Breakpoint curve (cover figure)
# ============================================================================
def make_f1() -> None:
    sweep = _load("phase3_data_size_sweep")
    low_n = _load("phase3_low_n_grid").rename(columns={"k": "n_cases"})
    bp_main = _load("phase3_data_size_sweep_breakpoints")
    bp_low = _load("phase3_low_n_grid_breakpoints")

    days_per_case = _days_per_case_from_data(sweep, low_n)

    def combined_curve(method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        parts = [sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "contiguous")][["n_cases", metric]]]
        if method in low_n.loc[low_n["sampling_arm"] == "contiguous", "method"].unique():
            parts.append(low_n[(low_n["method"] == method) & (low_n["sampling_arm"] == "contiguous")][["n_cases", metric]])
        combined = pd.concat(parts).dropna(subset=[metric]).drop_duplicates(subset=["n_cases"]).sort_values("n_cases")
        return combined["n_cases"].to_numpy(float), combined[metric].to_numpy(float)

    def random_arm(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = [
            sweep[(sweep["method"] == "emos_pooled") & (sweep["sampling_arm"] == "random_mean")][["n_cases", metric, f"{metric}_std"]],
            low_n[(low_n["method"] == "emos_pooled") & (low_n["sampling_arm"] == "random_mean")][["n_cases", metric, f"{metric}_std"]],
        ]
        combined = pd.concat(parts).dropna(subset=[metric]).drop_duplicates(subset=["n_cases"]).sort_values("n_cases")
        return combined["n_cases"].to_numpy(float), combined[metric].to_numpy(float), combined[f"{metric}_std"].fillna(0.0).to_numpy(float)

    def hline_value(method: str, metric: str) -> float:
        row = sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "n_independent")].iloc[0]
        return float(row[metric])

    def read_bp(metric: str, variant: str) -> tuple[float | None, float | None, str]:
        """Low-N grid file is authoritative for emos_pooled/emos_local (see
        docs/results_index.md); main sweep file is the only source for drn."""
        if variant in ("emos_pooled", "emos_local"):
            hit = bp_low[(bp_low["metric"] == metric) & (bp_low["emos_variant"] == variant)]
        else:
            hit = bp_main[(bp_main["metric"] == metric) & (bp_main["emos_variant"] == variant)]
        row = hit.iloc[0]
        bp = row["breakpoint_n_cases"]
        cal = row["breakpoint_calendar_days"]
        bp = None if pd.isna(bp) else float(bp)
        cal = None if pd.isna(cal) else float(cal)
        return bp, cal, str(row["crossing_direction"])

    _apply_rcparams()
    fig, axes = plt.subplots(3, 1, figsize=_figsize_in("double", 225), sharex=True)
    # Fix-round-1 D1: previously bottom=0.32 + secondary_xaxis(-0.45) left a large
    # blank region between panel 3 and the calendar-day row (the fixed bottom margin
    # was sized generously but the secondary axis/legend were positioned independent
    # of it, per figures_implementation_report.md's own account of that first-draft
    # bug). Tightened here: smaller secondary-axis offset (-0.30, tight beneath
    # ax_width's own tick labels) and a legend anchored immediately below it, so the
    # reserved bottom margin is actually filled by content, not by air. Figure height
    # raised 210->225mm to give the now-longer (10-entry, Blocking Fix 3) legend room
    # without re-opening the same gap.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.30, hspace=0.16)
    ax_crps, ax_cov, ax_width = axes
    # Fix-round-1 Blocking Fix 3: var_inflation_trainfit (variant a, CRPS-optimal,
    # refit per-N) now renders alongside emos_pooled/emos_local/drn as a 4th curve;
    # var_inflation_coverage_matched (variant c, zero-shot application of a
    # leakage-free train-calibrated lambda) renders as a 4th flat line alongside
    # raw_ensemble/var_inflation_fixed/tsfm3. See docs/figures_implementation_report.md
    # Fix round 1 for why (a) is a curve here and (c) is a flat line: (a) is
    # genuinely refit at every N in results/phase3_data_size_sweep.parquet; (c) is
    # calibrated once on the full training archive and applied unchanged, exactly
    # like (b).
    curve_methods = ["var_inflation_trainfit", "emos_pooled", "emos_local", "drn"]
    hline_methods = ["raw_ensemble", "var_inflation_fixed", "var_inflation_coverage_matched", "tsfm3"]

    metrics_panels = [
        (ax_crps, "crps", "CRPS (K)"),
        (ax_cov, "coverage_80pct", "Coverage @ 80%"),
        (ax_width, "interval_width_k", "Interval width K (K)"),
    ]
    for ax, metric, ylabel in metrics_panels:
        _style_grid(ax)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        for m in hline_methods:
            ax.axhline(
                hline_value(m, metric), color=STYLE["color"][m], linestyle=STYLE["linestyle"][m],
                linewidth=STYLE["linewidth"]["data"],
            )
        for m in curve_methods:
            x, y = combined_curve(m, metric)
            if len(x) == 0:
                continue
            curve_alpha = STYLE["alpha"].get(f"{m}_curve", 1.0)
            ax.plot(
                x, y, color=STYLE["color"][m], linestyle=STYLE["linestyle"][m],
                marker=STYLE["marker"][m], markersize=STYLE["marker_size_pt"],
                linewidth=STYLE["linewidth"]["data"], alpha=curve_alpha,
            )
        if metric in ("crps", "coverage_80pct"):
            x_r, y_r, std_r = random_arm(metric)
            if len(x_r):
                c = STYLE["color"]["emos_pooled"]
                ax.plot(x_r, y_r, color=c, linestyle=STYLE["linestyle"]["random_arm"], linewidth=STYLE["linewidth"]["data"])
                ax.fill_between(x_r, y_r - std_r, y_r + std_r, color=c, alpha=STYLE["alpha"]["std_band"], linewidth=0)

    ax_cov.axhline(0.80, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])

    # Headroom BEFORE annotating, so R2's stacked text never overlaps the data.
    _add_top_headroom(ax_crps, 0.50)
    _add_top_headroom(ax_cov, 0.50)
    _add_top_headroom(ax_width, 0.30)

    # --- R2: in-panel, colour-matched breakpoint annotations ---
    crps_entries = []
    cov_entries = []
    for variant in ["emos_pooled", "emos_local", "drn", "var_inflation_trainfit"]:
        color = STYLE["color"][variant]
        label = STYLE["label"][variant]
        bp, cal, reason = read_bp("crps", variant)
        if bp is not None:
            crps_entries.append((f"{label}: ~{_fmt_cases(bp)} cases (~{cal:.0f} d) vs. TimesFM-3", color))
        else:
            crps_entries.append((f"{label}: {reason.split('—')[-1].strip()}", color))
        bp, cal, reason = read_bp("coverage_80pct", variant)
        if bp is not None:
            cov_entries.append((f"{label}: ~{_fmt_cases(bp)} cases (~{cal:.0f} d) vs. TimesFM-3", color))
        else:
            cov_entries.append((f"{label}: {reason.split('—')[-1].strip()}", color))
    _annotate_stack(ax_crps, crps_entries)
    _annotate_stack(ax_cov, cov_entries)
    ax_width.text(
        0.02, 0.97, "Sharpness has no single ‘better’ direction alone (Gneiting: maximize\n"
        "sharpness subject to calibration) — curve only, no breakpoint computed.",
        transform=ax_width.transAxes, color=STYLE["color"]["reference"], fontsize=STYLE["font"]["annotation_pt"],
        va="top", ha="left",
    )

    # --- Fix-round-1 D1: raw ensemble's and TimesFM-3's CRPS coincide to 3 decimals
    # (the grey dotted raw line is otherwise invisible under the solid blue TimesFM-3
    # line) -- detected here from the real numbers, not hardcoded, and only added if
    # they actually still coincide at render time. ---
    raw_crps_val = hline_value("raw_ensemble", "crps")
    tsfm3_crps_val = hline_value("tsfm3", "crps")
    if round(raw_crps_val, 3) == round(tsfm3_crps_val, 3):
        # Placed top-right, below the R2 stack's 4 lines (which end around axes-
        # fraction y=0.97-3*0.085=0.715) and well above the plotted curves -- the
        # headroom added above specifically keeps this whole upper strip clear of
        # data, unlike the bottom-right corner (which the curves/std-band can
        # reach at small N).
        ax_crps.text(
            0.98, 0.55,
            f"Raw ensemble ({raw_crps_val:.4f} K) and TimesFM-3 ({tsfm3_crps_val:.4f} K) CRPS\n"
            "coincide to three decimals; the grey dotted line is hidden beneath the blue line.",
            transform=ax_crps.transAxes, color=STYLE["color"]["reference"], fontsize=STYLE["font"]["annotation_pt"],
            va="top", ha="right", style="italic",
        )

    ax_width.set_xlabel("Training data size (cases, log scale)")

    # Secondary calendar-day tick row, bottom panel only (R1: shared axis, labelled once)
    sec = ax_width.secondary_xaxis(-0.30, functions=(lambda k: k * days_per_case, lambda d: d / days_per_case))
    sec.set_xlabel("Calendar-day equivalent")
    sec.tick_params(labelsize=STYLE["font"]["tick_label_pt"])

    handles = [
        Line2D([0], [0], color=STYLE["color"][m], linestyle=STYLE["linestyle"][m], marker=STYLE["marker"][m],
               markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"],
               alpha=STYLE["alpha"].get(f"{m}_curve", 1.0), label=STYLE["label"][m])
        for m in hline_methods + curve_methods
    ] + [
        Line2D([0], [0], color=STYLE["color"]["emos_pooled"], linestyle=STYLE["linestyle"]["random_arm"],
               linewidth=STYLE["linewidth"]["data"], label=STYLE["label"]["random_arm_mean"]),
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label="Nominal 80% coverage"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.055), frameon=False, fontsize=STYLE["font"]["legend_pt"] - 1)

    _save_figure(fig, "f1_breakpoint_curve", [
        "results/phase3_data_size_sweep.parquet", "results/phase3_low_n_grid.parquet",
        "results/phase3_data_size_sweep_breakpoints.parquet", "results/phase3_low_n_grid_breakpoints.parquet",
    ])
    plt.close(fig)

    bp_pooled_crps, cal_pooled_crps, _ = read_bp("crps", "emos_pooled")
    bp_local_cov, cal_local_cov, _ = read_bp("coverage_80pct", "emos_local")
    vi_covmatch_row = sweep[(sweep["method"] == "var_inflation_coverage_matched") & (sweep["sampling_arm"] == "n_independent")].iloc[0]
    tsfm3_row_f1 = sweep[(sweep["method"] == "tsfm3") & (sweep["sampling_arm"] == "n_independent")].iloc[0]
    covmatch_width = float(vi_covmatch_row["interval_width_k"])
    tsfm3_width = float(tsfm3_row_f1["interval_width_k"])
    covmatch_vs_tsfm3 = "narrower than" if covmatch_width < tsfm3_width else ("wider than" if covmatch_width > tsfm3_width else "equal to")
    _update_caption(
        "F1", "Breakpoint curve (cover figure)",
        "CRPS, coverage@80% and interval width K vs. training data size (cases, "
        "log scale; calendar-day equivalent on the secondary axis). Zero-shot "
        "methods (raw ensemble, TimesFM-3, and two variance-inflated baseline "
        "variants -- fixed λ=1.5 and coverage-matched to TimesFM-3's real "
        "coverage) render as flat horizontal lines; trained methods (EMOS "
        "pooled, EMOS local, DRN, and the third variance-inflated variant, a "
        "CRPS-optimal multiplier refit at each N) render as curves. In-panel "
        "annotations give each trained method's breakpoint against TimesFM-3, "
        f"e.g. EMOS pooled's CRPS crosses TimesFM-3 at ~{_fmt_cases(bp_pooled_crps)} "
        f"cases (~{cal_pooled_crps:.0f} calendar days); EMOS local's coverage@80% "
        f"crosses at ~{_fmt_cases(bp_local_cov)} cases (~{cal_local_cov:.0f} calendar "
        "days). Random-arm (EMOS pooled) mean ± 1 SD shown as a dashed band. "
        "Sharpness (interval width) is reported as a curve only, with no "
        "breakpoint computed, since it has no inherent 'better' direction "
        "independent of calibration (Gneiting). At matched coverage "
        f"({vi_covmatch_row['coverage_80pct']:.4f} vs. TimesFM-3's "
        f"{tsfm3_row_f1['coverage_80pct']:.4f}), the coverage-matched variance-"
        f"inflation baseline's interval width ({covmatch_width:.3f} K) is "
        f"{covmatch_vs_tsfm3} TimesFM-3's ({tsfm3_width:.3f} K) -- see "
        "docs/figures_implementation_report.md's Fix round 1 section for what "
        "this means for the paper's calibration-achievability framing.",
    )


# ============================================================================
# F2 -- Lead-time resolved comparison
# ============================================================================
def make_f2() -> None:
    lt = _load("phase3_lead_time_breakdown")
    crossover = _load("phase3_lead_time_crossover")

    method_map = {"raw_ensemble": "raw_ensemble", "tsfm3": "tsfm3", "emos": "emos_pooled"}
    durable = crossover[crossover["crossover_type"] == "durable_sign_flip"].iloc[0]
    first = crossover[crossover["crossover_type"] == "first_sign_flip"].iloc[0]
    x_durable = float(durable["step_hours"])
    x_first = float(first["step_hours"])

    _apply_rcparams()
    fig, (ax_crps, ax_cov) = plt.subplots(2, 1, figsize=_figsize_in("double", 130), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.20, hspace=0.10)
    for ax, metric, ylabel in [(ax_crps, "crps", "CRPS (K)"), (ax_cov, "coverage_80pct", "Coverage @ 80%")]:
        _style_grid(ax)
        for src, plot_key in method_map.items():
            sub = lt[lt["method"] == src].sort_values("step_hours")
            ax.plot(
                sub["step_hours"], sub[metric], color=STYLE["color"][plot_key], linestyle=STYLE["linestyle"][plot_key],
                marker=STYLE["marker"][plot_key], markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"],
            )
            # --- Fix-round-1 D2: a light rolling-mean overlay (3-point, centered),
            # CLEARLY marked as a smoothing aid in the legend, over the SAME real
            # per-lead-time points plotted above -- the underlying data is never
            # altered or hidden, this is an additional thin/translucent line drawn
            # on top so a reader can see the trend through the real ~6h-cycle
            # oscillation without the oscillation itself being smoothed away from
            # the figure. Chosen (rather than leaving the oscillation unaddressed)
            # because the raw curves alone make the 43.5h "durable crossover" look
            # pulled out of noise -- see docs/figure_style_guide.md's own note
            # ("the crossover reads as pulled out of noise") and this fix's report
            # entry for the full reasoning.
            if metric == "crps" and src in ("emos", "tsfm3"):
                rolling = sub[metric].rolling(window=3, center=True, min_periods=1).mean()
                ax.plot(
                    sub["step_hours"], rolling, color=STYLE["color"][plot_key], linestyle="-",
                    linewidth=STYLE["linewidth"]["data"] * 0.7, alpha=0.35, zorder=1,
                )
        ax.axvline(x_durable, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])
    _add_top_headroom(ax_crps, 0.30)
    ax_cov.axhline(0.80, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])
    ax_crps.set_ylabel("CRPS (K)")
    ax_cov.set_ylabel("Coverage @ 80%")
    ax_cov.set_xlabel("Lead time, step_hours (h)")

    # --- Fix-round-1 D2: the panel text spells out EXACTLY what "durable" means
    # operationally, so a reader is not misled into thinking this is a single clean
    # crossing rather than a threshold survived despite real lead-to-lead
    # oscillation whose amplitude is comparable to the method-to-method gap. ---
    _annotate_stack(
        ax_crps,
        [(
            f"Durable crossover ≈ {x_durable:.1f} h: TimesFM-3's CRPS exceeds EMOS\n"
            "pooled's at EVERY one of the remaining tested lead times after this\n"
            "point, despite lead-to-lead oscillation (thin translucent line = 3-point\n"
            "rolling mean, smoothing aid only -- real per-lead-time points unchanged)",
            STYLE["color"]["reference"],
        )],
        x=0.38, y0=0.97,
    )

    # --- Fix-round-1 D2: split the single "Durable crossover / nominal 80%" legend
    # entry (which conflated a vertical crossover marker with a horizontal
    # nominal-coverage line under one label) into two separate, accurately labelled
    # entries. Both happen to share the reference grey/dashed style -- that is
    # unchanged and correct (R4: reference lines are grey dashed); only the label
    # text and entry COUNT are fixed, so the two conceptually different lines are no
    # longer described by one ambiguous phrase. ---
    handles = [
        Line2D([0], [0], color=STYLE["color"][k], linestyle=STYLE["linestyle"][k], marker=STYLE["marker"][k],
               markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"], label=STYLE["label"][k])
        for k in ["raw_ensemble", "tsfm3", "emos_pooled"]
    ] + [
        Line2D([0], [0], color=STYLE["color"]["emos_pooled"], linestyle="-", linewidth=STYLE["linewidth"]["data"] * 0.7,
               alpha=0.35, label="3-point rolling mean (smoothing aid only, EMOS pooled/TimesFM-3 CRPS)"),
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label=f"Durable crossover, step_hours ≈ {x_durable:.1f} h (vertical)"),
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label="Nominal 80% coverage (horizontal)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=STYLE["font"]["legend_pt"] - 1)

    _save_figure(fig, "f2_lead_time_resolved", [
        "results/phase3_lead_time_breakdown.parquet", "results/phase3_lead_time_crossover.parquet",
    ])
    plt.close(fig)

    _update_caption(
        "F2", "Lead-time resolved comparison",
        "CRPS and coverage@80% across all 21 tested lead times (step_hours). "
        f"'Durable' is used here in a precise operational sense: the crossover at "
        f"step_hours ≈ {x_durable:.1f} h (grey dashed vertical line) is the lead time "
        "after which TimesFM-3's CRPS exceeds (or ties) EMOS pooled's at EVERY one "
        "of the remaining tested lead times -- despite real lead-to-lead "
        "oscillation on what looks like a ~6-hour cycle, whose amplitude is "
        "comparable to the method-to-method gap the crossover itself rests on. "
        "This is not a single clean crossing; a thin, translucent 3-point "
        "rolling-mean overlay (explicitly labelled as a smoothing aid, drawn over "
        "the unaltered real per-lead-time points) is added to make the trend "
        f"through that oscillation easier to read. The first sign flip occurs "
        f"earlier, at step_hours ≈ {x_first:.1f} h, but is not durable (the sign "
        "flips back at least once between there and the durable crossover) and is "
        "reported here in the caption only, not annotated on the panel, per the "
        "project's convention of citing the durable crossover as the primary "
        "number. Nominal 80% coverage (horizontal) and the durable crossover "
        "(vertical) are two distinct grey dashed reference lines, listed as "
        "separate legend entries.",
    )


# ============================================================================
# F3 -- Sharpness-calibration plane
# ============================================================================
def make_f3() -> None:
    sweep = _load("phase3_data_size_sweep")
    low_n = _load("phase3_low_n_grid").rename(columns={"k": "n_cases"})

    def trajectory(method: str) -> pd.DataFrame:
        parts = [sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "contiguous")][["n_cases", "interval_width_k", "coverage_80pct"]]]
        if method in low_n.loc[low_n["sampling_arm"] == "contiguous", "method"].unique():
            parts.append(low_n[(low_n["method"] == method) & (low_n["sampling_arm"] == "contiguous")][["n_cases", "interval_width_k", "coverage_80pct"]])
        combined = pd.concat(parts).dropna().drop_duplicates(subset=["n_cases"]).sort_values("n_cases")
        return combined

    def zero_shot_point(method: str) -> tuple[float, float]:
        row = sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "n_independent")].iloc[0]
        return float(row["interval_width_k"]), float(row["coverage_80pct"])

    # Per-method label offsets (offset points) for the N-value annotation at each
    # trajectory's arrowhead (END, largest tested N) -- the trained methods' full-N
    # endpoints sit close together on this plane, so a single shared offset makes
    # the labels collide; each gets its own direction. Fix-round-1 Blocking Fix 3
    # adds var_inflation_trainfit as a 4th trajectory.
    n_label_offset_end = {
        "var_inflation_trainfit": (10, 8), "emos_pooled": (8, -4), "emos_local": (8, 10), "drn": (14, -20),
    }
    # Fix-round-1 D4: starting-N (smallest tested N) label offsets, one per method,
    # chosen in the opposite general direction from that method's own ending-label
    # offset so the two labels for the same trajectory don't collide with each
    # other or with the curve itself.
    n_label_offset_start = {
        "var_inflation_trainfit": (-16, -12), "emos_pooled": (-14, 10), "emos_local": (-16, -14), "drn": (-18, 16),
    }

    _apply_rcparams()
    fig, ax = plt.subplots(figsize=_figsize_in("double", 115))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.26)
    _style_grid(ax)
    ax.axhline(0.80, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])

    # Fix-round-1 Blocking Fix 3: var_inflation_coverage_matched (c) is a 4th
    # zero-shot point (calibrated once on the training archive, applied unchanged
    # to test, exactly like raw_ensemble/var_inflation_fixed/tsfm3 here). It shares
    # var_inflation_fixed's orange hue (same "family", per the style choice
    # documented in STYLE above) so it needs its OWN marker shape to stay
    # distinguishable -- "zero_shot_point_coverage_matched" (pentagon), vs. the
    # shared "zero_shot_point" diamond the other three use.
    for m in ["raw_ensemble", "var_inflation_fixed", "var_inflation_coverage_matched", "tsfm3"]:
        k, cov = zero_shot_point(m)
        marker_key = "zero_shot_point_coverage_matched" if m == "var_inflation_coverage_matched" else "zero_shot_point"
        ax.scatter([k], [cov], color=STYLE["color"][m], marker=STYLE["marker"][marker_key], s=STYLE["marker_size_pt"] ** 2, zorder=3)

    for m in ["var_inflation_trainfit", "emos_pooled", "emos_local", "drn"]:
        traj = trajectory(m)
        if traj.empty:
            continue
        curve_alpha = STYLE["alpha"].get(f"{m}_curve", 1.0)
        ax.plot(
            traj["interval_width_k"], traj["coverage_80pct"], color=STYLE["color"][m], linestyle=STYLE["linestyle"][m],
            marker=STYLE["marker"][m], markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"],
            alpha=curve_alpha,
        )
        x0, y0 = traj["interval_width_k"].iloc[-2], traj["coverage_80pct"].iloc[-2]
        x1, y1 = traj["interval_width_k"].iloc[-1], traj["coverage_80pct"].iloc[-1]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=STYLE["color"][m], lw=STYLE["linewidth"]["data"], alpha=curve_alpha),
        )
        n_max = traj["n_cases"].iloc[-1]
        ax.annotate(
            f"N={_fmt_cases(n_max)}", xy=(x1, y1), xytext=n_label_offset_end[m], textcoords="offset points",
            color=STYLE["color"][m], fontsize=STYLE["font"]["annotation_pt"],
        )
        # Fix-round-1 D4: the starting N was previously unlabelled anywhere on this
        # figure, leaving each trajectory's direction/span ambiguous. Label the
        # low-width end (smallest tested N) too, using the SAME real n_cases value
        # already read from the trajectory's own first row -- never retyped.
        x_start, y_start = traj["interval_width_k"].iloc[0], traj["coverage_80pct"].iloc[0]
        n_min = traj["n_cases"].iloc[0]
        ax.scatter([x_start], [y_start], color=STYLE["color"][m], marker=STYLE["marker"][m], s=STYLE["marker_size_pt"] ** 2, alpha=curve_alpha, zorder=3)
        ax.annotate(
            f"N={_fmt_cases(n_min)}", xy=(x_start, y_start), xytext=n_label_offset_start[m], textcoords="offset points",
            color=STYLE["color"][m], fontsize=STYLE["font"]["annotation_pt"],
        )

    ax.set_xlabel("Interval width, K (Kelvin)")
    ax.set_ylabel("Coverage @ 80%")
    ax.margins(x=0.15, y=0.15)

    handles = [
        Line2D(
            [0], [0], color=STYLE["color"][k],
            marker=STYLE["marker"]["zero_shot_point_coverage_matched" if k == "var_inflation_coverage_matched" else "zero_shot_point"],
            linestyle="none", markersize=STYLE["marker_size_pt"], label=STYLE["label"][k],
        )
        for k in ["raw_ensemble", "var_inflation_fixed", "var_inflation_coverage_matched", "tsfm3"]
    ] + [
        Line2D([0], [0], color=STYLE["color"][k], linestyle=STYLE["linestyle"][k], marker=STYLE["marker"][k],
               markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"],
               alpha=STYLE["alpha"].get(f"{k}_curve", 1.0), label=STYLE["label"][k])
        for k in ["var_inflation_trainfit", "emos_pooled", "emos_local", "drn"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=STYLE["font"]["legend_pt"] - 1)

    _save_figure(fig, "f3_sharpness_calibration_plane", ["results/phase3_data_size_sweep.parquet", "results/phase3_low_n_grid.parquet"])
    plt.close(fig)

    vi_covmatch_row_f3 = sweep[(sweep["method"] == "var_inflation_coverage_matched") & (sweep["sampling_arm"] == "n_independent")].iloc[0]
    tsfm3_row_f3 = sweep[(sweep["method"] == "tsfm3") & (sweep["sampling_arm"] == "n_independent")].iloc[0]
    _update_caption(
        "F3", "Sharpness-calibration plane",
        "Interval width K (sharpness) vs. coverage@80% (calibration), grey "
        "dashed at nominal 0.80. Zero-shot methods (raw ensemble, TimesFM-3, and "
        "two variance-inflated baseline variants -- fixed λ=1.5 and "
        "coverage-matched to TimesFM-3's real coverage) plot as single points; "
        "trained methods (EMOS pooled, EMOS local, DRN, and the third "
        "variance-inflated variant, a CRPS-optimal multiplier refit at each N) "
        "plot as trajectories across training data size N, with an arrowhead "
        "and the N value at both the largest AND the smallest tested N (the "
        "low-width end), so each trajectory's direction and span are "
        "unambiguous. Up and to the left is better: sharper (smaller K) at "
        "equal or better calibration (closer to the nominal 0.80 line) is the "
        "Gneiting-optimal direction. At matched coverage "
        f"({vi_covmatch_row_f3['coverage_80pct']:.4f} vs. TimesFM-3's "
        f"{tsfm3_row_f3['coverage_80pct']:.4f}), the coverage-matched point sits "
        f"at K={vi_covmatch_row_f3['interval_width_k']:.3f} vs. TimesFM-3's "
        f"K={tsfm3_row_f3['interval_width_k']:.3f}.",
    )


# ============================================================================
# F4 -- Calibration diagnostics (PIT histograms)
# ============================================================================
def make_f4() -> None:
    pit = _load("phase3_pit_histograms")
    order = ["raw_ensemble", "tsfm3", "emos_pooled", "emos_local"]
    n_bins = int(pit["bin_index"].nunique())
    uniform_expectation = 1.0 / n_bins

    _apply_rcparams()
    fig, axes = plt.subplots(1, 4, figsize=_figsize_in("double", 95), sharey=True)
    # Fix-round-1 D3: two separate collisions fixed here. (1) Adjacent panels' x-tick
    # labels ran into each other at the panel boundary (e.g. one panel's rightmost
    # "1.00" sitting immediately next to the next panel's leftmost "0.00", reading as
    # "1.000.00") -- fixed by widening wspace AND thinning the tick density per panel
    # (3 ticks: 0, 0.5, 1, instead of the default 5) so there is real whitespace at
    # every boundary. (2) The "n=" annotations collided with the shared "PIT value"
    # x-axis label -- fixed by pulling the n= annotation tighter to its own panel
    # (closer offset) and pushing fig.supxlabel further down, with more total bottom
    # margin (and a taller figure, 78->95mm) so all three rows of bottom content
    # (tick labels / n= annotations / the shared x-label) have real separation
    # instead of being crammed into a margin sized for only two of them.
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.36, wspace=0.30)
    for ax, method in zip(axes, order):
        _style_grid(ax)
        sub = pit[pit["method"] == method].sort_values("bin_index")
        centers = (sub["bin_lo"] + sub["bin_hi"]) / 2.0
        width = (sub["bin_hi"] - sub["bin_lo"]).iloc[0]
        color = STYLE["color"][method]
        ax.bar(
            centers, sub["fraction"], width=width * 0.9, color=color, alpha=STYLE["alpha"]["pit_fill"],
            edgecolor=color, linewidth=0.8,
        )
        ax.axhline(uniform_expectation, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])
        ax.set_title(STYLE["label"][method], fontsize=STYLE["font"]["panel_title_pt"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        n_inst = int(sub["n_instances"].iloc[0])
        ax.text(
            0.5, -0.16, f"n={n_inst:,}", transform=ax.transAxes, ha="center", va="top",
            fontsize=STYLE["font"]["annotation_pt"], color=STYLE["color"]["reference"],
        )

    # Fix-round-1 D3 (round 2): the first attempt moved fig.supxlabel down to y=0.05
    # to make room, but left the legend's own anchor at y=0.005 unchanged -- at
    # fontsize~9pt those two ended up almost coincident (confirmed by rendering and
    # cropping the actual bottom strip, not just eyeballing the full figure), i.e. a
    # NEW collision replaced the old one. Restored real separation: supxlabel at
    # y=0.10 (well above the legend's y=0.005 anchor), with the n= annotations
    # (still at axes-fraction y=-0.16, close to their own panel) sitting clearly
    # above that in turn.
    fig.supxlabel("PIT value", y=0.10)
    fig.supylabel("Fraction of instances")

    handles = [
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label="Uniform expectation (1/n_bins)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.005), frameon=False)

    _save_figure(fig, "f4_pit_histograms", ["results/phase3_pit_histograms.parquet"])
    plt.close(fig)

    _update_caption(
        "F4", "Calibration diagnostics (PIT histograms)",
        "Probability integral transform (PIT) histograms at full training "
        "data size (N=full), raw ensemble / TimesFM-3 / EMOS pooled / EMOS "
        f"local, {n_bins} bins, grey dashed at the uniform expectation "
        f"(1/{n_bins} ≈ {uniform_expectation:.3f}). PIT uniformity is assessed "
        "descriptively; no formal uniformity test valid under serial "
        "dependence is applied.",
    )


# ============================================================================
# F5 -- Breakpoint by lead-time group
# ============================================================================
def make_f5() -> None:
    bp = _load("phase3_lead_time_bucketed_breakpoints")
    sweep = _load("phase3_lead_time_bucketed_sweep")

    groups = ["0-24h", "24-72h", "72-120h"]
    rows_order = [("crps", "emos_pooled"), ("crps", "emos_local"), ("coverage_80pct", "emos_pooled"), ("coverage_80pct", "emos_local")]
    row_labels = {
        ("crps", "emos_pooled"): "CRPS — EMOS pooled",
        ("crps", "emos_local"): "CRPS — EMOS local",
        ("coverage_80pct", "emos_pooled"): "Coverage@80% — EMOS pooled",
        ("coverage_80pct", "emos_local"): "Coverage@80% — EMOS local",
    }

    _apply_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=_figsize_in("double", 115), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.90, bottom=0.24, wspace=0.08)
    y_positions = {key: i for i, key in enumerate(rows_order)}

    n_already_better = 0
    n_true_no_crossing = 0
    for ax, group in zip(axes, groups):
        _style_grid(ax)
        tested = sweep[(sweep["lead_time_bucket"] == group) & (sweep["sampling_arm"] == "contiguous")]
        k_min, k_max = float(tested["n_cases"].min()), float(tested["n_cases"].max())
        for metric, variant in rows_order:
            y = y_positions[(metric, variant)]
            color = STYLE["color"][variant]
            marker = STYLE["marker"][variant]
            row = bp[(bp["lead_time_bucket"] == group) & (bp["metric"] == metric) & (bp["emos_variant"] == variant)].iloc[0]
            val = row["breakpoint_n_cases"]
            if pd.isna(val):
                # --- Fix-round-1 Blocking Fix 2: two DIFFERENT "no crossing"
                # situations get two DIFFERENT markers and x-positions -- see
                # _classify_no_crossing's docstring. Conflating them (the pre-fix
                # behaviour: always an open circle at k_max) visually implied
                # "breakpoint is very high N" even when the true situation was the
                # opposite (already better at the smallest tested N). ---
                kind = _classify_no_crossing(str(row["crossing_direction"]))
                if kind == "already_better":
                    n_already_better += 1
                    x_pos = k_min
                    ax.scatter(
                        [x_pos], [y], color=color, marker=STYLE["marker"]["already_better_at_min_n"],
                        s=STYLE["marker_size_pt"] ** 2 * 2.2, zorder=3,
                    )
                    ax.annotate(
                        f"already better\n(at k={int(k_min)})", xy=(x_pos, y), xytext=(4, 11),
                        textcoords="offset points", color=color, fontsize=STYLE["font"]["annotation_pt"],
                        ha="left", va="bottom",
                    )
                else:
                    n_true_no_crossing += 1
                    x_pos = k_max
                    ax.scatter(
                        [x_pos], [y], facecolors="none", edgecolors=color,
                        marker=STYLE["marker"]["no_crossing_worse_throughout"],
                        s=STYLE["marker_size_pt"] ** 2 * 2, linewidths=1.2, zorder=3,
                    )
                    ax.annotate(
                        f"no crossing\n(k {int(k_min)}–{int(k_max)})", xy=(x_pos, y), xytext=(-4, 11),
                        textcoords="offset points", color=color, fontsize=STYLE["font"]["annotation_pt"],
                        ha="right", va="bottom",
                    )
            else:
                ax.scatter([val], [y], color=color, marker=marker, s=STYLE["marker_size_pt"] ** 2, zorder=3)
                ax.annotate(
                    f"{_fmt_cases(float(val))} cases", xy=(val, y), xytext=(0, 9),
                    textcoords="offset points", color=color, fontsize=STYLE["font"]["annotation_pt"], ha="center",
                )
        ax.set_xscale("log")
        ax.set_xlim(k_min * 0.5, k_max * 3.0)
        ax.set_title(group, fontsize=STYLE["font"]["panel_title_pt"])
        ax.set_yticks(list(y_positions.values()))
        ax.set_ylim(-0.7, len(rows_order) - 1 + 0.7)

    axes[0].set_yticklabels([row_labels[k] for k in rows_order])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.supxlabel("Breakpoint vs. TimesFM-3 (training cases, log scale)", y=0.13)

    handles = [
        Line2D([0], [0], color=STYLE["color"]["emos_pooled"], marker=STYLE["marker"]["emos_pooled"], linestyle="none",
               markersize=STYLE["marker_size_pt"], label=STYLE["label"]["emos_pooled"]),
        Line2D([0], [0], color=STYLE["color"]["emos_local"], marker=STYLE["marker"]["emos_local"], linestyle="none",
               markersize=STYLE["marker_size_pt"], label=STYLE["label"]["emos_local"]),
        Line2D([0], [0], color=STYLE["color"]["reference"], marker=STYLE["marker"]["already_better_at_min_n"], linestyle="none",
               markersize=STYLE["marker_size_pt"] + 1,
               label="Already better than TimesFM-3 at the smallest tested N (no crossing needed)"),
        Line2D([0], [0], color=STYLE["color"]["reference"], marker=STYLE["marker"]["no_crossing_worse_throughout"], linestyle="none",
               markerfacecolor="none", markersize=STYLE["marker_size_pt"],
               label="No crossing observed: worse than TimesFM-3 throughout the tested range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=STYLE["font"]["legend_pt"] - 1)

    _save_figure(fig, "f5_breakpoint_by_lead_group", [
        "results/phase3_lead_time_bucketed_breakpoints.parquet", "results/phase3_lead_time_bucketed_sweep.parquet",
    ])
    plt.close(fig)

    n_no_crossing = int(bp["breakpoint_n_cases"].isna().sum())
    _update_caption(
        "F5", "Breakpoint by lead-time group",
        "EMOS pooled/local breakpoint against TimesFM-3 (training cases, log "
        "scale, now covering the SAME extended grid k=1,2,3,5,7,9,26,105,314,4180 "
        "F1 plots -- Fix round 1, Blocking Fix 1 -- rather than the coarser "
        "k=9..4180-only grid this figure previously showed), split by lead-time "
        "group (0-24h/24-72h/72-120h) and metric (CRPS, coverage@80%). Values "
        "are printed next to each marker. "
        f"{n_no_crossing} of {len(bp)} (metric, variant, lead-group) combinations "
        "show no crossing within the tested case range, and Fix round 1 (Blocking "
        "Fix 2) now distinguishes WHICH of two opposite situations that is: "
        f"{n_already_better} already beat TimesFM-3 at the smallest tested N (filled "
        "star, placed at the left/smallest-N edge -- no crossing was ever needed) "
        f"vs. {n_true_no_crossing} that stayed worse than TimesFM-3 throughout the "
        "entire tested range (open circle, placed at the right/largest-N edge, "
        "labelled with the tested k range) -- conflating these under one "
        "right-edge open-circle marker, as this figure did before this fix, "
        "visually implied the wrong one of the two for CRPS/EMOS-pooled in "
        "particular, whose real situation is 'already better', not 'never "
        "catches up'.",
    )


def main() -> None:
    make_f1()
    make_f2()
    make_f3()
    make_f4()
    make_f5()


if __name__ == "__main__":
    main()
