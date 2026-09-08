# ZeroPP figure style guide

Derived from the design grammar of Schlechtweg et al. (2026), *City and
Environment Interactions* 31, 100415 (CC BY). We adopt the visual grammar,
not the figures: every panel here is built from our own results.

The reference figures are ggplot2 output. We reproduce the grammar in Python.
Two viable routes, decide once and stay with it:

- **plotnine** (ggplot grammar in Python) — gets faceting, strip labels and
  shared legends almost for free, closest match, extra dependency.
- **matplotlib** with a shared `zeropp_style.mplstyle` and helper functions —
  no new dependency, but faceting and strip labels must be hand-built.

Recommendation: matplotlib. The dependency is not worth it for five figures,
and we need fine control over in-panel annotation anyway.

---

## 1. The seven rules that define the look

Each rule is a concrete, checkable property. These are what make the reference
figures read well; missing any one of them is what makes a figure look like a
working plot instead of a published one.

**R1. Small multiples with strip labels, not separate figures.**
Column titles sit above the top row, row labels sit on the right edge rotated
90 degrees. Axes are shared and labelled once, on the outer edge only. No
repeated axis labels per panel.

**R2. Key numbers are annotated inside the panel, colour-matched to the series.**
The reference prints Gini indices in the top-left of each Lorenz panel, each
value in the colour of its city, stacked. The reader never crosses to the
legend to decode a number. This is the single most transferable device and our
figures need it for breakpoints.

**R3. One shared horizontal legend at the bottom of the figure.**
Never per-panel. Legend title only where the category is not self-evident
(the reference uses "Land cover class" but omits a title for the city legend).

**R4. Reference lines are grey and dashed, never coloured.**
Equality diagonal in the Lorenz plots, nominal levels, zero-skill lines. Grey
dashed says "this is a benchmark, not data".

**R5. Values are printed on the geometry where the count matters.**
The stacked bars carry their percentages inside the segments. Where a reader
would otherwise squint at an axis, print the number.

**R6. Minimal chrome.**
No panel borders, no heavy tick marks, light grey gridlines on white, no
background fill. Density fills are translucent with a thin darker outline so
overlaps stay readable.

**R7. One categorical palette across every figure in the paper.**
In the reference, salmon always means Iași. A reader learns the mapping once.
Note: the reference is inconsistent here (density plots use one palette, the
Lorenz plots another). We fix that rather than copy it.

---

## 2. Palette

The reference uses the ggplot2 default hue palette (approximately `#F8766D`
salmon, `#00BA38` green, `#619CFF` blue) in some figures and a Dark2-like
palette in others.

**We do not copy the default-hue palette.** Salmon and green differ mainly in
hue, so it fails for red-green colour blindness (~8% of male readers) and
collapses in greyscale printing. We use Okabe-Ito, which is colour-blind safe
and prints legibly.

Fixed method-to-colour mapping, used in every figure without exception:

| Method | Hex | Name | Line style | Marker |
|---|---|---|---|---|
| Raw ensemble | `#999999` | grey | dotted | none |
| Variance-inflated baseline | `#E69F00` | orange | dashdot | none |
| TimesFM-3 (zero-shot) | `#0072B2` | blue | solid | none |
| EMOS pooled | `#D55E00` | vermillion | solid | circle |
| EMOS local | `#009E73` | bluish green | solid | triangle |
| DRN | `#CC79A7` | reddish purple | solid | square |
| Reference / nominal | `#666666` | dark grey | dashed | none |

Reserved for later use if needed: `#56B4E9` sky blue, `#F0E442` yellow.

`#0072B2` and `#D55E00` are the two protagonists of the paper and carry the
strongest contrast in the set. That is deliberate.

**Line style is not decorative.** It carries the same information as colour so
the figures survive greyscale printing. Never let colour be the only channel.

Zero-shot methods (raw, inflated, TimesFM-3) render as horizontal lines in any
N-axis figure because they do not depend on N. Trained methods render as
curves with markers. That visual distinction is itself the paper's argument.

---

## 3. Typography and geometry

| Property | Value |
|---|---|
| Base font | sans-serif; Arial or Helvetica |
| Panel title (strip) | 10 pt, medium weight |
| Axis label | 9 pt |
| Tick label | 8 pt |
| In-panel annotation | 8 pt |
| Legend | 9 pt |
| Line width, data | 1.4 pt |
| Line width, reference | 1.0 pt |
| Gridline | 0.5 pt, `#DDDDDD` |
| Marker size | 5 pt |

**Sizing rule:** single-column width is 90 mm, double-column is 190 mm. Set
figure width in the code, never scale afterwards. Render at final size and
read the tick labels before accepting a figure; do not judge a figure zoomed in.

---

## 4. The five main figures

### F1 — Breakpoint curve (cover figure)
Three panels stacked vertically, sharing the x axis: CRPS, coverage@80%,
interval width (K). Log-scaled x axis in cases, with the calendar-day
equivalent on a secondary tick row beneath.

Applies R1 (stacked panels, x labelled once at the bottom), R2 (breakpoint
value annotated inside each panel, colour-matched to the EMOS variant that
crosses), R4 (nominal 0.80 as grey dashed in the coverage panel).

Horizontal lines: raw, inflated baseline, TimesFM-3. Curves: EMOS pooled,
EMOS local, DRN. Random-arm mean as a dashed line of the same colour with a
translucent standard-deviation band.

### F2 — Lead-time resolved comparison
Two panels stacked: CRPS and coverage@80% across 21 lead times. Crossover lead
time marked with a grey dashed vertical line, annotated in-panel with its
value. Probably the strongest figure in the paper, so it gets the most care.

### F3 — Sharpness-calibration plane
Scatter: x = interval width (K), y = coverage@80%. Grey dashed horizontal at
nominal 0.80. Each zero-shot method is a single point; each trained method is
a trajectory across N, drawn with connected markers and an arrow at the
high-N end. Annotate the arrow head with the N value.

This panel makes the Gneiting principle legible at a glance: up and to the
left is better. Say that in the caption.

### F4 — Calibration diagnostics
PIT histograms in a 1x4 row: raw, TimesFM-3, EMOS pooled, EMOS local. Uniform
expectation as a grey dashed horizontal. Bars filled in the method colour at
0.7 alpha with a thin darker outline (R6).

Caption must carry: "PIT uniformity is assessed descriptively; no formal
uniformity test valid under serial dependence is applied."

### F5 — Breakpoint by lead-time group
Three lead groups (0-24h, 24-72h, 72-120h). Breakpoint in cases per group,
per metric. Values printed on the geometry (R5). This is the figure that
carries the nowcasting-regime claim.

### Supplementary
S1 per-station CRPS distribution, S2 random-arm seed variance, S3 contiguous
vs random arm, S4 DRN detail, S5 inflation-factor sensitivity.

---

## 5. Implementation notes

- One module, `scripts/make_figures.py`. One function per figure, each
  independently callable. Notebooks produce nothing that ships.
- Palette and typography live in a single `STYLE` dict at the top. No colour
  literal appears anywhere else in the file.
- Export PDF and SVG (vector). PNG at 300 dpi for preview only.
- Every figure writes its source parquet path and the git SHA into the PDF
  metadata, not onto the canvas.
- Numbers in captions are read from the results files, never retyped.
  Figure-text mismatch is among the most commonly caught reviewer complaints
  and it is entirely avoidable.
- Check the target journal's figure limit in `docs/candidate_journals.md`
  before adding a sixth main figure.

## 6. Attribution

The design grammar is adopted from a CC BY paper; visual conventions are not
themselves copyrightable and no figure is reproduced. No attribution is
required, and none is misleading to omit. If a co-author asks, the reference
is Schlechtweg et al. (2026), doi:10.1016/j.cacint.2026.100415.
