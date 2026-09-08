# ZeroPP reference coverage

Source: 14 Consensus searches run 2026-09-05, 193 unique DOIs harvested,
triaged down to the selection in `selected_dois.tsv`.

**Rule: nothing enters `refs.bib` except via `scripts/build_refs.sh`, which
fetches canonical BibTeX from doi.org. No hand-written entries. No entries
recalled from model memory.**

## Block status

**Updated 2026-09-05** after resolving `gap_fills.tsv` (14 targeted-search entries, 13 confirmed + added, 1 held for user confirmation — see below): 98/98 DOIs in `selected_dois.tsv` now resolve cleanly, `check_references.py` reports 0 errors / 1 warning against the real `refs.bib`.

| Block | Topic | Selected | Target | Status |
|---|---|---|---|---|
| B1 | Statistical postprocessing foundations | 9 | ~8 | **complete** (Vannitsem 2020/2021 review added) |
| B2 | ML-based postprocessing | 14 | ~12 | **complete** (Muschinski 2023 MOS-RF added) |
| B3 | EUPPBench and work on it | 6 | ~6 | complete |
| B4 | Time series foundation models | 10 | ~10 | near-complete, 1 gap (TimesFM 3.0 technical report/model card — not a Crossref/arXiv item, cite directly) |
| B5 | TSFM benchmarking and contamination | 8 | ~6 | complete |
| B6 | TSFMs applied in geoscience | 6 | ~6 | **complete — no longer blocking** (all 6 gap-fill entries added: Sun 2026 streamflow, Bharadwaj 2026 Air Quality Arena, Huang 2026 wildfire PM2.5, Rollo 2026 particulate matter, Gao 2026 zero-inflation wrapper, Liu 2025 RNN-to-transformer hydrology) |
| B7 | Covariate-aware zero-shot forecasting | 7 | ~6 | complete |
| B8 | Probabilistic evaluation theory | 15 | ~9 | **complete** (Hamill 2001 rank histograms, Dirkson 2026 misdiagnosing reliability, Brocker 2018/2020 serial-dependence pair all added) |
| B8b | Forecast comparison / significance | 7 | ~4 | **complete** (`luger_2004_exact_tests` resolved and added — see below; Diebold 1995 corrected from a wrong auto-match, see bugs list) |
| B9 | AI weather prediction models | 9 | ~7 | complete |
| B10 | Downscaling and multivariate alternatives | 8 | ~6 | complete |
| | **Total** | **99** | **~78** | trim LOW priority if over budget |

## `luger_2021_exact_tests` — RESOLVED 2026-09-05, added as `luger_2004_exact_tests`

Time-boxed (~10 minute) follow-up search: checked Richard Luger's own faculty publication page (FSA ULaval) for any later journal appearance of this exact title — found none. The page lists it only as a series of seminar presentations (2003-2004), never as a peer-reviewed journal article. Combined with the earlier finding (no journal match on Crossref or general web search), this confirms the paper was never published beyond the 2004 Bank of Canada Staff Working Paper. Per instruction ("if not resolved in 10 minutes, leave it in working-paper form, do not invent a journal citation"): added to `selected_dois.tsv` as `luger_2004_exact_tests` (bibkey year corrected from the originally-guessed 2021 to the real 2004, cited by its institutional working-paper series — Bank of Canada Staff Working Paper 04-2 — not as a fabricated journal article), DOI `10.34989/swp-2004-2`.

**Two entries resolved with a real, disclosed year discrepancy that IS just an online-first/print-issue convention (added as-is, not held):** `vannitsem_2020_postprocessing_review` requested as "2020" resolves to a BAMS article with Crossref `issued` date 2021-03 (title match 1.00, single unambiguous record — the 2020 figure was the online-early year); `dirkson_2025_misdiagnosing_reliability` requested as "2025" resolves to a QJRMS article with Crossref `issued` date 2026-03-27 (title match 1.00, single unambiguous record). Both kept their original bibkey year suffix for stability; the label field notes the real publication year.

## Gaps that must be filled before drafting

**All resolved 2026-09-05 except one** (via `gap_fills.tsv` + `scripts/resolve_dois.py`, every row manually verified — see "DOIs needing manual repair" and "Held for user confirmation" above/below).

| Block | Missing work | Status |
|---|---|---|
| B1 | Vannitsem et al., BAMS | **Resolved** — added as `vannitsem_2020_postprocessing_review` |
| B2 | Muschinski et al. 2023, NPG | **Resolved** — added as `muschinski_2023_mos_random_forests` |
| B4 | TimesFM 3.0 technical report / model card | **Still open** — not a Crossref/arXiv-indexable item (Google Research release + HF model card); cite directly by URL when drafting, not through this DOI pipeline |
| B6 | Sun & Sun 2026, Machine Learning: Earth | **Resolved** — added as `sun_2026_streamflow_zeroshot` |
| B6 | Air Quality Arena 2026 | **Resolved** — added as `bharadwaj_2026_air_quality_arena` (arXiv 2607.19381) |
| B6 | Wildfire PM2.5 generalizability 2026 | **Resolved** — added as `huang_2026_wildfire_pm25` (arXiv 2607.07951) |
| B6 | Rollo et al. 2026, Expert Syst. Appl. | **Resolved** — added as `rollo_2026_particulate_foundation` |
| B6 | Gao et al. 2026, WWW | **Resolved** — added as `gao_2026_zero_inflation_wrapper` |
| B8 | Hamill 2001, MWR | **Resolved** — DOI was truncated in the harvest, full DOI recovered via Crossref |
| B8b | Diebold & Mariano 1995, JBES | **Resolved, with a real correction** — `resolve_dois.py`'s title search picked the wrong record (the 1994 NBER working-paper preprint, same title, different DOI/venue); the actual 1995 JBES article DOI was found via a container-title-filtered Crossref search and substituted before adding to `selected_dois.tsv` |

## DOIs needing manual repair — RESOLVED 2026-09-05

| bibkey | Problem | Resolution |
|---|---|---|
| `hersbach_2000_crps` | DOI recorded as `10.1175/1520-0434(2000)015`, truncated. Legacy AMS DOIs carry an article-ID suffix. | Resolved via Crossref bibliographic search: full DOI is `10.1175/1520-0434(2000)015<0559:dotcrp>2.0.co;2`. `selected_dois.tsv` corrected, re-fetched — title/journal/author confirmed exact match ("Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems", *Weather and Forecasting*, 2000). |
| `coroneo_2024_strong_dependence` | DOI ends `...2024.11.00`, looks cut off. | Resolved via Crossref: full DOI is `10.1016/j.ijforecast.2024.11.003`. `selected_dois.tsv` corrected, re-fetched — confirmed "Testing for equal predictive accuracy with strong dependence", *International Journal of Forecasting*. |
| `ansari_2025_chronos2` | arXiv 2510.15821 assumed to be Chronos-2. | Confirmed: fetched title is exactly "Chronos-2: From Univariate to Universal Forecasting". No DOI change needed. |
| `mulayim_2025_can_tsfm` | Title truncated in harvest. | Fetched title is complete: "Can time-series foundation models perform building energy management tasks?". No DOI change needed. |
| `kreusel_2026_covariates_key` | Title truncated. | Fetched title is complete: "Covariates Are the Key to Accurate Probabilistic Building Energy Forecasting with Time Series Foundation Models". No DOI change needed. |

All 85 DOIs in `selected_dois.tsv` now resolve cleanly via `scripts/build_refs.sh` (`docs/references/failed_dois.txt` is empty).

**Bugs found and fixed in `scripts/build_refs.sh` while resolving the above** (none of these were pre-existing coverage.md items — found during this pass):
1. The publisher-key-to-bibkey normalisation used a GNU-sed-only construct (`0,/regex/` range address + empty-pattern `s//repl/` reuse) that silently no-ops on BSD/macOS sed — every fetched entry kept its raw publisher key instead of our normalised `\cite{}` key. Rewritten in `awk`, portable.
2. Calling the script with zero block-filter arguments (the documented "fetch everything" mode) silently fetched nothing, because the filter-check function's `$# -eq 0` test was checking the function's own argument count, which is never zero once the block name is prepended — not the script's original argument count. Fixed by capturing the script's real argument count in a variable before the loop.
3. **Most serious**: the "did this resolve?" check only verified that some line, anywhere in the response, started with `@` — a doi.org "DOI Not Found" HTML error page happens to contain such a line, so both `hersbach_2000_crps` and `coroneo_2024_strong_dependence`'s truncated DOIs were silently accepted as "successful" fetches, injecting ~230 lines of raw HTML into `refs.bib` under those bibkeys instead of failing loudly. Fixed with a much stricter check: HTTP status must be 200, the body must not contain `<!doctype`/`<html` anywhere, and the first non-blank character of the body must be `@`.

**Bug found and fixed in `scripts/check_references.py` while running it against the real `refs.bib`:** its BibTeX field parser (a single regex anchored per-line) only correctly parsed a multi-line, one-field-per-line layout. doi.org's actual Crossref output is single-line, every-field-comma-separated-on-one-line — a materially different real-world shape the parser had never been run against. Against the real file this silently swallowed every field of 59 of 85 entries into one bogus `title` value, including the `doi` field itself, so `check_references.py` reported 59 false "no DOI or arXiv id" errors on entries that genuinely have DOIs. Replaced with a brace-depth-aware comma splitter (`_split_top_level`/`_parse_fields`) that handles both layouts identically; added a regression test (`test_compact_single_line_crossref_style_entry_parses_correctly` in `tests/test_references.py`) using this project's real Demaeyer-2023 entry verbatim as the fixture. Real run against `refs.bib` after the fix: **0 errors, 1 warning** (the expected "no manuscript files found" — `paper/` doesn't exist yet).

## Findings in the harvest that bear on our own results

Flagged because they affect claims in the paper, not just the bibliography.

**Four papers actually read and assessed against our real code/results, 2026-09-05** (per explicit request — not just added to the bibliography, their arguments checked against what we actually compute):

**`dirkson_2025_misdiagnosing_reliability` (Dirkson, QJRMS 2026) — partially applies; mitigation available and cheap.**
Core mechanism (verified via the paper's abstract/arXiv 2512.02160): spread-error equality and flat rank histograms are only NECESSARY, not SUFFICIENT, conditions for reliability — a "climatological variance bias" in the ensemble members' covariance structure can make both diagnostics look perfect while the ensemble is not actually reliable up to second order. Our entire narrative rests on coverage/calibration diagnostics (raw ensemble severely under-dispersed, TimesFM-3 moderate, EMOS near-nominal), so this is a real, relevant critique to check against.

**Revised 2026-09-05 after re-reading the code against Dirkson's exact mechanism — the earlier "mitigation" claim below was too optimistic and is corrected here.** `reliability_index` (`zeropp/eval/calibration.py`) is `sqrt(mean((pit_histogram - 1/n_bins)**2))` — a root-mean-square deviation of the full 10-bin PIT/rank histogram from uniform. This is **not** a CRPS-decomposition term (it has no mathematical relationship to the CRPS score at all, computed purely from `pit_values`) — so in the user's own branching, this is case (b), an independent measure, not case (a). But being independent of CRPS's own reliability decomposition does NOT make it independent of Dirkson's critique: `reliability_index` is still built from the rank/PIT histogram itself, exactly the object Dirkson's paper shows can be made to look uniformly flat by a "climatological variance bias" covariance-structure pathology that does not reflect true reliability. Checking all 10 bins with an RMS statistic instead of one threshold's coverage proportion is a real, useful improvement — a pathology would need to fool the histogram's shape across its *entire* range, not just near one cutoff, which is a strictly harder (if not theoretically impossible) thing to engineer by accident — but it provides no formal immunity from the mechanism Dirkson actually demonstrates, since their own paper's point is precisely that a suitably-chosen covariance structure CAN produce a uniformly flat-looking rank histogram end to end while the ensemble is not truly reliable.

**Honest limitation for the paper (per direct instruction: write the real limitation, don't give false reassurance):** "PIT/rank-histogram-based calibration diagnostics (both `coverage@80%` and the full-histogram `reliability_index`) cannot rule out the kind of climatological-variance-bias pathology Dirkson et al. (2026) show can produce falsely flat rank histograms; our calibration comparisons across methods should be read as comparisons of this specific diagnostic family's output, not as a fully sufficient proof of true second-order reliability." This is a real, disclosed, field-wide limitation of rank/PIT-based calibration claims, not something specific to a bug in our pipeline — state it plainly rather than implying `reliability_index`'s multi-bin design solves the problem.

**`luger_2004_exact_tests` (real year 2004, not 2021 as originally guessed — see "Held for user confirmation" above) — a real, unaddressed vulnerability; concrete fix identified, not yet implemented.**
Confirmed by direct code inspection (`src/zeropp/eval/significance.py`): our station-blocked bootstrap/paired test aggregates per-instance losses to one mean per STATION, which correctly absorbs *temporal* autocorrelation within a station (repeated issue times, correlated lead times) — but does nothing about *spatial* dependence ACROSS the ~49 German stations, which the paper's framework (exact tests valid under contemporaneous correlation) is specifically built to handle. The same synoptic weather system affects many/most German stations simultaneously, so the ~49 station-block means are not independent draws — the effective sample size is smaller than 49, and both our block-bootstrap CI and our paired t-test/Wilcoxon p-values are very likely too optimistic (CIs too narrow, p-values too small) as a result. This directly threatens the smallest, most fragile finding in the project: the 0.0065 coverage-gap significance test at k=9.

**The fix is genuinely cheap given the current code**: `block_bootstrap_skill_score_ci` and `station_blocked_paired_test` both take a generic `block_ids` array — nothing in either function is hardcoded to "station." Every current caller happens to pass `station_id` as `block_ids`; the same functions can be called again passing a date/synoptic-period identifier (e.g. `valid_time`'s calendar date) instead, with zero changes to `significance.py` itself. Preferred plan (matches the user's stated preference): add a day-blocked re-analysis alongside the existing station-blocked one for the project's key significance claims (Task 4's k=9 coverage gap, Task 6's E2 CRPS/coverage gaps, Task 3's lead-time-crossover significance) and report both. If they agree closely, the station-blocked result is robust to this critique after all (spatial dependence wasn't actually inflating the apparent significance much in practice); if they diverge, the day-blocked (more conservative) number is the one that should be quoted in the paper, with the divergence itself reported as a real finding about the danger of ignoring cross-station synoptic dependence. **Not yet implemented — flagged as the next concrete task**, since it requires a real (cheap, CPU-only) re-run against the already-persisted `results/*.parquet` files on `altay`, not a documentation-only fix.

**`broecker_2018_serial_dependence` + `broecker_2020_stratified_rank` (Bröcker, QJRMS 2018/2020) — checked against our actual PIT code; does NOT apply, no code or framing change needed.**
Both papers' critique targets a SPECIFIC failure mode: classical rank-histogram-flatness goodness-of-fit TESTS (e.g. Pearson's chi-squared) assume independent ranks and give invalid (over-confident) p-values when ranks are serially dependent, which ours genuinely are. Checked our actual code (`zeropp/eval/calibration.py`): `pit_histogram` and `reliability_index` are purely DESCRIPTIVE — a bin-count histogram and a root-mean-square deviation from uniform, respectively. Neither computes a hypothesis test, a p-value, or a "statistically flat" claim anywhere in this project. We only ever compare these descriptive numbers ACROSS methods (e.g. "EMOS-local's reliability_index is lower than raw ensemble's"), never claim "method X's PIT is significantly/perfectly flat at p<0.05" — the exact claim type Bröcker's papers show is invalid under serial dependence. Since we never make that claim, the critique's specific failure mode doesn't apply to anything we currently report. Still worth citing as evidence we're aware of the literature on this exact point (and as a caveat: if a future PIT hypothesis test is ever added, it must use Bröcker's serial-dependence-corrected version, not a naive chi-squared test).

**Paper limitations-section sentence (T3.3, exact wording as specified):** "PIT uniformity has been assessed descriptively; no formally valid uniformity test under serial dependence has been applied."

**`zhang_2025_context_parroting` — read before finalising the framing.**
"Context parroting: a simple but tough-to-beat baseline for foundation models
in scientific machine learning." This is the same argument as our
variance-inflation baseline: a trivial procedure competitive with a large
pretrained model. It is simultaneously a threat (the general point is already
published) and an asset (it gives our specific instance a home in the
literature). Our contribution is the postprocessing-specific instance with a
calibration axis, not the general observation.

**`kreusel_2026_covariates_key` — supports our covariate-wiring finding.**
Independent evidence that covariate handling dominates TSFM performance,
which is what our misconfigured first run demonstrated the hard way.

**`li_2026_tsfmaudit` and `pan_2026_contamination_electricity` — cite for the
contamination argument.** Our claim that station weather data is unlikely to
sit in TimesFM's pretraining corpus needs external grounding, not just
assertion.

**`coroneo_2020_small_samples` — reviewed against our actual significance test 2026-09-05; NOT a vulnerability, but a naming clarification is needed.**
The paper's abstract (Coroneo & Iacone 2020, *J. Applied Econometrics* 35(4),
391-405): standard Diebold-Mariano's HAC-type long-run-variance estimator is
size-distorted under "increasing-smoothing" asymptotics when there are few
out-of-sample observations in a *single autocorrelated loss-differential
series*; their fix ("fixed-smoothing asymptotics") holds the smoothing
bandwidth fixed as a fraction of sample size, giving a correctly-sized test
statistic even at small T.

Our `zeropp.eval.significance.station_blocked_paired_test` does not use this
class of estimator at all: it collapses the ~737,809 autocorrelated
station×time×lead instances to ONE mean per station block *first*, then runs
a plain paired t-test and Wilcoxon signed-rank test on those ~49
(approximately independent) block means. A paired t-test's null distribution
is exact under normality for any sample size -- it never relies on HAC/
long-run-variance asymptotics, so it is not exposed to the specific failure
mode this paper documents. (This is also exactly why the project's own docs
already refuse to call this test "Diebold-Mariano" -- see `significance.py`'s
docstring.)

One naming correction worth making explicit: "our k=9 comparison" is not the
sample the significance test runs on. k=9 is EMOS's *training*-set size (how
many reforecast cases the model was fit on); the significance test itself
always evaluates the fitted model's test-set performance aggregated to ~49
station means, regardless of k. The already-known, real fragility in our
approach is a different and more mundane one (flagged in Task 3/Task 6's own
findings): a handful of outlier stations can dominate the ~49-point t-test's
variance while the sign-majority still favors Wilcoxon significance,
producing t-vs-Wilcoxon disagreement -- a moderate-N t-test weakness, not
the small-T HAC-asymptotics problem Coroneo & Iacone address. No code change
needed here; reporting both p-values with this caveat (already the project's
practice) remains the right call.

**`gneiting_2007_calibration_sharpness` — the basis of finding 2.**
"Maximise sharpness subject to calibration" is the principle under which
local EMOS dominates TimesFM-3. Currently uncited in our drafts.

**`baran_2026_sharpness_penalty` — closest prior work on our sharpness axis,
and it is on EUPPBench.** Check for overlap before claiming novelty.

**`wessel_2024_leadtime_continuous` and Muschinski 2023 — the small-data
rivals.** Both target the regime our breakpoint curve explores. Neither is
currently a baseline in our code.

## T3.4 — lead-time pooling limitation (closed as a disclosed limitation, no new code)

Already confirmed with real evidence during Phase 3's Task 6 (E4), re-stated here as the final, closed limitation write-up per direct instruction not to reopen it as new work:

- **EMOS pools all 21 lead times into one global fit.** `build_train_ensemble_stats_with_ids` (`src/zeropp/data/build.py`) selects columns `station_id, time_idx, year_idx, ens_mean, ens_var, t2m_obs` — `step` (lead time) is dropped from the column selection even though the underlying rows are genuinely duplicated per lead time (verified via a real test fixture: 2 stations x 2 time x 1 year x 2 step = 8 combinations -> 7 rows after one NaN drop, both lead times present as separate, unlabeled rows). `EMOS.fit()` (`src/zeropp/models/emos.py`) takes these rows as flat arrays with no grouping by lead time, so a single `(a, b, c, d)` parameter set is fit across all 21 lead times at once.
- **DRN inherits the identical limitation** for the identical reason: it trains on the same `build_train_ensemble_stats_with_ids` output, with no lead-time input anywhere in its feature set (`ens_mean`, `ens_var`, `station_id` embedding only).
- **Direction of the effect (both cases): this weakens EMOS and DRN, not TimesFM-3.** A single pooled fit is necessarily too wide at short lead (where the true uncertainty is small) and too narrow at long lead (where it is large) relative to a lead-time-aware fit — so the "EMOS/DRN beat TimesFM-3 by k=9" headline crossings are, if anything, understated, not artefacts of this limitation. It is a genuine limitation for the SHARPNESS comparison specifically, since a lead-blind fit's interval-width behavior is not directly comparable to TimesFM-3's per-instance, per-lead-time-varying output.
- **DRN-specific consequence, must be stated in the paper exactly this way:** without lead-time information, DRN's architectural difference from EMOS (a nonlinear per-station-embedding network vs. a linear model) is the ONLY thing distinguishing it from EMOS in this feature-constrained setup — DRN in this project is not evaluated with its originally-intended full feature set (Rasp & Lerch 2018 use auxiliary NWP fields and lead time directly). **If DRN underperforms, the correct statement is "this architecture, under this feature-constrained setup, was outperformed" — never "DRN failed" or "DRN is a weak baseline," since that would misattribute a feature-set limitation to the architecture itself.**

## D4 (Results review, 04_results.tex) — pruned 26 unlikely-to-be-cited entries

99 -> 73 entries. `check_references.py`'s "never cited" list stood at 65
(64 after fixing a real regex bug below) with only Data/Methods/Results
written; Introduction and Discussion were expected to absorb most of them,
but not all. Rather than defer triage to the end, decided now which ~25
of the ~65 genuinely will not be used, per each entry's block purpose in
`selected_dois.tsv` and this paper's actual scope (univariate per-station
postprocessing — EMOS/DRN/TimesFM-3 — not multivariate/spatial dependence
methods, and not a domain-application survey).

**Also fixed while auditing:** `check_references.py`'s `CITE_RE` regex
didn't match `\citealp`/`\citealt` (only `\citep`/`\citet`), so
`luger_2004_exact_tests` — genuinely cited in `03_methods.tex` via
`\citealp` — was being falsely reported as unused. Fixed to
`\\cite(?:alp|alt|author|year)?[tp]?\*?...`; re-ran before doing the D4
triage so the "unused" list itself was trustworthy.

**Removed (26), by block and reason:**
- B2 (6): `wessel_2024_extreme_wind`, `primo_2024_mos_vs_nn`,
  `bremnes_2019_bernstein`, `bremnes_2019_quantile_splines`,
  `feik_2024_gnn_spatial`, `benacek_2023_decision_tree` — redundant with
  the 4 B2 entries kept (`taillardat_2016_qrf`,
  `muschinski_2023_mos_random_forests` — both CLAUDE.md-mandated
  baselines — plus `hoehlein_2023_permutation_invariant`,
  `groenquist_2020_deep_learning_pp` as the neural-postprocessing
  representatives); a 10-deep citation list for one Introduction sentence
  repeats the D7 citation-dump problem already fixed once in Methods.
- B5 (2): `qiao_2026_its_time`, `xu_2025_fidel_ts` — redundant benchmark-
  methodology papers once `aksu_2024_gift_eval` is kept as the
  representative.
- B6 (4): `huang_2026_wildfire_pm25`, `rollo_2026_particulate_foundation`,
  `gao_2026_zero_inflation_wrapper`, `liu_2025_rnn_to_transformer_hydro`
  — kept `sun_2026_streamflow_zeroshot` (real DOI-correction effort
  already invested, directly analogous "TSFM zero-shot in an adjacent
  forecasting domain" point) and `bharadwaj_2026_air_quality_arena`
  (benchmark-shaped); the other four make the same point redundantly.
- B7 (2): `mulayim_2025_can_tsfm`, `fu_2026_chronos2_building` — same
  domain-application redundancy pattern as B6; kept
  `zhang_2024_chaotic_zeroshot` (companion to the already-central
  `zhang_2025_context_parroting`).
- B8/B8b (3): `broecker_2007_being_proper` (redundant with
  `gneiting_2007_proper_scoring`, already cited), `pohle_2026_uncertainty_comparison`,
  `harvey_2024_heteroskedasticity` — tangential to the actual
  station/day-block testing this paper runs.
- B9 (3): `allen_2025_aardvark`, `kochkov_2024_neuralgcm`,
  `charltonperez_2024_storm` — redundant specific-model examples once
  `lam_2023_graphcast`/`price_2024_gencast`/`bodnar_2025_aurora` (already
  cited in Methods) anchor the TSFM-vs-spatial-model distinction; kept
  `benbouallegue_2024_rise`, `bonavita_2024_limitations`,
  `zhang_2026_physics_outperform` for Introduction's fuller treatment.
- B10 (6): `aich_2024_conditional_diffusion`,
  `lakatos_2022_multivariate_comparison`, `chen_2022_generative_multivariate`,
  `lakatos_2025_gnn_composite`, `schefzik_2013_ecc`,
  `wilks_2015_empirical_copulas` — all multivariate-dependence
  postprocessing methods (ECC, copulas, GNN composite losses); genuinely
  orthogonal to this paper's univariate marginal-distribution scope, real
  scope creep. Kept `molinaro_2026_universal_downscaling` (tagged "KEY
  RIVAL" in `selected_dois.tsv`) and `mardani_2025_corrdiff` (spatial
  diffusion downscaling, complements the B9 distinction).

Removed from both `refs.bib` and `selected_dois.tsv` (so a future
`build_refs.sh` rerun does not silently re-add them). Result: 73 entries,
38 still unused pending Discussion/Introduction, 0 `check_references.py`
errors.

## D16 (Introduction review, 2026-09-07) — das_2023_timesfm renamed to das_2024_timesfm_icml

The manuscript cited only the arXiv preprint (`10.48550/ARXIV.2310.10688`)
for TimesFM's architecture paper, which was actually published at ICML
2024 (Das, Kong, Sen, Zhou, "A decoder-only foundation model for
time-series forecasting", PMLR v235, pp. 10148-10167). Checked for a real
DOI on the published version before updating the entry:

- Crossref bibliographic search (`api.crossref.org/works?query...`): no
  match for this title/author combination under any container-title.
- A candidate ACM DL identifier surfaced by web search
  (`10.5555/3692070.3692474`) does **not** resolve — `doi.org` returns
  "DOI Not Found" (confirmed via direct `curl` with BibTeX content
  negotiation, the same failure mode `build_refs.sh`'s 3-signal validity
  check exists to catch).
- **Conclusion: no resolvable DOI exists for the PMLR proceedings record.**
  Per this project's rule (never invent a DOI, never hand-write bib
  metadata from memory), the entry was updated using PMLR's own official
  BibTeX export (fetched directly from
  `https://proceedings.mlr.press/v235/das24c.html`, not recalled) rather
  than through `build_refs.sh`'s doi.org pipeline — the same kind of
  documented exception already used for the TimesFM 3.0 model card (see
  the B4 row above). The arXiv DOI is kept in the entry's `doi` field
  (real, resolving, same underlying work) with a `note` explaining it is
  the preprint's identifier, not the proceedings record's.
- Citation key renamed `das_2023_timesfm` -> `das_2024_timesfm_icml`
  throughout the manuscript and in `selected_dois.tsv`.

## D24 (Introduction review round 3, 2026-09-07) — 3 keys orphaned by D14's cut

D14 removed Introduction P3's closing sentence (a tangent connecting
spatial-AI-model quality to this paper's own CRPS finding, judged to add
nothing to this paper's argument). That sentence was the only place these
3 B9-block keys were cited:

- `benbouallegue_2024_rise` (Ben-Bouallegue 2024, "rise of data-driven
  weather forecasting")
- `bonavita_2024_limitations` (Bonavita 2024, "limitations of current ML
  weather models")
- `zhang_2026_physics_outperform` (Zhang 2026, "physics-based models
  outperform AI on record extremes")

Deliberately left in `refs.bib` rather than removed: they remain
plausible citations for Introduction's B9 distinction (spatial AI weather
models) or a future Discussion mention, and D4's original audit already
judged them worth keeping over 26 other candidates that were removed
outright. Logged here so their continued "unused" status is understood as
intentional, not overlooked. Current unused-citation count: 19/73
(`check_references.py`, 2026-09-07, 0 errors).
