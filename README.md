# Analysis code for "Zero-shot time series foundation models for statistical postprocessing of weather forecasts: how much training data do they replace?"

This repository contains the Python code that reproduces the tables and figures of the paper

> Avcı, Ö. F. and Tan, E.: Zero-shot time series foundation models for
> statistical postprocessing of weather forecasts: how much training data do
> they replace?, submitted to *Adv. Stat. Clim. Meteorol. Oceanogr.*, 2026.

The aggregate result files behind every table and figure are included, so the
tables and figures can be regenerated without the raw data or a GPU.

## Authors
- Ömer Faruk Avcı — Department of Climate Science and Meteorological Engineering, İstanbul Technical University, Istanbul, Türkiye — avcio20@itu.edu.tr
- Elçin Tan — Department of Climate Science and Meteorological Engineering, Climate Change Application and Research Center (UYGAR) and ITU ARI Teknokent, İstanbul Technical University, Istanbul, Türkiye

## Summary
The study asks how much station-specific training data the pretrained time series
foundation models TimesFM-3 and Chronos-2, used zero-shot with the ensemble mean as
covariate, replace in the statistical postprocessing of ensemble forecasts. They are
compared with ensemble model output statistics (EMOS) in pooled, local and
lead-specific variants, a distributional regression network (DRN) and variance
inflation, trained on (i) subsets of the reforecast archive, year-round and
season-matched, (ii) rolling windows of the station's own recent forecasts and (iii)
the archives of the other stations only (leave-one-station-out, LOSO). The setting is
probabilistic 2 m temperature forecasts for lead times of 0–120 h at 49 German
stations of the EUPPBench benchmark. Methods are compared with the CRPS, 80 %
coverage, interval width and rank histograms, and with Diebold–Mariano tests with a
HAC variance on daily mean score differences.

## Contents
Scripts are run from the repository root and read `data/raw/` and `results/`.

### (a) Zero-shot forecasts (GPU)

| Script | Produces | Description |
|--------|----------|-------------|
| `scripts/run_timesfm.py` | `results/predictions_main`, `timesfm_context_*`, `timesfm_nocov_context_*`, `gap24_tsfm3` | TimesFM-3 forecasts with the ensemble mean as covariate, together with raw ensemble and pooled EMOS on the same instances; options for the context length (`--context-length`), no covariate (`--no-covariate`) and withheld recent observations (`--context-gap-steps 4`) |
| `scripts/run_chronos2.py` | `results/chronos2_predictions`, `chronos2_context_*`, `gap24_chronos2` | Chronos-2 forecasts on the same instances (same options for context length and gap) |
| `scripts/run_timesfm_spread_covariate.py` | `results/spread_covariate` | TimesFM-3 with the ensemble spread as a second covariate, compared with the mean-only run |
| `scripts/run_lead_matched.py` | `results/lead_matched_*` | Zero-shot runs on lead-matched daily series (one series per station and lead) |

The output file names of the runs are set with `--output` (`--out` for Chronos-2), e.g. `timesfm_context_160` or `gap24_tsfm3`.

### (b) Trained postprocessing and evaluation (CPU)

| Script | Produces | Description |
|--------|----------|-------------|
| `scripts/run_baselines.py` | printed scores | Raw ensemble and pooled EMOS on the test set (quick check) |
| `scripts/refit_emos.py` | `results/predictions_main` (EMOS rows) | Refits the pooled EMOS rows of the prediction file on the training data without overlap with the test period |
| `scripts/training_size.py` | `results/training_size*`, Fig. 3, Table A1 | EMOS (pooled, local), DRN and variance inflation fit on archive subsets of increasing size (contiguous and random cases); crossing points with TimesFM-3 and station-blocked tests |
| `scripts/training_size_drn_small.py` | `results/training_size_small` | DRN at the smallest training sizes |
| `scripts/seasonal_training.py` | `results/seasonal_*`, Fig. 4, Tables 2, 7, A2 | Season-matched training: EMOS fit on k same-season cases, scored on the season's test instances, with significance tests |
| `scripts/emos_variants.py` | `results/emos_variants`, `rolling_emos`, `lead_time_curves`; Figs. 2, 3, 5, 6 | Lead-specific EMOS, rolling EMOS on the forecasts of the preceding W days, and LOSO EMOS; Diebold–Mariano tests against both zero-shot models (`--parts sweep,loso,rolling_all`) |
| `scripts/reference_scores.py` | `results/reference_scores`; Tables 4–6, Figs. 2, 3, 6 | Scores of all zero-shot runs and the raw ensemble on the matched and the common instance set |
| `scripts/context_length_summary.py` | `results/context_length_summary` | TimesFM-3 at each context length on the common instance set |
| `scripts/short_context_summary.py` | `results/short_context_summary`; Fig. 2, Table 6 | Zero-shot runs with short contexts (with and without covariate) |
| `scripts/lead_matched_summary.py` | `results/lead_matched_summary` | Lead-matched runs against the 6-hourly runs |
| `scripts/gap_control_summary.py` | `results/gap_control_summary` | CRPS by lead with and without a 24 h observation gap |
| `scripts/error_metrics.py` | `results/error_metrics`, `station_metadata`; Table 5 | Bias, MAE, RMSE of the median and probabilistic scores of all methods, by season |
| `scripts/error_metrics_extra.py` | `results/error_metrics`; Table 5 | The same for local lead-specific EMOS and the single-rule rolling EMOS |
| `scripts/lambda_curve.py` | `results/lambda_curve`; Fig. 6 | Coverage and width of the ensemble rescaled by a multiplier |
| `scripts/rank_histograms.py` | `results/rank_histograms`; Fig. 7 | Rank histograms of the nine predicted quantiles |

### (c) Figures and tables

| Script | Produces | Description |
|--------|----------|-------------|
| `scripts/fig01_study_area.py` | Fig. 1 | Map of the stations (needs cartopy and Natural Earth boundaries) |
| `scripts/make_figures.py` | Figs. 2–7 | Fig. 2 recent data (context length vs rolling window), Fig. 3 training size, Fig. 4 seasonal training, Fig. 5 lead time, Fig. 6 calibration–sharpness, Fig. 7 rank histograms; written to `figures/` as PDF and SVG |
| `scripts/make_tables.py` | Tables 2, 4–7, A1–A2, `numbers.tex` | Table 2 data by season, Table 4 scores, Table 5 errors, Table 6 recent data, Table 7 seasonal, Tables A1–A2 training size; written to `output/` as `table02_data_by_season.tex`, `table04_scores.tex`, `table05_errors.tex`, `table06_recent_data.tex`, `table07_seasonal.tex`, `tableA1_training_size.tex`, `tableA2_seasonal_training_size.tex`, plus `numbers.tex` with the macros for the numbers in the text |

The library code is in `src/zeropp/` (data loading, models, scores, tests of significance),
settings in `configs/`, and unit tests in `tests/`
(`pytest -q tests --ignore=tests/test_tsfm_timesfm.py --ignore=tests/test_tsfm_timesfm_mean_only_regression.py`;
the two ignored files need the TimesFM-3 package and weights).

### Precomputed results (`results/`)
The aggregate results used for all tables and figures are stored as parquet files, each
with a JSON sidecar that records the model version, configuration and hash, and the code
version that wrote it. With them, `python scripts/make_tables.py` and
`python scripts/make_figures.py` run without the data. The large per-instance prediction
files written by the scripts in (a) (`predictions_main`, `chronos2_predictions`,
`timesfm_context_*`, `timesfm_nocov_context_*`, `chronos2_context_*`, `lead_matched_*`,
`gap24_*`) are not included.

## Requirements
Python 3.11 and the packages in `requirements.txt` (pinned versions):
numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, xarray 2026.9.0, netCDF4 1.7.4, pyarrow 26.0.0,
matplotlib 3.11.2, cartopy 0.26.0 (Fig. 1 only), torch 2.14.1 (DRN) and pyyaml 6.0.3.
The zero-shot forecasts additionally need a GPU, `timesfm` 3.0.1 (checkpoint
`google/timesfm-3.0-pytorch`) and `chronos-forecasting` 2.3.2 (checkpoint `amazon/chronos-2`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install timesfm==3.0.1 chronos-forecasting==2.3.2   # only for the scripts in (a)
export PYTHONPATH=src
```

## Data availability
The scripts read EUPPBench v1.0 (Demaeyer et al., 2023, https://doi.org/10.5281/zenodo.7429236),
German subset, 2 m temperature. The four files are expected in `data/raw/`:

```
data/raw/germany_ensemble_reforecasts_t2m.nc
data/raw/germany_reforecasts_observations_t2m.nc
data/raw/germany_ensemble_forecasts_t2m.nc
data/raw/germany_forecasts_observations_t2m.nc
```

## Citation
Avcı, Ö. F. and Tan, E.: Zero-shot time series foundation models for statistical
postprocessing of weather forecasts: how much training data do they replace?, submitted to
*Adv. Stat. Clim. Meteorol. Oceanogr.*, 2026.

Code archive: Zenodo, DOI to be added (see `CITATION.cff`).

## License
The code is released under the MIT License (see `LICENSE`). The model weights and the data
have their own terms: TimesFM-3 is under the TimesFM non-commercial licence, Chronos-2 under
Apache-2.0, and the EUPPBench data under the terms of its Zenodo record.
