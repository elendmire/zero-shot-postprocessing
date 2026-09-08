# Phase 2 results/ schema — confirmed 2026-09-03

`results/phase2_comparison_raw.parquet` (2,213,427 rows, produced by `scripts/02_run_tsfm.py`), inspected directly on the `altay` server via `pd.read_parquet`:

- Columns (14 total, in order): `method`, `station_id`, `valid_time`, `step_hours`, `obs`, `q0.1`, `q0.2`, `q0.3`, `q0.4`, `q0.5`, `q0.6`, `q0.7`, `q0.8`, `q0.9`
- Dtypes: `method` str, `station_id` int64, `valid_time` datetime64[ns], `step_hours` float64, `obs` float64, `q0.1`..`q0.9` float64 each
- `method` values: `raw_ensemble` (737,809 rows), `emos` (737,809 rows), `tsfm3` (737,809 rows) — three methods, evenly split, no `timesfm3` string as guessed in the task brief
- How quantile predictions are stored: 9 separate float64 columns named `q0.1` through `q0.9`, one column per level in `configs/experiment.yaml`'s `quantile_levels`, matching the brief's `f"q{level}"` naming guess exactly

Sample rows (raw_ensemble, station 460, first 3 valid_times):
```
        method  station_id          valid_time  step_hours     obs        q0.1        q0.2  ...        q0.9
0  raw_ensemble         460 2017-01-11 00:00:00         0.0  273.75  271.413269  271.648132  ...  273.505829
1  raw_ensemble         460 2017-01-11 06:00:00         6.0  273.85  271.792847  272.001099  ...  273.259094
2  raw_ensemble         460 2017-01-11 12:00:00        12.0  274.95  275.200439  275.400879  ...  276.052246
```

## Consequence for this plan's tasks

The brief's Step 7 guess was almost exactly right on quantile-column naming (`q{level}`), but wrong on two identifier details that the summarization script must use as-written from this inspection, not the brief's placeholder names: the observation column is `obs`, not `t2m_obs`, and the zero-shot TSFM method's string value is `tsfm3`, not `timesfm3`. `scripts/04_summarize_results.py` groups by the real `method` column (`raw_ensemble`, `emos`, `tsfm3`) and reads `obs` for observations and `q{level}` for each of the 9 quantile columns in `configs/experiment.yaml`'s `quantile_levels`; no other structural adaptation was needed since the file is already one row per (method, station, valid_time) instance with the full 9-quantile grid attached.
