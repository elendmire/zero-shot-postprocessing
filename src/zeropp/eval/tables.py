def build_summary_tables(results_dir: str, out_dir: str) -> None:
    """Build the CRPS/MAE/pinball/coverage summary tables from model run output.

    BLOCKED: needs real results/*.parquet output from real model runs
    (scripts/01_run_baselines.py, scripts/02_run_tsfm.py) — a table built
    from stub or synthetic output would not reflect real performance.
    """
    raise NotImplementedError(
        "blocked: needs real results/*.parquet output from real model runs "
        "(for score summary tables)"
    )
