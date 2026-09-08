"""One-time backfill (final-review fix round, item 3): `results/phase2_comparison_raw.parquet`
predates `zeropp.eval.results.write_result` (Task 1) -- it is written directly via
`results_df.to_parquet(RESULTS_PATH, index=False)` in `scripts/02_run_tsfm.py` -- and was
never given a provenance sidecar. Every other results/ file has one; this one, the file
every later Phase 3 number ultimately derives from, did not.

write_result() requires the actual DataFrame (it writes {name}.parquet itself, unmodified,
alongside {name}.json) -- there is no metadata-only mode. This script reads the existing
parquet and passes it straight through write_result with the SAME name, so the parquet's
bytes are re-written (pandas/pyarrow-serialized, expected to be content-equivalent -- not
byte-identical to the original file, since it goes through a fresh to_parquet call) but a
real .json sidecar is produced for the first time.

Run once, on altay, after syncing this fix round's code:
    python scripts/_backfill_phase2_sidecar.py
Not part of the regular scripts/01-08 pipeline -- intentionally unnumbered so it does not
look like a repeatable step in the normal run order.
"""
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.results import write_result

RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels
    df = pd.read_parquet(RAW_RESULTS_PATH)
    print(f"Backfilling provenance sidecar for {RAW_RESULTS_PATH}: {df.shape[0]} rows, "
          f"methods={sorted(df['method'].unique().tolist()) if 'method' in df.columns else 'no method column'}")

    write_result(
        df,
        name="phase2_comparison_raw",
        model_version="phase2-tsfm3-v1",
        config={
            # What's known about how this file was produced (scripts/02_run_tsfm.py):
            # zero-shot TimesFM-3 vs. raw ensemble vs. EMOS, per-instance quantile
            # predictions on the real Germany t2m EUPPBench test set. This config
            # dict is a RECONSTRUCTION for backfill purposes -- it describes the
            # known production parameters, it is not literally the config object
            # 02_run_tsfm.py held in memory when the file was originally written
            # (that run predates write_result and was never captured).
            "backfill_note": (
                "One-time provenance backfill (final-review fix round, item 3). This "
                "file predates write_result and was originally written directly via "
                "results_df.to_parquet(...) in scripts/02_run_tsfm.py. git_sha/"
                "source_tree_sha256 below reflect the state of the repo AT BACKFILL "
                "TIME, not the (unrecorded, now unrecoverable) state when the original "
                "per-instance predictions were actually computed."
            ),
            "produced_by": "scripts/02_run_tsfm.py",
            "methods": ["raw_ensemble", "emos", "tsfm3"],
            "quantile_levels": quantile_levels,
            "context_length": 40,
            "coverage_band": "q0.1-q0.9 (80% nominal)",
        },
    )
    print("Backfill complete: results/phase2_comparison_raw.json written.")


if __name__ == "__main__":
    main()
