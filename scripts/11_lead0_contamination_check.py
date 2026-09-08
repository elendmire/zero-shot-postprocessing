"""Madde 3 / BL-3 (Introduction review round, 2026-09-07): is step_hours=0
contaminated -- i.e., is the "forecast" at lead 0 actually the analysis,
tightly coupled to (or literally copied from) the observation it is later
verified against, rather than a genuine 0-hour-ahead forecast?

This affects every lead-time-resolved result in the paper that includes
lead 0 (Results Section 4.3, Figures F2/F5, Discussion Section 5.3's
persistence-augmented EMOS test, which uses the 0-24h bucket).

Method: for each of the 21 lead times, compute the per-instance residual
(ensemble-mean forecast minus observation) at (station, valid_time,
step_hours) resolution -- the same aggregation every other script in this
project uses. Two distinct signatures are checked, not conflated:

1. EXACT-MATCH CONTAMINATION (the strong, unambiguous signature): the
   fraction of instances where the residual is exactly zero (or within
   1e-6 K, i.e. floating-point noise) at each lead. If lead 0 shows a
   materially higher exact-match fraction than every other lead, the
   "forecast" there is very likely a literal copy of the observation for
   at least some instances -- direct evidence of contamination, not
   inference from variance alone.
2. DISCONTINUOUS SKILL IMPROVEMENT (the softer, ambiguous signature):
   residual standard deviation naturally shrinks as lead time shortens
   (forecast skill improves closer to the analysis time) -- some
   tightening at lead 0 is EXPECTED and is NOT by itself contamination.
   What would be diagnostic is a discontinuous jump at lead 0 relative to
   the smooth trend leads 6-120 establish: this script fits a trend to
   leads >= 6h and reports how far lead 0's actual std falls below what
   that trend would predict, as a z-score-like ratio, without asserting a
   threshold for "contaminated" -- that judgment call is made when writing
   Results/Discussion, informed by this number, not automated here.

Output: results/phase3_lead0_check.parquet (one row per lead time: n, exact
match fraction, residual mean/std) plus a printed summary highlighting lead
0 specifically against the trend from other leads.
"""
import numpy as np
import pandas as pd

from zeropp.data.splits import load_test
from zeropp.eval.results import write_result

EXACT_MATCH_ATOL = 1e-6


def main() -> None:
    test_df = load_test()

    grouped = test_df.groupby(["station_id", "valid_time", "step_hours"]).agg(
        ens_mean=("t2m_forecast", "mean"),
        obs=("t2m_obs", "first"),
    ).reset_index()
    grouped["residual"] = grouped["ens_mean"] - grouped["obs"]
    grouped["exact_match"] = np.abs(grouped["residual"]) < EXACT_MATCH_ATOL

    by_lead = grouped.groupby("step_hours").agg(
        n_instances=("residual", "size"),
        residual_mean=("residual", "mean"),
        residual_std=("residual", "std"),
        exact_match_fraction=("exact_match", "mean"),
    ).reset_index().sort_values("step_hours")

    print("Per-lead residual summary (ensemble mean vs. observation):")
    print(by_lead.to_string(index=False))

    lead0 = by_lead[by_lead["step_hours"] == 0.0]
    others = by_lead[by_lead["step_hours"] >= 6.0]
    assert not lead0.empty, "no step_hours=0 instances found -- check load_test() output"
    assert not others.empty, "no step_hours>=6 instances found to fit a trend against"

    lead0_exact = float(lead0["exact_match_fraction"].iloc[0])
    max_other_exact = float(others["exact_match_fraction"].max())
    print(
        f"\nEXACT-MATCH CHECK: lead 0 exact-match fraction={lead0_exact:.6f}, "
        f"max over all other leads={max_other_exact:.6f}"
    )
    if lead0_exact > max_other_exact * 10 and lead0_exact > 0.01:
        print(
            "STRONG SIGNAL: lead 0's exact-match fraction is far above every other "
            "lead's -- consistent with literal analysis=observation contamination "
            "for a meaningful share of instances. This is direct evidence, not an "
            "inference from variance."
        )
    else:
        print(
            "No strong exact-match signal: lead 0 does not show a materially higher "
            "rate of exact-zero residuals than other leads."
        )

    # Trend check: fit a simple linear trend of residual_std vs. step_hours
    # over leads >= 6h (the un-contaminated range by construction), then see
    # how far lead 0's actual std falls below that trend's extrapolation to
    # step_hours=0. This is descriptive, not a formal statistical test --
    # reported as a ratio for a human judgment call, not a pass/fail here.
    coeffs = np.polyfit(others["step_hours"], others["residual_std"], deg=1)
    predicted_std_at_0 = float(np.polyval(coeffs, 0.0))
    actual_std_at_0 = float(lead0["residual_std"].iloc[0])
    ratio = actual_std_at_0 / predicted_std_at_0 if predicted_std_at_0 > 0 else float("nan")
    print(
        f"\nTREND CHECK: linear trend (leads>=6h) predicts residual_std={predicted_std_at_0:.4f} K "
        f"at step_hours=0; actual lead-0 residual_std={actual_std_at_0:.4f} K "
        f"(ratio actual/predicted={ratio:.4f}; well below 1 suggests a discontinuous "
        "tightening beyond what the smooth skill-improvement trend alone would predict)."
    )

    write_result(
        by_lead,
        name="phase3_lead0_check",
        model_version="phase3-lead0-check-v1",
        config={
            "exact_match_atol": EXACT_MATCH_ATOL,
            "trend_fit_leads": "step_hours >= 6",
            "lead0_exact_match_fraction": lead0_exact,
            "max_other_exact_match_fraction": max_other_exact,
            "predicted_std_at_lead0_from_trend": predicted_std_at_0,
            "actual_std_at_lead0": actual_std_at_0,
            "actual_over_predicted_ratio": ratio,
        },
    )


if __name__ == "__main__":
    main()
