"""Step 5: the analysis. Descriptive core + propensity-matching robustness.

Reads the frame, the sample, and the collected flags; writes result tables to
results/ (committed; all are safe aggregates). All outputs are associational by design.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ksai import analysis, config, matching

logger = logging.getLogger("run_05")

RESULTS = config.RESULTS


def load_merged() -> pd.DataFrame:
    sample = pd.read_parquet(config.DERIVED / "sample.parquet")
    flags = pd.read_json(config.DERIVED / "ai_flags.jsonl", lines=True)
    flags = flags.dropna(subset=["ai_disclosed"])[["id", "ai_disclosed"]].drop_duplicates("id")
    merged = sample.merge(flags, on="id", how="inner")
    merged["ai_disclosed"] = merged["ai_disclosed"].astype(int)
    merged["log_goal"] = np.log1p(merged["goal_usd"])
    logger.info(
        "merged %d projects with flags (%d AI-disclosing)",
        len(merged),
        int(merged["ai_disclosed"].sum()),
    )
    return merged


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_merged()

    # 1) prevalence: overall, by category, by quarter
    analysis.prevalence_by(df, "category").to_csv(
        RESULTS / "prevalence_by_category.csv", index=False
    )
    analysis.prevalence_by(df, "launch_quarter").to_csv(
        RESULTS / "prevalence_by_quarter.csv", index=False
    )

    # 2) outcome comparison, overall (descriptive core)
    comp = pd.DataFrame([vars(c) for c in analysis.compare_outcomes(df)])
    comp["bh_reject_05"] = analysis.benjamini_hochberg(comp["p_value"].tolist())
    comp.to_csv(RESULTS / "outcomes_overall.csv", index=False)

    # 3) inclusion layer: primary lens region, secondary lens experience
    analysis.inclusion_split(df, "english_majority").to_csv(
        RESULTS / "inclusion_region.csv", index=False
    )
    analysis.inclusion_split(df, "first_time").to_csv(
        RESULTS / "inclusion_experience.csv", index=False
    )

    # 4) robustness: propensity-matched comparison
    ps = matching.estimate_propensity(df)
    result = matching.match(df, ps)
    result.balance.to_csv(RESULTS / "matching_balance.csv", index=False)
    if len(result.matched):
        matched_comp = pd.DataFrame([vars(c) for c in analysis.compare_outcomes(result.matched)])
        matched_comp.to_csv(RESULTS / "outcomes_matched.csv", index=False)
        logger.info(
            "matched comparison on %d pairs (caliper %.4f); balance: %s",
            result.n_matched,
            result.caliper,
            "OK" if result.balance["balanced_after"].all() else "CHECK SMDs",
        )

    logger.info("results written to %s", RESULTS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
