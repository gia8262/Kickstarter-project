"""Step 6: dictionary content analysis and descriptive tables/figures.

METHOD: dictionary-based content analysis of the disclosure text under an
assumption of truthful self-report. A fixed list of terms assigns each disclosure
to one or more output-type categories; the category definitions are the
researcher's. Descriptive evidence only.

Reads the collected flags and the sample. Safe aggregate tables and figures go to
results/content/ (committed); the per-project files (extracted_output_types,
verification_sample, verification_human_BLANK) go to data/derived/results/content/
(gitignored, identifiable):
  term_tool_frequency.csv       ranked stated terms and named AI tools
  extracted_output_types.csv    per-project labels + matched terms + source
  output_type_distribution.csv  label, count, % of disclosing projects
  output_type_by_region.csv     label x english_majority, counts + row-%
  output_type_by_experience.csv label x first_time, counts + row-%
  outcomes_by_output_type.csv   funding outcomes per stated output type
  verification_sample.csv       a seeded 10% subsample with the dictionary labels
  verification_human_BLANK.csv  the same subsample with blank columns, to be
                                re-coded blind for the Cohen's kappa validation
  figures/                      PNG (150 dpi) + SVG, one chart per file
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from ksai import charts, config, disclosures, extract

logger = logging.getLogger("run_06")

CONTENT = config.RESULTS / "content"  # committed: safe aggregate tables + figures
CONTENT_PRIVATE = config.RESULTS_PRIVATE / "content"  # gitignored: per-project files
FIGURES = CONTENT / "figures"
VALIDATION_FRACTION = 0.10


def label_projects(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the dictionary classifier to each disclosing project."""
    rows = []
    for rec in df.itertuples(index=False):
        flags = {"involvesFunding": rec.involves_funding, "involvesOther": rec.involves_other}
        result = extract.classify(rec.disclosure_text, category=rec.category, flags=flags)
        rows.append(
            {
                "id": int(rec.id),
                "output_labels": result.label_str,
                "source": result.source,
                "matched_terms": "; ".join(
                    f"{label}: {', '.join(terms)}" for label, terms in result.matched.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_validation_sample(labelled: pd.DataFrame, df: pd.DataFrame) -> int:
    """Draw a seeded 10% subsample and write the validation files.

    ``verification_sample.csv`` carries the dictionary labels; the BLANK file is
    the same rows with empty reference columns, to be re-coded blind (no human
    code is ever written by this pipeline). ``verification_agreement.py`` then
    computes Cohen's kappa between the dictionary and the re-coded reference.
    """
    n_val = math.ceil(VALIDATION_FRACTION * len(labelled))
    sample = labelled.sample(n=n_val, random_state=config.RANDOM_SEED).merge(
        df[["id", "disclosure_text"]], on="id", validate="one_to_one"
    )
    sample[["id", "disclosure_text", "output_labels"]].to_csv(
        CONTENT_PRIVATE / "verification_sample.csv", index=False
    )
    blank = sample[["id", "disclosure_text"]].copy()
    blank["human_labels"] = ""
    blank["human_notes"] = ""
    blank.to_csv(CONTENT_PRIVATE / "verification_human_BLANK.csv", index=False)
    return n_val


def main() -> None:
    CONTENT.mkdir(parents=True, exist_ok=True)
    CONTENT_PRIVATE.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(config.DERIVED / "sample.parquet")
    df = disclosures.load_disclosing(config.DERIVED / "ai_flags.jsonl", sample)
    n = len(df)
    logger.info("classifying %d disclosing projects", n)

    freq = extract.term_frequencies(df["disclosure_text"].tolist())
    freq.to_csv(CONTENT / "term_tool_frequency.csv", index=False)

    labelled = label_projects(df).merge(
        df[
            [
                "id",
                "category",
                "english_majority",
                "first_time",
                "success",
                "pledged_usd",
                "pct_funded",
                "backers",
            ]
        ],
        on="id",
        validate="one_to_one",
    )
    labelled.drop(columns=["success", "pledged_usd", "pct_funded", "backers"]).to_csv(
        CONTENT_PRIVATE / "extracted_output_types.csv", index=False
    )

    dist = extract.label_distribution(labelled, total=n)
    dist.to_csv(CONTENT / "output_type_distribution.csv", index=False)
    by_region = extract.label_crosstab(labelled, "english_majority")
    by_region.to_csv(CONTENT / "output_type_by_region.csv", index=False)
    by_experience = extract.label_crosstab(labelled, "first_time")
    by_experience.to_csv(CONTENT / "output_type_by_experience.csv", index=False)
    merged = extract.explode_labels(labelled)
    outcomes = extract.outcomes_by_label(merged)
    outcomes.to_csv(CONTENT / "outcomes_by_output_type.csv", index=False)

    n_val = write_validation_sample(labelled, df)

    charts.plot_output_type_distribution(dist, FIGURES)
    charts.plot_output_type_pie(
        dist, FIGURES
    )  # alternative format; multi-label caveat in the figure
    charts.plot_output_type_by_region(by_region, FIGURES)
    charts.plot_success_by_output_type(outcomes[outcomes["label"] != extract.UNCLASSIFIED], FIGURES)
    charts.plot_top_tools(freq, FIGURES)
    prevalence_csv = config.RESULTS / "prevalence_by_quarter.csv"
    if prevalence_csv.exists():
        charts.plot_prevalence_by_quarter(pd.read_csv(prevalence_csv), FIGURES)
    else:
        logger.warning("no prevalence_by_quarter.csv (run run_05 first); skipping that figure")

    unclassified = int((labelled["output_labels"] == extract.UNCLASSIFIED).sum())
    flags_only = int((labelled["source"] == "flags_only").sum())
    print("=" * 72)
    print("DICTIONARY CONTENT ANALYSIS SUMMARY (descriptive; truthful self-report assumed)")
    print("=" * 72)
    print(
        f"projects: {n} | flags_only (no text): {flags_only} | "
        f"unclassified: {unclassified} ({100 * unclassified / n:.1f}%)"
    )
    print(f"validation subsample written: {n_val} projects ({100 * n_val / n:.1f}%)")
    print("\n-- output-type distribution (multi-label; % of disclosing projects) --")
    print(dist.to_string(index=False))
    print("\n-- top named tools / terms --")
    print(freq.head(20).to_string(index=False))
    print("\n-- output type x english_majority --")
    print(by_region.to_string(index=False))
    print("\n-- output type x first_time --")
    print(by_experience.to_string(index=False))
    print("\n-- funding outcomes per stated output type (descriptive) --")
    print(outcomes.to_string(index=False))
    print(f"\nall tables and figures in {CONTENT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
