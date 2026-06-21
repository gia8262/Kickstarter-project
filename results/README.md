# Results

These are the aggregate result tables and figures from the study, committed so
they can be read directly on GitHub. The pipeline writes them here directly
(`run_05_analysis.py` and `run_06_content.py`); the per-project, identifiable
files produced alongside them stay under gitignored `data/derived/results/`, on
personal-data grounds. Nothing here contains individual-level scraped text or
per-project rows.

## Provenance

`run_05_analysis.py` writes the structural tables:

- `prevalence_by_category.csv`, `prevalence_by_quarter.csv` — disclosure rate per
  category and per launch quarter, each with a 95% Wilson interval.
- `outcomes_overall.csv` — AI-disclosing versus other on the outcome variables,
  with Benjamini-Hochberg rejection flags.
- `inclusion_region.csv`, `inclusion_experience.csv` — the inclusion splits.
- `matching_balance.csv`, `outcomes_matched.csv` — the propensity-matching
  balance check and the matched comparison.

`run_06_content.py` writes the dictionary content-analysis tables and figures:

- `content/output_type_distribution.csv`, `content/outcomes_by_output_type.csv`,
  `content/output_type_by_region.csv`, `content/output_type_by_experience.csv`.
- `content/term_tool_frequency.csv` — the corpus-grounded scan of stated terms
  and named tools (aggregate counts; no per-project text).
- `content/figures/` — one chart per file as PNG (150 dpi) and SVG.

`verification_agreement.py` writes `content/verification_report.csv`: per-category
agreement and Cohen's kappa between the dictionary and an independent blind
re-coding of a random 10% subsample (157 disclosures).

`content/dictionary.csv` is the classification dictionary itself: the keyword
terms per output category, the named tools and the category each maps to, and the
rejected terms with the reason each was excluded (the term list applied by
`ksai/extract.py`).

## Not included here

Per-project files (`extracted_output_types.csv`, the validation subsample and
disagreement list) stay under gitignored `data/` because they carry project
identifiers and disclosure text.
