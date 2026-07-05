# Kickstarter AI disclosure and social inclusion — code and data pipeline

Code for my bachelor thesis on disclosed AI use in reward-based crowdfunding. The
pipeline samples post-policy Kickstarter projects, collects each project's "Use
of AI" disclosure, labels the kind of AI use with a keyword dictionary, and
compares disclosing with non-disclosing projects on their funding outcomes by
creator region and experience.

> This README covers the code only. The background, methodology, results, and
> discussion are in the research paper.

## Summary

Kickstarter has required an AI-use disclosure on project pages since August 2023.
This pipeline builds a sampling frame from the public Web Robots dumps, draws a
stratified random sample, collects the disclosure per project, labels the stated
output types with a keyword dictionary, and reports prevalence, the
AI-versus-other outcome comparison, the region and experience splits, and a
propensity-matching robustness check. It is observational and descriptive, so it
reports associations, not causal effects. Result tables and figures are under
`results/`; the rest of the details are in the research paper.

## Install

I built and ran this on macOS with Python 3.11 or 3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev,analysis]" + pre-commit install
```

To reproduce the exact tested stack instead, use `pip install -r
requirements.txt`, which pins every version.

## Run order

The scripts are numbered in the order they run. The first time, generate
synthetic dumps so the offline steps work without any real data.

```bash
python make_synthetic_data.py   # optional: fake dumps for offline testing
python run_00_diagnose.py       # confirm which collection path works
python run_01_frame.py          # -> data/derived/frame.parquet
python run_02_sample.py         # -> data/derived/sample.parquet + sample_urls.csv
python run_03_probe.py          # free pilot: prevalence + path check
python run_04_collect.py graphql  # full free collection -> data/derived/ai_flags.jsonl
python run_05_analysis.py       # structural result tables -> results/
python run_06_content.py        # content tables + figures -> results/content/ (+ private files)
python verification_agreement.py  # dictionary validation: Cohen's kappa (after blind re-coding)
```

The analysis dependencies install with `pip install -e ".[analysis]"`. `make
pipeline` runs the synthetic generator and the two offline frame steps as a quick
smoke test.

## Quality

```bash
make check        # lint + typecheck + tests (what CI runs)
make test         # pytest with coverage (fails under 90%)
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
```

A GitHub Actions workflow runs the same gate on Python 3.11 and 3.12, and
pre-commit hooks enforce formatting, linting, and types on commit.

## Repository layout

```
pyproject.toml            package metadata, dependencies, tool config
requirements.txt          pinned versions for exact reproducibility
Makefile                  install / test / lint / format / typecheck / pipeline
results/                  the committed aggregate tables and figures
ksai/config.py            paths and study constants
ksai/webrobots.py         load and parse the Web Robots dumps
ksai/frame.py             build the frame, attach region and experience
ksai/sampling.py          stratified sampling, Wilson interval, size arithmetic
ksai/scrape.py            free GraphQL collection and the disclosure detector
ksai/analysis.py          prevalence, outcome comparisons, inclusion splits
ksai/matching.py          propensity scores, caliper matching, balance table
ksai/disclosures.py       assemble disclosure text, load the disclosing table
ksai/extract.py           the dictionary classifier and its term list
ksai/charts.py            the figures
ksai/verify.py            the dictionary validation report
ksai/synthetic.py         the offline synthetic-dump generator
run_00..run_06            the numbered pipeline scripts
verification_agreement.py the dictionary validation script
tests/                    the pytest suite and fixtures
data/                     raw/, archive/, derived/  (gitignored; see below)
```

## Data and reproducibility

The complete collection and analysis code is in the repository, and every
aggregate under `results/` is produced by it from the collected data. The
per-project files — the disclosure text, the analysis frame, and the validation
subsample — are **not** committed: each row pairs a project id and URL with
creator-written text and is therefore identifiable personal data, so the folders
under `data/` are gitignored and empty here. The underlying dataset can be
regenerated from the public source with the collection scripts, or provided
privately on request. Runs are deterministic given the dumps and the fixed
sampling seed (42), with dependency versions pinned in `requirements.txt`.

## License

MIT, see `LICENSE`.
