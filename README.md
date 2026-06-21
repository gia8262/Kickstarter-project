# Kickstarter AI disclosure and social inclusion: an exploratory study and its findings

This repository holds the empirical pipeline for my bachelor thesis on disclosed
artificial-intelligence use in reward-based crowdfunding. The study measures how
often Kickstarter creators declare AI use under the platform's post-2023
disclosure rule, what kind of AI use they declare, and whether that use and its
funding outcomes are distributed evenly across creator groups. The guiding
question is whether disclosed AI use looks like a force for wider participation
or one that tracks existing advantages.

The full method is set out as the [Methodology](#methodology) section at the end
of this document, which the sections in between summarise. The design is
observational, cross-sectional, and descriptive: I characterise patterns and
associations, and I do not claim causal effects.

## What the study measures

The unit of analysis is the individual Kickstarter project. The focal variable
is `ai_disclosed`, a binary flag set from the presence of a "Use of AI" section
on the project page; this captures disclosed and platform-approved AI use, not
all AI use, a boundary kept as a limitation. The outcomes are the standard
crowdfunding-determinants set: funding success (all-or-nothing, so pledged at
or above goal), amount pledged, percentage funded, and backer count, with goal
size as a covariate. Two inclusion lenses are implemented: creator region
(English-majority countries, the United States, United Kingdom, Canada,
Australia, New Zealand, and Ireland, against the rest) and creator experience
(first-time against repeat). The cautious name-based gender lens discussed in
the methodology is not implemented, and the package carries no gender-inference
code.

## Two data sources, and why both are needed

The disclosure exists in no bulk dataset, so the study joins two inputs on the
project id. The Web Robots monthly Kickstarter dumps give the project list,
terminal outcomes, structural covariates, and page URLs: the backbone and
sampling frame, carrying no AI information. Kickstarter's own public GraphQL
interface serves the `aiDisclosure` object behind the "Use of AI" section, and
is where the `ai_disclosed` flag and the disclosure text come from; retrieving
it per project is the study's primary collection. The frame keeps one terminal
record per project, filtered to completed (successful or failed) projects
launched on or after the policy date of 29 August 2023, holding **54,821
projects**, from which a stratified random sample of **25,000** is drawn
proportionally across the 15 top-level categories (with a per-category floor)
under a fixed seed of **42**.

## How the disclosure data was collected

Collection runs over the free path and cost nothing. A request to the homepage
yields a csrf token, and that token plus the session cookies let the `/graph`
endpoint answer the `aiDisclosure` query for each project without rendering
JavaScript, using `curl_cffi` with browser TLS impersonation
(`ksai/scrape.py`). The collector is deliberately polite and resumable: it
spaces requests, retries rate-limited and transient responses with exponential
backoff, honours a Retry-After header, and writes only successful HTTP 200
records, one per project id. A page that still fails is recorded separately
with its status and the flag left unknown, never as non-disclosing, so a
blocked page cannot be mistaken for a declared absence of AI. `run_03_probe.py`
pilots the path; `run_04_collect.py` runs the full collection.

## Content analysis: the dictionary

To describe what kind of AI use creators declare, a fixed and auditable list of
terms (`ksai/extract.py`) assigns each disclosure to one or more output-type
categories, generative images, text, language assistance, audio or video, and
functional or product use, or to `unclassified` where no rule matches. It codes
manifest content under an assumption of truthful self-report, while the
category definitions remain my judgement. The dictionary was grounded in a
corpus frequency scan and then verified and expanded, with the added entries
marked inline and the rejected boilerplate terms pinned in
`tests/test_verify.py`. The full dictionary is committed as a browsable table
at `results/content/dictionary.csv`. The 11.5% of disclosures that match no
rule are a genuinely unspecific residual rather than a gap in the dictionary.

## Validating the dictionary

Because the dictionary applies a fixed rule to every disclosure, its
reliability is whether the rule-based labels match how a reader interprets the
same text. A seeded 10% subsample (157 of the 1,565) was re-coded from the
disclosure text alone, blind to the dictionary output, through both an
independent automated pass and my own reading, which agreed. `ksai/verify.py`
reports per-category agreement and Cohen's kappa to
`results/content/verification_report.csv`. Agreement runs from 73% to 96% per
category; kappa is substantial for images (0.69) and audio or video (0.70),
moderate for language assistance (0.57), functional or product use (0.49), and
text (0.41), and low only for unclassified (0.12), where a small base rate
deflates kappa despite 85% raw agreement.

## The analysis

`run_05_analysis.py` produces the structural results, all of them
associational: prevalence per category and per launch quarter with Wilson score
95% intervals; the AI-disclosing group against the rest on success with a
chi-square test and an odds ratio, and on the skewed continuous outcomes with
Mann-Whitney U and Cliff's delta; Benjamini-Hochberg control across the family
of subgroup comparisons; and, for the inclusion layer, each group's disclosure
rate, representation ratio, and success gap. As a check on confounding from
observed covariates, `ksai/matching.py` fits a logistic propensity model and
performs 1:1 nearest-neighbour caliper matching (0.2 standard deviations of the
logit, Austin 2011) with standardised mean differences before and after; it
balances observed covariates only, so it is a check, not a causal estimate.
`run_06_content.py` produces the content tables and figures, and
`verification_agreement.py` the dictionary agreement report.

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

The repository contains the complete collection code. The disclosures are
collected from Kickstarter's public GraphQL interface over the free `curl_cffi`
path, and that code is fully open to inspection in `ksai/scrape.py`. The
disclosure data was collected with exactly this code, and every number under
`results/` derives from that collected data and nothing else. Flags were
collected for 12,023 of the 25,000 sampled projects (a random subset of the
seeded draw, since collection ran in shuffled order), one record per project id,
and 1,565 of those declared AI use.

The per-project disclosure text, the analysis frame, and the validation
subsample are not committed, because each row pairs a project id and URL with the
creator-written disclosure text and is therefore identifiable personal data. They
are held in restricted local storage, which is why the folders under `data/` are
empty in the published repository. The raw disclosures can be regenerated from
the public source with the collection scripts, and the complete underlying
dataset can be provided privately on request, for example to my supervisor, so
the provenance of every reported number can be verified directly.

Reproducibility rests on a fixed sampling seed (42), versions pinned in
`requirements.txt`, and the test suite with coverage reported by the gate. The
sampling frame and the stratified draw are deterministic given the dumps and the
seed.

## Results

The aggregate tables and figures are in `results/`, with a note there on which
script writes each file. The figures below are read from those tables.

Across the 12,023 sampled projects with a collected flag, 1,565 declared AI use,
a prevalence of 13.0%. Disclosure is most common in Technology (32.4%) and lowest
in Music (4.2%) and Photography (3.6%); no Dance project in the sample disclosed.

Disclosing projects do worse on every funding outcome, and the gaps survive
Benjamini-Hochberg correction. They succeed at 50.9% against 71.0% for the rest
(odds ratio 0.42), with a lower median amount pledged (about 1,006 against 2,819
US dollars) and fewer median backers (16 against 44). Goals are close (about
4,340 against 3,000), so the gap sits in backer response rather than in what
creators ask for.

The inclusion layer shows the disclosure is taken up more, not less, by the
groups the inclusion literature watches. Non-English-majority projects disclose
at 17.0% against 11.7% for English-majority ones (a representation ratio of 1.31),
and first-time creators disclose at 14.7% against 10.8% for repeat creators
(1.13). The funding penalty attached to disclosure is not even across groups: it
is larger for English-majority projects (a success gap of -0.23 against -0.14)
and much larger for first-time creators (-0.25 against -0.04 for repeat
creators).

As a check on confounding, 1,545 of the 1,565 disclosing projects matched a
non-disclosing control on goal, duration, region, and experience within the
caliper, and all four covariates were balanced afterwards (standardised mean
differences below 0.10). The success, pledged, and backer gaps persist in the
matched comparison (success 51.1% against 68.1%), while goals become
indistinguishable, so those gaps are not an artefact of the observed covariates.
This balances only observed covariates, so it is a check, not a causal estimate.

On what kind of AI use is declared, images dominate: of the 1,565 disclosing
projects, 65.4% state an image use, 30.7% text, 19.1% audio or video, 18.0% a
functional or product use, and 8.5% a language or translation use (the labels are
multi-label, so they sum above 100%). The most-named tools are ChatGPT,
MidJourney, and Photoshop's generative features. 11.5% of disclosures match no
output-type rule and are genuinely unspecific, as the dictionary section explains.

## Limits

I read these findings as associations, not effects. The design is cross-sectional
and observational: each project is seen once at its terminal state, and AI use is
self-selected by creators rather than assigned. The disclosure flag measures
declared and platform-approved AI use, not all AI use, and undisclosed users sit
in the comparison group, which biases comparisons toward the null. Project
country is a proxy for creator location, not verified origin, and the single
collection window undercounts repeat creators. The matched comparison balances
only observed covariates, so self-selection into AI use remains. The findings
apply to Kickstarter, reward-based crowdfunding, and the post-policy period, and
do not generalise to donation or equity crowdfunding.

## Methodology

The full methodological specification for the empirical chapter: research
design, sampling, variables, data collection, the content-analysis instrument
and its validation, the analytical techniques, and the validity, reliability,
and ethics framework. Methodological references are named inline (for example
Krippendorff, 2004).

### 0. Reading guide

The study has two strands run together: a quantitative structural strand on the full sample, and a dictionary-based content-analysis strand on the AI-disclosing subset. The design is **descriptive and exploratory**. It does not estimate causal effects. Wherever a stronger inferential or causal technique exists, it is named and either included as an explicitly labelled robustness layer or excluded with a stated reason.


### 1. Research design

**Design type.** Observational, cross-sectional, descriptive study with an embedded dictionary-based content analysis.
- *Observational* means no variable is manipulated by the researcher; AI use is self-selected by creators.
- *Cross-sectional* means each project is observed once, at its terminal state, rather than followed over time.
- *Descriptive and exploratory* means the goal is to characterise patterns and associations, not to test a causal hypothesis.

**Design classification.** Convergent embedded design: a dominant quantitative structural strand on the full sample with an embedded content-analysis strand on the AI-disclosing subset, analysed in parallel and integrated at interpretation. The content strand is dictionary-based content analysis, a systematic rule-based coding of manifest content, rather than interpretive qualitative coding.

**Unit of analysis.** The individual Kickstarter project.
**Level of analysis.** Project level. Not creator level and not backer level. Creator attributes enter only as project-level covariates.

**Why not the alternatives.**
- *Experimental / RCT*: impossible, AI use cannot be randomly assigned to real campaigns.
- *Quasi-experiment (difference-in-differences, regression discontinuity)*: feasible in principle around the 29 August 2023 policy, but already done in the published literature and dependent on text-based AI classification rather than the disclosure flag; excluded from the main design, noted as the causal route in the limitations.
- *Longitudinal panel*: not supported, a completed project has a single terminal outcome.


### 2. Population, sampling frame, and units

**Target population.** All Kickstarter projects launched on or after 29 August 2023 (the date the disclosure requirement took effect) that have reached a terminal funding state.

**Sampling frame.** The Web Robots monthly Kickstarter datasets, deduplicated to one terminal record per project id. The frame is a secondary scraped dataset of the type used across crowdfunding research.

**Frame error to acknowledge.** Coverage is limited to projects the Web Robots crawler captured; very short-lived or removed projects may be missing. This is *frame error* and is stated as a limitation.

**Inclusion criteria.** state in {successful, failed}; launched_at on or after 2023-08-29; non-missing goal and pledged.
**Exclusion criteria.** state in {live, canceled, suspended}; missing core fields. Canceled projects are excluded because the funding outcome is not interpretable; suspended projects are excluded because they were removed by the platform.


### 3. Sampling design

**Chosen method: stratified random sampling.**
- *Definition.* The frame is divided into mutually exclusive strata, and a simple random sample is drawn within each stratum.
- *Strata.* Project category (the 15 top-level Kickstarter categories), optionally crossed with launch quarter.
- *Allocation.* Proportional allocation by default (stratum sample size proportional to stratum size in the frame), with the option of disproportionate (optimum) allocation to oversample categories where AI disclosure is rare, so every category yields analysable AI counts.
- *Purpose / goal.* Guarantee that AI-disclosing projects are represented across categories and reduce sampling variance relative to simple random sampling.

**Alternatives considered.**
- *Simple random sampling (SRS).* Every project has equal selection probability. Simpler, but risks thin AI counts in small categories. Acceptable fallback.
- *Systematic sampling.* Every k-th project from an ordered frame. Risks periodicity bias; not used.
- *Cluster sampling.* Sample clusters (for example categories) then all units within. Higher variance per unit; not used.
- *Census / complete enumeration.* All post-policy projects. Statistically unnecessary (see 3.3) and operationally costly; not used.
- *Convenience or keyword (search-based) sampling.* Selecting projects that mention AI in text. Non-probability, introduces *selection bias*, cannot support prevalence estimates. Used only as a supplement (see 3.2).
- *Quota sampling.* Non-probability filling of fixed group quotas. Not used for the main sample.

#### 3.2 Supplementary purposive oversample (content strand only)
If the pilot shows AI disclosure prevalence is low and the random sample yields too few AI projects for the content coding, enlarge the AI subset with a **purposive (targeted) sample** drawn via Kickstarter's search, kept strictly separate and documented. This is a *sequential nested mixed-sampling* step. Prevalence and the outcome comparison continue to use only the probability sample; the purposive additions feed only the descriptive content analysis. State this separation explicitly so unbiased estimates are preserved.

#### 3.3 Sample size determination
**(a) For the prevalence estimate (a proportion).**
n = z^2 * p * (1 - p) / e^2, with z = 1.96 for 95% confidence.
- Worst case p = 0.5, margin e = +/- 1%: n ~= 9,604.
- With an expected p ~= 0.05, margin +/- 1%: n ~= 1,825.
- Apply the finite population correction n_adj = n / (1 + (n - 1)/N); with N in the hundreds of thousands it is negligible.
- A sample of 10,000 yields a prevalence margin of roughly +/- 0.4% at p = 0.05. Prevalence is therefore over-determined at any N at or above about 2,000.

**(b) For the content analysis and the inclusion subgroup splits (the binding constraint).**
Target a minimum AI-disclosing count. Required total N = target_AI / prevalence.
- For category-level content distributions, aim for at least 300 to 400 AI projects.
- For chi-square subgroup comparisons, satisfy Cochran's rule: all expected cell counts at or above 1, and no more than 20% below 5; as a practical floor keep observed cells at or above about 30.
- A single inclusion split (for example AI by region) is supported by roughly 400 AI projects. Finer two-way splits (region by experience) need substantially more, which is why a sample of 20,000 to 30,000 (yielding on the order of 800 to 1,500 AI projects at a 4 to 5% prevalence) is the recommended size.

**(c) For the optional logistic layer.**
Events-per-variable rule (Peduzzi et al., 1996): at least 10 events of the rarer outcome per predictor. With success common and a large sample this is comfortably met; the practical limit is AI-subgroup size, again satisfied at 20,000 plus.

**Recommendation.** Stratified random sample of about 20,000 to 30,000 rendered pages. Larger is permitted but yields diminishing returns; a census is not warranted.


### 4. Variables and operationalisation

#### 4.1 Focal independent variable
- **ai_disclosed** (binary, 0/1). Operationalised as the presence of a "Use of AI" section on the project page.
- *Measurement validity caveat.* This captures *disclosed and platform-approved* AI use, not all AI use. Undisclosed use is invisible, and the disclosed set skews toward generated images and AI-as-product rather than the writing assistance emphasised in parts of the inclusion literature. State this construct boundary explicitly.

#### 4.2 Dependent (outcome) variables
The set follows the indicators that the crowdfunding-determinants literature treats as standard (the four most-used measures are a binary funding-success indicator and the continuous amount raised, success ratio, and number of backers; see the systematic determinants review, Financial Innovation, 2022).
- **funding_success** (binary). Kickstarter is all-or-nothing, so success = pledged >= goal.
- **amount_pledged_usd** (continuous, right-skewed). Reported as median with IQR; for any modelling, log-transformed as log(1 + pledged), per standard practice for skewed funding amounts.
- **funding_ratio** = pledged / goal (continuous).
- **backers_count** (count). For modelling, negative binomial (count data with overdispersion).
- **pledge_per_backer** = pledged / backers (continuous; signals backer commitment).
- **goal_usd** (continuous). Treated both as a covariate and as an outcome of interest, since the penalty literature locates the disclosure effect in backer behaviour rather than goal-setting; showing similar goals across groups supports that reading.

#### 4.3 Content-level variable (dictionary-based, nominal, multi-label)
Assigned only to AI-disclosing projects, from the "Use of AI" disclosure text.
- **output_type** (multi-label): {generative_images, text, language_assistance, audio_or_video, functional_or_product}. A disclosure that matches no category is recorded as **unclassified**.
The category is assigned by an auditable keyword dictionary applied to the stated text (manifest content); the researcher defines and verifies what each category captures, and a project may carry more than one category.

#### 4.4 Inclusion / moderator variables
- **creator_region** (primary inclusion lens). Operationalised from the project country recorded by the platform.
  - Primary cut: English-majority country vs non-English-majority, matching the cited result that AI writing help benefits non-English-speaking creators most.
  - Secondary cut: World Bank income classification (high / upper-middle / lower-middle / low) or a core-versus-periphery scheme, per the geographic-disparity literature.
  - *Caveat.* Project country is a proxy for creator origin, not verified nationality. Calling it "origin" risks the *ecological fallacy*; label it as platform-recorded project location.
- **creator_experience** (secondary lens). First-time vs repeat, operationalised as the count of distinct campaigns by creator_id in the frame; first-time = 1, repeat >= 2.
  - *Caveat.* The single-window frame *left-censors* creator history and undercounts repeat creators. Mitigate by extending the frame window backward and state the residual bias.
- **inferred_gender** (cautious tertiary lens). From the creator first name via a named tool (the gender-guesser Python package offline, or the genderize.io API with confidence scores). Categories: male / female / unknown.
  - *Caveats.* Standard in the gender-and-crowdfunding literature but Western-name biased, binary, and noisy; report the unknown rate; treat results as suggestive. Personal data under GDPR: aggregate only, never publish a row-level file pairing names with inferred gender.

#### 4.5 Control / covariate variables
For the optional inferential layer and for descriptive stratification. The standard control set in the determinants literature includes log goal, campaign duration, and quality cues such as presence of a video, image count, story length, number of reward tiers, and update speed.
- Available directly in Web Robots: category, log_goal, duration_days, launch_quarter, country, staff_pick.
- Require scraping if used: has_video, image_count, story_length, rewards_count, update_speed. State which controls are included and which are out of scope.


### 5. Data sources and collection

- **Secondary structured data.** Web Robots monthly datasets provide project URLs and all outcome and covariate fields. *Secondary* means collected by a third party for general use.
- **Primary collection.** The ai_disclosed flag and the disclosure text are collected by the researcher by retrieving each sampled project page. *Primary* means generated by this study.
- **Retrieval mechanism.** A managed rendering API (JavaScript rendering plus anti-bot handling) or, if the disclosure text is present in the page payload, a TLS-impersonating request without a browser. Results are cached so each page is retrieved once.
- **Provenance.** Record the scrape date for every page, since pages change over time.


### 6. Content-analysis instrument and validation

**Approach: dictionary-based content analysis.** A predefined list of terms is applied to each disclosure to assign it to one or more output-type categories, while the interpretive judgement of what each category captures remains the researcher's (Krippendorff, 2004). It codes *manifest* content, the AI uses and tool names creators state explicitly, under an assumption of truthful self-report.
- *Alternatives.* Directed (deductive) hand coding by human coders with an intercoder-reliability check (Hsieh & Shannon, 2005); conventional (inductive) coding that lets categories emerge from the text. Dictionary-based coding is chosen because the categories follow directly from the manifest terms creators state, and the rule set is fully auditable and reproducible.

**Instrument.** The keyword dictionary: for each category, the stated terms and named tools that assign it. It was grounded in a corpus frequency scan of the disclosure texts, then verified and expanded by the researcher. Generic process or boilerplate words, and words that name a human medium or a product feature rather than an AI output, are deliberately excluded. The share of disclosures that no rule matches is reported as the unclassified residual.

**Validation.**
- *Coefficient.* Cohen's kappa per category (Cohen, 1960), between the dictionary classification and a blind reference coding.
- *Procedure.* A random subsample of about 10% of the AI-disclosing projects, drawn under a fixed seed, is re-coded from the disclosure text alone and blind to the dictionary output, through both an independent automated pass and a manual reading; where the two agree they form the reference set.
- *Why this check.* Because the dictionary applies a fixed rule to every disclosure, its reliability is the correspondence between the rule-based classifications and how a reader interprets the same disclosure, which per-category kappa measures directly.
- *Reporting.* Report kappa per category. Categories with substantial to almost-perfect agreement are treated as reliable; a low value on a category is reported as a limitation on that category.


### 7. Validity and reliability framework

State each explicitly in the limitations.
- **Construct validity.** Does ai_disclosed measure the intended construct? Partially: it measures disclosed, approved AI use, not all AI use.
- **Internal validity.** Limited by design; this is descriptive, no causal claim is made.
- **External validity.** Findings apply to Kickstarter, reward-based crowdfunding, the post-policy period, and a platform skewed toward English-speaking, higher-income countries. They do not generalise to equity or donation crowdfunding.
- **Measurement reliability.** Structural variables are reliable (machine-recorded); the dictionary classifications are validated against a blind manual re-coding with Cohen's kappa.
- **Statistical conclusion validity.** Protected by adequate sample size, reporting of effect sizes and confidence intervals, and correction for multiple comparisons.


### 8. Analytical techniques

#### 8.1 Descriptive statistics
- Proportions reported with the Wilson score 95% confidence interval (more accurate than the Wald interval for small proportions).
- Central tendency of skewed continuous variables reported as the median with interquartile range, not the mean, because of right skew.
- *Goal.* Characterise prevalence, composition, and group distributions.

#### 8.2 Bivariate association tests
| Comparison | Test | Small-sample variant | Effect size |
|---|---|---|---|
| categorical x categorical (category x AI; group x AI) | Pearson chi-square test of independence | Fisher's exact test when expected cells are small | Cramer's V (phi for 2x2) |
| binary outcome x group (success rate, AI vs non-AI) | two-proportion z-test / chi-square | Fisher's exact | odds ratio with 95% CI, risk ratio, Cohen's h |
| skewed continuous x binary group (pledged, ratio, backers, goal) | Mann-Whitney U (Wilcoxon rank-sum) | exact U | rank-biserial correlation / Cliff's delta |
| disclosure rate over quarters (trend) | Cochran-Armitage test for trend | Mann-Kendall | trend slope |

- *Why nonparametric for continuous outcomes.* Funding amounts and backer counts are heavily right-skewed and violate normality; rank-based tests do not assume normality. Confirm skew with a visual or Shapiro-Wilk before reporting.
- To localise which categories drive a significant chi-square, report adjusted standardised residuals.

#### 8.3 Multiple-comparison control
Because the inclusion layer runs many subgroup comparisons, control the error rate. Set alpha = 0.05 and apply the Benjamini-Hochberg false discovery rate procedure (Benjamini & Hochberg, 1995), or Bonferroni if a more conservative family-wise control is preferred. State the chosen procedure.

#### 8.4 Inclusion-distribution metrics
Two-pronged, mirroring the structure used in the crowdfunding-inequity literature (disparities in both access and outcomes).
- **Access / representation.** Disclosure rate within each group, and the representation ratio = (group share among AI projects) / (group share in the sample); a value above 1 means over-representation.
- **Outcome disparity.** Within each group, the AI-versus-non-AI gap on success rate and median pledged, then a comparison of those gaps across groups (a descriptive gap-of-gaps). Optionally, an ai_disclosed by group interaction term in a logistic model, reported descriptively.
- *Goal.* Describe whether disclosed AI use and its outcome association are distributed evenly across creator groups, the answer to the main research question.

#### 8.5 Optional inferential layer (explicitly labelled robustness, not the main claim)
- **funding_success**: logistic regression on ai_disclosed plus controls; report odds ratios with 95% CIs and pseudo-R-squared.
- **log amount pledged**: ordinary least squares; report coefficients as approximate percentage effects via exp(beta) - 1.
- **backers_count**: negative binomial regression (handles count overdispersion).
- Use heteroskedasticity-robust standard errors; check multicollinearity with the variance inflation factor (VIF, flag above about 5 to 10).
- *Interpretation limit.* Coefficients are associational, not causal. Threats: self-selection into AI use, omitted-variable bias, and endogeneity. The causal remedies (propensity-score matching or difference-in-differences) are named and excluded here because they exceed the descriptive scope and remain fragile given self-selection and the disclosure-versus-use construct gap.


### 9. Software and reproducibility

- Python: pandas (data handling), scipy.stats (tests and effect sizes), statsmodels (regression and the propensity model), and an auditable keyword dictionary with Cohen's kappa for the content analysis and its validation; or R with equivalents.
- Reproducibility: fix random seeds for sampling; version-control the keyword dictionary; archive the raw scraped pages with scrape dates; report all package versions.


### 10. Threats to validity and limitations (consolidated)

1. **Selection bias.** AI disclosure is self-selected and platform-moderated; the AI group is not a random subset of AI users.
2. **Construct validity.** Disclosed AI is not all AI; the disclosed forms differ from the writing assistance much of the inclusion literature studies.
3. **Non-compliance bias.** Undisclosed AI users sit in the non-AI group, biasing comparisons toward the null (a conservative bias).
4. **Frame and coverage error.** Web Robots may miss short-lived or removed projects.
5. **Proxy measurement.** Country as a proxy for origin (ecological fallacy risk); name-based gender inference (noise and Western-name bias); left-censored creator experience.
6. **Multiple comparisons.** Controlled by Benjamini-Hochberg.
7. **External validity.** One platform, reward-based, post-policy, English-skewed.
8. **Temporal confounding.** AI adoption and macro funding conditions both trend with time; descriptive trends are not net of these.


### 11. Ethics and legal

- **Terms of service.** Automated retrieval is restricted by Kickstarter's terms; collection is limited to public pages, rate-limited, cached to avoid repeat requests, and conducted for non-commercial academic research.
- **GDPR / data protection.** Creator names are personal data. Apply data minimisation, analyse and report only in aggregate, store inferred attributes separately from identifiers, and follow the university's data-handling and ethics rules. Do not publish individual-level identifiable data.
- **Transparency.** Document the collection date, the sampling procedure, and the keyword dictionary for reproducibility.


### 12. Research-question mapping

| Question | Strand / step | Variables | Technique |
|---|---|---|---|
| SQ1: which AI uses and tools | Content analysis (step 2) | output_type (multi-label) | dictionary-based content analysis, frequency distributions, Cohen's kappa validation |
| SQ2: how AI may support inclusion | Inclusion layer (step 4) | ai_disclosed x region / experience / gender; outcomes | representation ratio, subgroup comparisons, chi-square, Mann-Whitney, FDR control |
| Context: footprint and diffusion | Prevalence (step 1) | ai_disclosed, category, quarter | proportions, Wilson CI, chi-square, trend test |
| Validation: AI vs non-AI performance | Outcome comparison (step 3) | the six outcome variables | chi-square, Mann-Whitney, odds ratio, Cliff's delta |
| Main RQ: extent AI use relates to social inclusion | Integration | all of the above | convergent interpretation of the structural and content-analysis strands |


### 13. Sampling-size and cost decision (given an available budget)

- Recommended size: stratified random sample of 20,000 to 30,000 rendered pages.
- Statistical justification: prevalence precision is already at +/- 0.4% by 10,000; the binding constraint is AI-subgroup counts for the inclusion splits, which 20,000 to 30,000 satisfies including finer two-way splits.
- A census is not justified: precision gains past 30,000 are negligible while cost and operational risk rise.
- Procedure: pilot 500 to 1,000 pages first to estimate prevalence and confirm the retrieval path, then size the main run from the observed prevalence.
- Cost-control step: run the free TLS-impersonation probe before paying; if the disclosure text is in the page payload, the bulk retrieval is free.

## License

MIT, see `LICENSE`.
