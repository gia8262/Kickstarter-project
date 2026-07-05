"""Tests for ksai.extract (keyword extraction) and ksai.charts (figures)."""

from __future__ import annotations

import pandas as pd
import pytest

from ksai import charts, extract


def test_known_phrases_map_to_expected_labels():
    cases = {
        "We used MidJourney to create the cover art.": "images",
        "ChatGPT helped us draft the campaign description.": "text",
        "Suno generated the background music for the trailer.": "audio_video",
        "Our app uses a neural network for image recognition.": "functional_product",
    }
    for phrase, label in cases.items():
        result = extract.classify(phrase)
        assert label in result.labels, (phrase, result)
        assert result.source == "keywords"
        assert result.matched[label]  # evidence recorded


def test_multilabel_extraction():
    text = "MidJourney made the illustrations and ChatGPT wrote the story."
    result = extract.classify(text)
    assert result.labels == ("images", "text")
    assert result.label_str == "images|text"
    assert "MidJourney" in result.matched["images"]
    assert "ChatGPT" in result.matched["text"]


def test_language_motive_overrides_text():
    text = (
        "English is not my native language, so I used ChatGPT to proofread "
        "the grammar of the campaign description."
    )
    result = extract.classify(text)
    assert "language_translation" in result.labels
    assert "text" not in result.labels
    assert "proofread" in result.matched["language_translation"]


def test_text_without_language_motive_stays_text():
    result = extract.classify("AI helped with writing the story and descriptions.")
    assert "text" in result.labels
    assert "language_translation" not in result.labels


def test_empty_text_classifies_from_flags():
    result = extract.classify("", flags={"involvesAi": True, "involvesFunding": True})
    assert result.labels == ("functional_product",)
    assert result.source == "flags_only"
    # markers alone count as empty
    assert extract.classify("[generatedByAiDetails]", category="Technology").labels == (
        "functional_product",
    )
    # no flags, no informative category -> unclassified, still flags_only
    bare = extract.classify("", category="Art", flags={"involvesGeneration": True})
    assert bare.labels == (extract.UNCLASSIFIED,)
    assert bare.source == "flags_only"


def test_no_keyword_match_is_unclassified():
    result = extract.classify("Wir bedanken uns herzlich bei allen.")
    assert result.labels == (extract.UNCLASSIFIED,)
    assert result.source == "keywords"
    assert result.matched == {}


def test_term_frequencies_strips_markers_and_pools_aliases():
    texts = [
        "[generatedByAiDetails] We used MidJourney and mid journey renders.",
        "[generatedByAiConsent] MidJourney output was curated by our artist.",
        "Stable Diffusion artwork, reviewed by the artist.",
    ]
    freq = extract.term_frequencies(texts, min_projects=2)
    tools = freq[freq["kind"] == "tool"].set_index("term")
    # alias forms pool onto the canonical tool, counted once per project
    assert tools.loc["MidJourney", "n_projects"] == 2
    assert tools.loc["MidJourney", "n_mentions"] == 3
    assert tools.loc["Stable Diffusion", "n_projects"] == 1
    terms = freq[freq["kind"] == "term"]
    # marker field names are stripped, stopwords excluded
    assert not terms["term"].str.contains("generatedby", case=False).any()
    assert "the" not in set(terms["term"])
    assert "artist" in set(terms["term"])  # appears in 2 projects


def test_label_distribution_and_crosstab():
    labelled = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "output_labels": ["images|text", "images", "unclassified", "images"],
            "english_majority": [1, 0, 1, 1],
        }
    )
    dist = extract.label_distribution(labelled, total=4).set_index("label")
    assert dist.loc["images", "count"] == 3
    assert dist.loc["images", "pct_of_disclosing"] == 75.0
    assert dist.loc["text", "count"] == 1
    table = extract.label_crosstab(labelled, "english_majority").set_index("label")
    assert table.loc["images", "english_majority=1"] == 2
    assert table.loc["images", "english_majority=0"] == 1
    assert table.loc["images", "rowpct_english_majority=0"] == pytest.approx(33.33, abs=0.01)


@pytest.fixture
def merged_labels() -> pd.DataFrame:
    labelled = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "output_labels": ["images|text", "images", "text", "images"],
            "success": [1, 0, 1, 1],
            "pledged_usd": [1000.0, 200.0, 5000.0, 800.0],
            "pct_funded": [1.2, 0.4, 2.0, 1.1],
            "backers": [50, 5, 120, 30],
        }
    )
    return extract.explode_labels(labelled)


def test_outcomes_by_label_aggregates_correctly(merged_labels):
    out = extract.outcomes_by_label(merged_labels).set_index("label")
    # images: projects 1, 2, 4 -> success 2/3, median pledged 800
    assert out.loc["images", "n"] == 3
    assert out.loc["images", "success_rate"] == pytest.approx(2 / 3)
    assert out.loc["images", "median_pledged_usd"] == 800.0
    assert out.loc["images", "median_backers"] == 30.0
    # text: projects 1, 3 -> success 1.0, median pledged 3000
    assert out.loc["text", "n"] == 2
    assert out.loc["text", "success_rate"] == 1.0
    assert out.loc["text", "median_pledged_usd"] == 3000.0


def test_charts_write_files(tmp_path, merged_labels):
    labelled = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "output_labels": ["images|text", "images", "text"],
            "english_majority": [1, 0, 1],
        }
    )
    dist = extract.label_distribution(labelled, total=3)
    crosstab = extract.label_crosstab(labelled, "english_majority")
    outcomes = extract.outcomes_by_label(merged_labels)
    freq = extract.term_frequencies(
        ["MidJourney art and artwork", "ChatGPT wrote the story", "MidJourney again"],
        min_projects=1,
    )
    prevalence = pd.DataFrame(
        {
            "launch_quarter": ["2023Q4", "2024Q1", "2024Q2"],
            "rate": [0.03, 0.04, 0.05],
            "ci_lo": [0.02, 0.03, 0.04],
            "ci_hi": [0.04, 0.05, 0.06],
        }
    )
    written = [
        *charts.plot_output_type_distribution(dist, tmp_path),
        *charts.plot_output_type_pie(dist, tmp_path),
        *charts.plot_output_type_by_region(crosstab, tmp_path),
        *charts.plot_success_by_output_type(outcomes, tmp_path),
        *charts.plot_top_tools(freq, tmp_path),
        *charts.plot_prevalence_by_quarter(prevalence, tmp_path),
    ]
    assert len(written) == 12  # 6 charts x (png + svg)
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    assert {p.suffix for p in written} == {".png", ".svg"}


def test_structural_charts_write_files(tmp_path):
    region = pd.DataFrame(
        {
            "english_majority": [0, 1],
            "n": [3050, 8973],
            "ai_n": [519, 1046],
            "disclosure_rate": [0.170, 0.117],
            "rate_ci_lo": [0.157, 0.110],
            "rate_ci_hi": [0.184, 0.123],
            "representation_ratio": [1.31, 0.90],
            "success_ai": [0.559, 0.485],
            "success_other": [0.703, 0.713],
            "success_gap": [-0.145, -0.228],
        }
    )
    experience = pd.DataFrame(
        {
            "first_time": [0, 1],
            "n": [5146, 6877],
            "ai_n": [556, 1009],
            "disclosure_rate": [0.108, 0.147],
            "rate_ci_lo": [0.100, 0.139],
            "rate_ci_hi": [0.117, 0.155],
            "representation_ratio": [0.83, 1.13],
            "success_ai": [0.827, 0.334],
            "success_other": [0.869, 0.586],
            "success_gap": [-0.042, -0.252],
        }
    )
    overall = pd.DataFrame(
        {
            "outcome": ["success", "pledged_usd"],
            "median_ai": [0.509, 1006.0],
            "median_other": [0.710, 2819.0],
            "effect": [0.42, -0.22],
            "effect_name": ["odds_ratio", "cliffs_delta"],
        }
    )
    matched = overall.assign(
        median_ai=[0.511, 1010.0], median_other=[0.681, 3060.0], effect=[0.49, -0.24]
    )
    balance = pd.DataFrame(
        {
            "covariate": ["log_goal", "duration_days", "english_majority", "first_time"],
            "smd_before": [0.087, 0.090, -0.199, 0.172],
            "smd_after": [-0.016, 0.005, -0.015, -0.037],
        }
    )
    prevalence = pd.DataFrame(
        {
            "category": ["Technology", "Games", "Music", "Dance"],
            "n": [1224, 2529, 717, 35],
            "ai_n": [397, 467, 30, 0],
            "rate": [0.324, 0.185, 0.042, 0.0],
            "ci_lo": [0.299, 0.170, 0.029, 0.0],
            "ci_hi": [0.351, 0.200, 0.059, 0.099],
        }
    )
    written = [
        *charts.plot_prevalence_by_category(prevalence, tmp_path),
        *charts.plot_disclosure_uptake(region, experience, tmp_path),
        *charts.plot_disclosure_penalty(region, experience, tmp_path),
        *charts.plot_matching_balance(balance, tmp_path),
        *charts.plot_outcome_penalty(overall, matched, tmp_path),
    ]
    assert len(written) == 10  # 5 charts x (png + svg)
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    assert {p.suffix for p in written} == {".png", ".svg"}
