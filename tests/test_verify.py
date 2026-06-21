"""Tests for the dictionary expansion (ksai.extract) and ksai.verify."""

from __future__ import annotations

import pandas as pd
import pytest

from ksai import extract, verify

# --- (a) approved dictionary expansion ---------------------------------------


def test_expansion_terms_classify_known_phrases():
    cases = {
        "Einige Hintergründe der Bilder wurden mit KI erstellt.": "images",
        "Utilizare imágenes generadas con IA para la portada.": "images",
        "AI was used to generate early sketches of the figures.": "images",
        "We used AI for the AI-generated audiobook.": "audio_video",
        "KI half mir bei der Rechtschreibung, mehr nicht!": "language_translation",
        "AI powers our in-game NPC dialogue and spam detection.": "functional_product",
        "We use AI for personalized recommendations via Retrieval Augmented Generation.": (
            "functional_product"
        ),
        "AI helped create the storytelling of our project.": "text",
    }
    for phrase, label in cases.items():
        result = extract.classify(phrase)
        assert label in result.labels, (phrase, result)


def test_expansion_tools_map_to_labels():
    assert "text" in extract.classify("Health insights integrated DeepSeek models.").labels
    assert "audio_video" in extract.classify("Reel clips made using VEED.").labels
    assert "functional_product" in extract.classify("Built on Langchain RAG.").labels
    assert "text" in extract.classify("Drafted with MS Copilot.").labels
    # OpenAI is discovery-only: counted as a tool, assigns no label
    assert extract.TOOL_LABELS.get("OpenAI") is None
    assert extract.classify("We used OpenAI.").labels == (extract.UNCLASSIFIED,)


def test_rejected_vague_terms_are_absent():
    rejected = {
        "ai technology",
        "photography",
        "covers",
        "personalized",
        "analyze",
        "analyse",
        "reels",
        "content",
        "tool",
        "work",
        "create",
        "generate",
    }
    all_terms = {t for terms in extract.LABEL_KEYWORDS.values() for t in terms}
    assert rejected.isdisjoint(all_terms)


def test_existing_tool_mappings_unchanged():
    expected = {
        "ChatGPT": "text",
        "Claude": "text",
        "Gemini": "text",
        "Grok": "text",
        "MidJourney": "images",
        "DALL-E": "images",
        "Stable Diffusion": "images",
        "Adobe Firefly": "images",
        "Canva": "images",
        "Leonardo AI": "images",
        "ElevenLabs": "audio_video",
        "Suno": "audio_video",
        "Runway": "audio_video",
        "Sora": "audio_video",
        "Pika": "audio_video",
        "HeyGen": "audio_video",
        "DeepL": "language_translation",
        "Grammarly": "language_translation",
    }
    for tool, label in expected.items():
        assert extract.TOOL_LABELS[tool] == label


# --- (b) agreement tooling ----------------------------------------------------


@pytest.fixture
def three_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "output_labels": ["images|text", "images", "audio_video", "unclassified", "images"],
        }
    )
    auto = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "auto_labels": ["text|images", "images", "audio_video", "text", "images"],
        }
    )
    human = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "human_labels": ["images", "images", "audio_video", "unclassified", "images"],
        }
    )
    return sample, auto, human


def test_label_sets_are_order_insensitive(three_sources):
    sample, auto, _ = three_sources
    a = verify.label_sets(sample, "output_labels")
    b = verify.label_sets(auto, "auto_labels")
    assert a[1] == b[1] == frozenset({"images", "text"})


def test_exact_match_rate(three_sources):
    sample, auto, human = three_sources
    dictionary = verify.label_sets(sample, "output_labels")
    auto_sets = verify.label_sets(auto, "auto_labels")
    human_sets = verify.label_sets(human, "human_labels")
    # dict vs auto: ids 1 (set-equal), 2, 3, 5 match; 4 differs -> 4/5
    assert verify.exact_match_rate(dictionary, auto_sets) == (0.8, 5)
    # dict vs human: ids 2, 3, 4, 5 match; 1 differs -> 4/5
    assert verify.exact_match_rate(dictionary, human_sets) == (0.8, 5)


def test_cohens_kappa_hand_computed():
    # presence of "images": x = [1,1,0,0,1], y = [1,0,0,0,1]
    # po = 4/5; pe = (3/5)(2/5) + (2/5)(3/5) = 12/25; kappa = (.8-.48)/.52 = 8/13
    x = [True, True, False, False, True]
    y = [True, False, False, False, True]
    assert verify.cohens_kappa(x, y) == pytest.approx(8 / 13)
    # constant identical ratings -> chance agreement 1 -> undefined
    assert verify.cohens_kappa([True, True], [True, True]) != verify.cohens_kappa(
        [True, True], [True, True]
    )  # NaN
    with pytest.raises(ValueError):
        verify.cohens_kappa([], [])


def test_per_label_agreement(three_sources):
    sample, _, human = three_sources
    dictionary = verify.label_sets(sample, "output_labels")
    human_sets = verify.label_sets(human, "human_labels")
    table = verify.per_label_agreement(dictionary, human_sets).set_index("label")
    # text: dict has it for id 1 only, human never -> agreement 4/5
    assert table.loc["text", "agreement"] == pytest.approx(0.8)
    assert table.loc["text", "n_a"] == 1
    assert table.loc["text", "n_b"] == 0
    # images: identical presence everywhere -> agreement 1, kappa 1
    assert table.loc["images", "agreement"] == 1.0
    assert table.loc["images", "kappa"] == pytest.approx(1.0)
    assert (table["n"] == 5).all()


def test_run_report_full_and_human_not_ready(tmp_path, capsys, three_sources):
    sample, auto, human = three_sources
    sample.to_csv(tmp_path / "verification_sample.csv", index=False)
    auto.to_csv(tmp_path / "verification_auto.csv", index=False)

    verify.run_report(
        tmp_path / "verification_sample.csv",
        tmp_path / "verification_auto.csv",
        tmp_path / "verification_human.csv",
        tmp_path,
    )
    out = capsys.readouterr().out
    assert "human file not ready" in out
    assert "dictionary_vs_auto" in out and "dictionary_vs_human" not in out
    assert (tmp_path / "verification_report.csv").exists()

    human.to_csv(tmp_path / "verification_human.csv", index=False)
    verify.run_report(
        tmp_path / "verification_sample.csv",
        tmp_path / "verification_auto.csv",
        tmp_path / "verification_human.csv",
        tmp_path,
    )
    out = capsys.readouterr().out
    assert "human file not ready" not in out
    assert "dictionary_vs_human" in out and "auto_vs_human" in out
    report = pd.read_csv(tmp_path / "verification_report.csv")
    assert set(report["comparison"]) == {
        "dictionary_vs_auto",
        "dictionary_vs_human",
        "auto_vs_human",
    }
    disagreements = pd.read_csv(tmp_path / "verification_disagreements.csv")
    # ids 1 (dict images|text vs human images) and 4 (auto text) disagree somewhere
    assert set(disagreements["id"]) == {1, 4}
