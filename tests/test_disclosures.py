"""Tests for ksai.disclosures."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ksai import disclosures


def test_disclosure_text_concatenates_populated_fields():
    d = {
        "generatedByAiDetails": "made the images",
        "generatedByAiConsent": "",
        "otherAiDetails": "and more",
    }
    text = disclosures.disclosure_text(d)
    assert "[generatedByAiDetails] made the images" in text
    assert "[otherAiDetails] and more" in text
    assert "generatedByAiConsent" not in text  # empty field skipped
    # non-dict inputs yield empty text
    assert disclosures.disclosure_text(None) == ""
    assert disclosures.disclosure_text("not a dict") == ""


def _write_flags(tmp_path, records):
    path = tmp_path / "ai_flags.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def flags_and_sample(tmp_path):
    records = [
        {
            "id": 1,
            "url": "u1",
            "http_status": 200,
            "ai_disclosed": 1,
            "disclosure": {"involvesAi": True, "generatedByAiDetails": "We used MidJourney."},
        },
        {
            "id": 2,
            "url": "u2",
            "http_status": 200,
            "ai_disclosed": 0,
            "disclosure": {"involvesAi": False},
        },
        {
            "id": 3,
            "url": "u3",
            "http_status": 200,
            "ai_disclosed": 1,
            "disclosure": {
                "involvesAi": True,
                "involvesFunding": True,
                "otherAiDetails": "AI app.",
            },
        },
    ]
    path = _write_flags(tmp_path, records)
    sample = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["Games", "Art", "Technology"],
            "english_majority": [1, 0, 1],
            "first_time": [0, 1, 1],
            "success": [1, 0, 1],
            "pledged_usd": [100.0, 0.0, 50.0],
            "pct_funded": [1.2, 0.1, 0.5],
            "backers": [10, 1, 5],
        }
    )
    return path, sample


def test_load_disclosing_keeps_disclosers_with_context(flags_and_sample):
    path, sample = flags_and_sample
    df = disclosures.load_disclosing(path, sample)
    assert set(df["id"]) == {1, 3}  # only ai_disclosed == 1 records
    row1 = df[df["id"] == 1].iloc[0]
    assert "MidJourney" in row1["disclosure_text"]
    assert row1["category"] == "Games"
    assert row1["involves_funding"] is False or not row1["involves_funding"]
    row3 = df[df["id"] == 3].iloc[0]
    assert bool(row3["involves_funding"]) is True


def test_load_disclosing_raises_on_unknown_id(flags_and_sample):
    path, sample = flags_and_sample
    with pytest.raises(ValueError, match="not in the sample"):
        disclosures.load_disclosing(path, sample[sample["id"] != 1])


def test_load_disclosing_raises_when_no_disclosers(tmp_path):
    path = _write_flags(
        tmp_path,
        [{"id": 1, "url": "u", "http_status": 200, "ai_disclosed": 0, "disclosure": None}],
    )
    sample = pd.DataFrame({"id": [1], "category": ["Games"]})
    with pytest.raises(ValueError, match="no AI-disclosing"):
        disclosures.load_disclosing(path, sample)
