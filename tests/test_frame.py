"""Tests for ksai.webrobots and ksai.frame."""

from __future__ import annotations

import pandas as pd
import pytest

from ksai import frame as F
from ksai import webrobots


def test_loader_parses_and_skips_bad_lines(dump_file):
    df = webrobots.load_dir(dump_file)
    assert len(df) == 2  # malformed third line skipped
    assert list(df.columns) == list(webrobots.SCHEMA)
    assert df["id"].dtype == "int64"
    # wrapped and unwrapped both parsed; usd_pledged coerced to float
    assert set(df["id"]) == {100, 101}
    assert df.loc[df["id"] == 100, "pledged_usd"].iloc[0] == 1500.0
    # category derived from slug parent, normalised to title case
    assert df.loc[df["id"] == 100, "category"].iloc[0] == "Games"


def test_loader_normalises_category_case(tmp_path):
    import json

    path = tmp_path / "d.json"
    rows = [
        # one record carries the capitalised parent name
        {
            "data": {
                "id": 1,
                "creator": {"id": 1},
                "category": {"slug": "games/tabletop", "parent_name": "Games"},
                "urls": {"web": {"project": "u1"}},
            }
        },
        # another carries only the lowercase slug
        {
            "data": {
                "id": 2,
                "creator": {"id": 2},
                "category": {"slug": "games/video"},
                "urls": {"web": {"project": "u2"}},
            }
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = webrobots.load_dir(tmp_path, pattern="d.json")
    # both collapse to a single normalised category
    assert set(df["category"]) == {"Games"}


def test_loader_raises_without_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        webrobots.load_dir(tmp_path)


def test_build_frame_filters_to_postpolicy_terminal(raw_frame):
    fr = F.build_frame(raw_frame, "2023-08-29")
    # ids 3 (pre-policy) and 4 (live) dropped; 1 and 2 kept
    assert set(fr["id"]) == {1, 2}
    assert fr.loc[fr["id"] == 1, "success"].iloc[0] == 1
    assert fr.loc[fr["id"] == 2, "success"].iloc[0] == 0
    assert fr.loc[fr["id"] == 1, "pct_funded"].iloc[0] == pytest.approx(2.0)
    assert "launch_quarter" in fr.columns


def test_attach_region(raw_frame):
    fr = F.attach_region(F.build_frame(raw_frame, "2023-08-29"), {"US", "GB"})
    assert fr.loc[fr["id"] == 1, "english_majority"].iloc[0] == 1  # US
    assert fr.loc[fr["id"] == 2, "english_majority"].iloc[0] == 0  # DE


def test_attach_experience(raw_frame):
    fr = F.build_frame(raw_frame, "2023-08-29")
    # creator 10 owns both kept projects -> repeat; build a history where 10 has 2, others 1
    history = pd.DataFrame({"id": [1, 2, 99], "creator_id": [10, 10, 99]})
    fr = F.attach_experience(fr, history)
    assert fr.loc[fr["id"] == 1, "first_time"].iloc[0] == 0  # creator 10 is a repeat
    assert (fr["creator_total_projects"] >= 1).all()


def test_require_raises_on_missing_columns():
    with pytest.raises(KeyError):
        F.build_frame(pd.DataFrame({"id": [1]}), "2023-08-29")


def test_attach_experience_missing_creator_defaults_first_time(raw_frame):
    fr = F.build_frame(raw_frame, "2023-08-29")
    history = pd.DataFrame({"id": [50], "creator_id": [99]})  # frame creators absent
    fr = F.attach_experience(fr, history)
    assert (fr["first_time"] == 1).all()
    assert (fr["creator_total_projects"] == 1).all()
