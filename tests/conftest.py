"""Shared pytest fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest


def _ts(y: int, m: int, d: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(y, m, d, tzinfo=UTC))


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small loaded-style frame covering the edges the frame builder must handle."""
    rows = [
        # post-policy, successful  -> kept, success=1
        dict(
            id=1,
            state="successful",
            launched_at=_ts(2024, 1, 1),
            deadline=_ts(2024, 2, 1),
            goal_usd=1000.0,
            pledged_usd=2000.0,
            backers=40,
            country="US",
            creator_id=10,
            state_changed_at=_ts(2024, 2, 1),
            category="games",
        ),
        # post-policy, failed -> kept, success=0
        dict(
            id=2,
            state="failed",
            launched_at=_ts(2024, 3, 1),
            deadline=_ts(2024, 4, 1),
            goal_usd=5000.0,
            pledged_usd=1000.0,
            backers=10,
            country="DE",
            creator_id=10,
            state_changed_at=_ts(2024, 4, 1),
            category="art",
        ),
        # pre-policy -> dropped
        dict(
            id=3,
            state="successful",
            launched_at=_ts(2022, 1, 1),
            deadline=_ts(2022, 2, 1),
            goal_usd=1000.0,
            pledged_usd=1500.0,
            backers=20,
            country="US",
            creator_id=11,
            state_changed_at=_ts(2022, 2, 1),
            category="games",
        ),
        # live -> dropped (not terminal)
        dict(
            id=4,
            state="live",
            launched_at=_ts(2024, 5, 1),
            deadline=_ts(2024, 6, 1),
            goal_usd=1000.0,
            pledged_usd=500.0,
            backers=5,
            country="US",
            creator_id=12,
            state_changed_at=_ts(2024, 5, 15),
            category="design",
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def dump_file(tmp_path):
    """Write a tiny line-delimited JSON dump (valid, unwrapped, and malformed lines)."""
    path = tmp_path / "dump.json"
    wrapped = {
        "data": {
            "id": 100,
            "goal": 1000,
            "static_usd_rate": 1.0,
            "usd_pledged": "1500",
            "backers_count": 30,
            "state": "successful",
            "country": "US",
            "creator": {"id": 5, "name": "A"},
            "category": {"slug": "games/tabletop", "name": "Tabletop"},
            "urls": {"web": {"project": "https://k/p/100"}},
            "launched_at": 1704067200,
            "deadline": 1706745600,
            "state_changed_at": 1706745600,
        }
    }
    unwrapped = {
        "id": 101,
        "goal": 2000,
        "static_usd_rate": 1.0,
        "usd_pledged": "0",
        "backers_count": 0,
        "state": "failed",
        "country": "FR",
        "creator": {"id": 6, "name": "B"},
        "category": {"slug": "art/painting", "name": "Painting"},
        "urls": {"web": {"project": "https://k/p/101"}},
        "launched_at": 1709251200,
        "deadline": 1711929600,
        "state_changed_at": 1711929600,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(wrapped) + "\n")
        f.write(json.dumps(unwrapped) + "\n")
        f.write("{this is not valid json}\n")  # must be skipped
    return path.parent
