"""Load Web Robots Kickstarter dumps (line-delimited JSON) into a tidy DataFrame.

Web Robots publishes monthly CSV and JSON datasets. This loader expects the JSON
form (one project object per line). Some exports wrap the project under a "data"
key and some do not; both are handled. Epoch timestamps are parsed at the column
level (vectorised) rather than per row.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["load_dir", "SCHEMA"]

#: Columns the loader guarantees on its output frame.
SCHEMA: tuple[str, ...] = (
    "id",
    "name",
    "blurb",
    "url",
    "state",
    "category",
    "subcategory",
    "country",
    "creator_id",
    "creator_name",
    "goal_usd",
    "pledged_usd",
    "backers",
    "staff_pick",
    "created_at",
    "launched_at",
    "deadline",
    "state_changed_at",
)
#: Columns holding Unix epoch seconds, converted to UTC datetimes after loading.
_DATE_COLS: tuple[str, ...] = ("created_at", "launched_at", "deadline", "state_changed_at")


def _record(obj: dict[str, Any]) -> dict[str, Any]:
    p = obj.get("data", obj)  # some exports nest the project under "data"
    cat = p.get("category", {}) or {}
    slug = (cat.get("slug") or "").split("/")
    raw_cat = cat.get("parent_name") or (slug[0] if slug and slug[0] else None)
    # Real dumps mix cases ("Games" vs "games"); normalise so they form one stratum.
    parent_cat = raw_cat.strip().title() if raw_cat else None
    loc = p.get("location", {}) or {}
    creator = p.get("creator", {}) or {}
    urls = (p.get("urls", {}) or {}).get("web", {}) or {}

    goal = p.get("goal")
    rate = p.get("static_usd_rate") or 1.0
    goal_usd = (goal * rate) if goal is not None else None

    pledged_usd = p.get("usd_pledged")
    if pledged_usd in (None, ""):
        pledged_usd = p.get("converted_pledged_amount")
    pledged_usd = float(pledged_usd) if pledged_usd not in (None, "") else None

    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "blurb": p.get("blurb"),
        "url": urls.get("project"),
        "state": p.get("state"),
        "category": parent_cat,
        "subcategory": cat.get("name"),
        "country": p.get("country") or loc.get("country"),
        "creator_id": creator.get("id"),
        "creator_name": creator.get("name"),
        "goal_usd": goal_usd,
        "pledged_usd": pledged_usd,
        "backers": p.get("backers_count") or 0,
        "staff_pick": bool(p.get("staff_pick")),
        "created_at": p.get("created_at"),
        "launched_at": p.get("launched_at"),
        "deadline": p.get("deadline"),
        "state_changed_at": p.get("state_changed_at"),
    }


def load_dir(path: str | Path, pattern: str = "*.json") -> pd.DataFrame:
    """Load every line-delimited JSON dump in ``path`` into one DataFrame.

    Malformed lines are skipped and counted; the count is logged so silent data
    loss is visible. Epoch-second columns are converted to UTC datetimes.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    files = sorted(glob.glob(str(Path(path) / pattern)))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {path}")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_record(json.loads(line)))
                except json.JSONDecodeError:
                    skipped += 1
    if skipped:
        logger.warning("skipped %d malformed JSON lines across %d files", skipped, len(files))

    df = pd.DataFrame(rows, columns=list(SCHEMA)).dropna(subset=["id"])
    df["id"] = df["id"].astype("int64")
    for col in _DATE_COLS:
        df[col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
    logger.info("loaded %d projects from %d files in %s", len(df), len(files), path)
    return df
