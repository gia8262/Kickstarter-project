"""Assemble the disclosing-projects table for the dictionary content analysis.

Each AI-disclosing project carries an ``aiDisclosure`` object from the GraphQL
collection. This module turns that object into the disclosure text and loads the
table of disclosing projects (text plus project context from the sample) that the
dictionary classifier in ``ksai.extract`` and the validation consume. There is no
human coding step: the categories are assigned by the keyword dictionary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TEXT_FIELDS", "disclosure_text", "load_disclosing"]

#: Disclosure text fields from the GraphQL ``aiDisclosure`` object, concatenated
#: in this order and tagged with their source, skipping the empty ones.
TEXT_FIELDS: tuple[str, ...] = (
    "generatedByAiDetails",
    "generatedByAiConsent",
    "otherAiDetails",
)

#: Context columns copied from the sample when present.
_CONTEXT_COLS: tuple[str, ...] = (
    "category",
    "english_majority",
    "first_time",
    "launched_at",
    "state",
    "success",
    "pledged_usd",
    "pct_funded",
    "backers",
)


def disclosure_text(disclosure: object) -> str:
    """Concatenate the populated disclosure fields, each tagged with its source."""
    if not isinstance(disclosure, dict):
        return ""
    parts = []
    for field in TEXT_FIELDS:
        value = disclosure.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(f"[{field}] {value.strip()}")
    return "\n\n".join(parts)


def load_disclosing(flags_path: str | Path, sample: pd.DataFrame) -> pd.DataFrame:
    """Table of AI-disclosing projects with disclosure text and project context.

    Reads the collected flags, keeps the records that disclose AI (one per id),
    builds the disclosure text from the ``aiDisclosure`` object, carries the
    ``involvesFunding`` / ``involvesOther`` flags for the empty-text fallback, and
    joins the project context from ``sample`` on ``id``.
    """
    rows: list[dict[str, object]] = []
    with open(flags_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec.get("ai_disclosed") or 0) != 1:
                continue
            disclosure = rec.get("disclosure") or {}
            rows.append(
                {
                    "id": int(rec["id"]),
                    "url": rec.get("url"),
                    "disclosure_text": disclosure_text(disclosure),
                    "involves_funding": bool(disclosure.get("involvesFunding")),
                    "involves_other": bool(disclosure.get("involvesOther")),
                }
            )
    ai = pd.DataFrame(rows).drop_duplicates("id")
    if ai.empty:
        raise ValueError("no AI-disclosing records in the flags file")
    missing = set(ai["id"]) - set(sample["id"])
    if missing:
        raise ValueError(f"{len(missing)} disclosing ids not in the sample: {sorted(missing)[:5]}")
    context = [c for c in _CONTEXT_COLS if c in sample.columns]
    out = ai.merge(sample[["id", *context]], on="id", how="left", validate="one_to_one")
    logger.info("disclosing projects: %d", len(out))
    return out
