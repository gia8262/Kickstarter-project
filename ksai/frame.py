"""Build the analysis frame and attach the inclusion lenses.

frame      = completed projects launched on/after the policy date, one row per project
region     = English-majority vs not (deterministic, from project country)
experience = first-time vs repeat (from full-archive creator history)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_TERMINAL_STATES = ("successful", "failed")


def _require(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"frame is missing required columns: {missing}")


def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per project id: the latest observed snapshot."""
    _require(df, ("id", "state_changed_at"))
    return (
        df.sort_values("state_changed_at")
        .drop_duplicates(subset="id", keep="last")
        .reset_index(drop=True)
    )


def build_frame(df: pd.DataFrame, policy_date: str) -> pd.DataFrame:
    """Filter to completed, post-policy projects and derive outcome variables."""
    _require(df, ("state", "launched_at", "pledged_usd", "goal_usd", "backers", "deadline"))
    d = dedup_latest(df)
    d = d[d["state"].isin(_TERMINAL_STATES)].copy()
    d = d[d["launched_at"] >= pd.Timestamp(policy_date, tz="UTC")]
    d["success"] = (d["state"] == "successful").astype(int)
    d["pct_funded"] = d["pledged_usd"] / d["goal_usd"]
    d["pledge_per_backer"] = d["pledged_usd"] / d["backers"].where(d["backers"] > 0)
    d["duration_days"] = (d["deadline"] - d["launched_at"]).dt.days
    d["launch_quarter"] = d["launched_at"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    d = d.reset_index(drop=True)
    logger.info("frame: %d completed post-policy projects", len(d))
    return d


def attach_region(frame: pd.DataFrame, english_majority: Iterable[str]) -> pd.DataFrame:
    """Add a binary english_majority flag from the project country."""
    _require(frame, ("country",))
    frame = frame.copy()
    frame["english_majority"] = frame["country"].isin(set(english_majority)).astype(int)
    return frame


def attach_experience(frame: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Set first_time from a creator's full project history.

    ``history_df`` should be the union of the study-window dumps and the archive,
    so every project in the frame is itself counted. Creators absent from the
    history default to first-time, which is logged.
    """
    _require(frame, ("creator_id",))
    _require(history_df, ("id", "creator_id"))
    unique = history_df.drop_duplicates(subset="id")
    counts = unique.groupby("creator_id")["id"].nunique()
    frame = frame.copy()
    mapped = frame["creator_id"].map(counts)
    n_missing = int(mapped.isna().sum())
    if n_missing:
        logger.warning("%d creators not found in history; defaulting them to first-time", n_missing)
    mapped = mapped.fillna(1)
    frame["creator_total_projects"] = mapped.astype(int)
    frame["first_time"] = (mapped <= 1).astype(int)
    return frame
