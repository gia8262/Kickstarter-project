"""Generate fake Web Robots-style dumps so the pipeline can run and be tested offline."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

CATS = [
    "games",
    "design",
    "technology",
    "film & video",
    "music",
    "art",
    "publishing",
    "fashion",
    "food",
    "comics",
    "crafts",
    "photography",
    "theater",
    "dance",
    "journalism",
]
# Country mix weighted toward the US, as on the real platform.
COUNTRIES = (
    ["US"] * 70
    + ["GB"] * 8
    + ["CA"] * 6
    + ["AU"] * 4
    + ["DE"] * 4
    + ["FR"] * 3
    + ["JP"] * 2
    + ["NL"] * 2
    + ["BR"] * 1
)


def fake_project(pid: int, launched: datetime, rng: random.Random) -> dict:
    """Build one fake project record in Web Robots' nested shape."""
    goal = rng.choice([1000, 5000, 10000, 25000, 50000])
    funded = rng.random() < 0.42
    pledged = goal * (rng.uniform(1.0, 3.0) if funded else rng.uniform(0.0, 0.95))
    creator = rng.randint(1, 12000)  # large pool -> realistic mix of first-time and repeat
    deadline = launched + timedelta(days=rng.choice([30, 45, 60]))
    return {
        "data": {
            "id": pid,
            "name": f"Project {pid}",
            "blurb": "A fake project for testing.",
            "goal": goal,
            "static_usd_rate": 1.0,
            "usd_pledged": round(pledged, 2),
            "backers_count": max(1, int(pledged / rng.uniform(20, 120))),
            "state": "successful" if funded else "failed",
            "country": rng.choice(COUNTRIES),
            "staff_pick": rng.random() < 0.1,
            "creator": {"id": creator, "name": f"Creator {creator}"},
            "category": {"slug": f"{rng.choice(CATS)}/sub", "name": "Subcat"},
            "urls": {"web": {"project": f"https://www.kickstarter.com/projects/{creator}/p{pid}"}},
            "created_at": int((launched - timedelta(days=20)).timestamp()),
            "launched_at": int(launched.timestamp()),
            "deadline": int(deadline.timestamp()),
            "state_changed_at": int(deadline.timestamp()),
        }
    }


def write_dump(
    path: Path,
    start: datetime,
    months: int,
    per_month: int = 400,
    pid_offset: int = 0,
    seed: int = 0,
) -> int:
    """Write `months * per_month` fake project lines to `path`; return the last id."""
    rng = random.Random(seed)
    pid = pid_offset
    with open(path, "w", encoding="utf-8") as f:
        for m in range(months):
            base = start + timedelta(days=30 * m)
            for _ in range(per_month):
                pid += 1
                launched = base + timedelta(days=rng.randint(0, 27))
                f.write(json.dumps(fake_project(pid, launched, rng)) + "\n")
    return pid


def generate(config: ModuleType) -> None:
    """Write one post-policy window dump and one pre-policy archive dump."""
    config.ensure_dirs()
    write_dump(
        config.RAW / "window_fake.json",
        datetime(2023, 10, 1, tzinfo=UTC),
        months=18,
        pid_offset=1_000_000,
        seed=1,
    )
    write_dump(
        config.ARCHIVE / "archive_fake.json",
        datetime(2018, 1, 1, tzinfo=UTC),
        months=24,
        pid_offset=2_000_000,
        seed=2,
    )
