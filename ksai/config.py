"""Central configuration: paths and study constants.

Paths are resolved relative to the project root (one level above this package),
so they work regardless of the current working directory once the package is
installed with `pip install -e .`.
"""

from __future__ import annotations

from pathlib import Path

# --- paths ---
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data"
RAW: Path = DATA / "raw"  # Web Robots dumps for the study window (~Oct 2023 -> now)
ARCHIVE: Path = DATA / "archive"  # Web Robots dumps back to ~2014 (creator history only)
DERIVED: Path = DATA / "derived"  # pipeline outputs (frame, sample, flags)
RESULTS: Path = ROOT / "results"  # committed: safe aggregate tables and figures
RESULTS_PRIVATE: Path = DERIVED / "results"  # gitignored: per-project / identifiable outputs

# --- study constants ---
POLICY_DATE: str = "2023-08-29"  # AI disclosure became mandatory
ENGLISH_MAJORITY: frozenset[str] = frozenset({"US", "GB", "CA", "AU", "NZ", "IE"})
TARGET_SAMPLE_SIZE: int = 25_000  # rendered pages for the full run
TARGET_AI_PROJECTS: int = 800  # minimum AI-disclosing projects wanted
PILOT_SIZE: int = 800  # size of the free probe pilot
MIN_PER_STRATUM: int = 50  # floor so small categories stay analysable
RANDOM_SEED: int = 42
STRATA_COL: str = "category"


def ensure_dirs() -> None:
    """Create the data directories if they do not exist. Call this from run scripts."""
    for d in (RAW, ARCHIVE, DERIVED):
        d.mkdir(parents=True, exist_ok=True)
