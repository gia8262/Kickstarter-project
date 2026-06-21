"""Step 1 (FREE): build the analysis frame with the region and experience lenses.

Reads Web Robots dumps from data/raw (study window) and data/archive (history),
writes data/derived/frame.parquet.
"""

from __future__ import annotations

import logging

import pandas as pd

from ksai import config, webrobots
from ksai import frame as F

logger = logging.getLogger("run_01")


def main() -> None:
    config.ensure_dirs()

    raw = webrobots.load_dir(config.RAW)
    fr = F.build_frame(raw, config.POLICY_DATE)
    fr = F.attach_region(fr, config.ENGLISH_MAJORITY)

    try:
        archive = webrobots.load_dir(config.ARCHIVE)
        history = pd.concat([raw, archive], ignore_index=True)
    except FileNotFoundError:
        logger.warning("no archive found; using the window only (experience will be biased)")
        history = raw
    fr = F.attach_experience(fr, history)

    out = config.DERIVED / "frame.parquet"
    fr.to_parquet(out)

    logger.info("frame written to %s", out)
    logger.info("region share:\n%s", fr["english_majority"].value_counts(normalize=True).round(3))
    logger.info("first-time share: %.3f", fr["first_time"].mean())
    logger.info("overall success rate: %.3f", fr["success"].mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
