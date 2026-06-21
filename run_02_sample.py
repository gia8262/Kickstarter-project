"""Step 2 (FREE): draw the stratified random sample to scrape.

Reads data/derived/frame.parquet, writes the sample and a plain URL list.
"""

from __future__ import annotations

import logging

import pandas as pd

from ksai import config, sampling

logger = logging.getLogger("run_02")


def main() -> None:
    fr = pd.read_parquet(config.DERIVED / "frame.parquet")
    sample = sampling.stratified_sample(
        fr,
        strata_col=config.STRATA_COL,
        n=config.TARGET_SAMPLE_SIZE,
        seed=config.RANDOM_SEED,
        min_per_stratum=config.MIN_PER_STRATUM,
    )
    out = config.DERIVED / "sample.parquet"
    sample.to_parquet(out)
    sample[["id", "url"]].to_csv(config.DERIVED / "sample_urls.csv", index=False)

    logger.info(
        "sampled %d projects across %d categories -> %s",
        len(sample),
        sample[config.STRATA_COL].nunique(),
        out,
    )
    logger.info("per-category counts:\n%s", sample[config.STRATA_COL].value_counts())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
