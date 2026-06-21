"""Step 4: full AI-flag collection over the whole sample.

Run AFTER the diagnostic (run_00) has shown which path works:
    python run_04_collect.py graphql   # free: Kickstarter's GraphQL API (preferred)
    python run_04_collect.py free      # free: raw-HTML parsing (only if run_00 says it works)

``--limit N`` collects a random N of the sampled projects, drawn after a
shuffle with the fixed study seed (config.RANDOM_SEED), so pilots are
reproducible random subsets, not head slices. The default is the whole sample.

Results stream to data/derived/ai_flags.jsonl, one record per project carrying
its id. On the GraphQL path only successful (HTTP 200) responses are written
there; a rerun skips every project that already has a 200 record and retries the
rest, so interruptions and rate-limited pages are safe to resume. Pages that
still fail after the built-in backoff are logged to ai_flags_failures.jsonl with
the flag left unknown, never recorded as non-disclosing.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ksai import config, scrape

logger = logging.getLogger("run_04")

CACHE = config.DERIVED / "ai_flags.jsonl"


def main(path: str, limit: int | None = None) -> None:
    sample = pd.read_parquet(config.DERIVED / "sample.parquet")
    pairs = scrape.pairs_from_sample(sample, limit, seed=config.RANDOM_SEED)
    logger.info(
        "collecting AI flags for %d sampled projects via the %s path (seed=%d)",
        len(pairs),
        path,
        config.RANDOM_SEED,
    )
    if path == "graphql":
        scrape.collect_graphql(pairs, cache=CACHE)
    elif path == "free":
        scrape.probe_free(pairs, cache=CACHE)
    else:
        raise SystemExit("usage: python run_04_collect.py [graphql|free] [--limit N]")
    logger.info("collection finished -> %s", CACHE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="graphql",
        choices=("graphql", "free"),
        help="collection path (default: graphql)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="collect a seeded random N of the sampled projects (default: all)",
    )
    args = parser.parse_args()
    main(args.path, args.limit)
