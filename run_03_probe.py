"""Step 3 (FREE): pilot probe.

Fetches a few hundred sampled pages with TLS impersonation (no browser, no cost)
to (a) confirm whether the 'Use of AI' section is reachable for free and
(b) estimate the disclosure prevalence that sizes the full run.
"""

from __future__ import annotations

import logging

import pandas as pd

from ksai import config, sampling, scrape

logger = logging.getLogger("run_03")


def main() -> None:
    sample = pd.read_parquet(config.DERIVED / "sample.parquet")
    pilot = sample.sample(n=min(config.PILOT_SIZE, len(sample)), random_state=config.RANDOM_SEED)
    pairs = [
        (int(pid), url)
        for pid, url in pilot[["id", "url"]]
        .dropna(subset=["url"])
        .itertuples(index=False, name=None)
    ]
    cache = scrape.probe_free(pairs)

    res = pd.read_json(cache, lines=True)
    if "id" in res:
        logger.info(
            "each result carries a project id (%d unique), so flags join to the frame on id",
            res["id"].nunique(),
        )
    fetched = res[res["http_status"].eq(200)] if "http_status" in res else res
    n = len(fetched)
    k = int(fetched["ai_disclosed"].sum()) if ("ai_disclosed" in fetched and n) else 0
    blocked = int(res["blocked"].sum()) if "blocked" in res else 0

    p, lo, hi = sampling.wilson_ci(k, n)
    logger.info("fetched OK: %d | blocked: %d", n, blocked)
    logger.info(
        "AI disclosure prevalence: %.1f%% (95%% CI %.1f%%-%.1f%%)", p * 100, lo * 100, hi * 100
    )
    if n:
        need = sampling.total_for_target_ai(config.TARGET_AI_PROJECTS, p)
        logger.info(
            "to expect %d AI projects -> ~%s pages total", config.TARGET_AI_PROJECTS, f"{need:,}"
        )
    if "text_hit" in fetched:
        logger.info("section visible in raw HTML (text_hit): %d", int(fetched["text_hit"].sum()))
    if "json_hit" in fetched:
        logger.info("disclosure in embedded JSON (json_hit): %d", int(fetched["json_hit"].sum()))
    logger.info(
        "decision: if text_hit or json_hit > 0 on AI pages, scale the free path; "
        "if pages are mostly blocked, collect via the free GraphQL path (run_04 graphql)."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
