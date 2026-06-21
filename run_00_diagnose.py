"""Step 0 (diagnostic): decide how the AI flag will be collected.

Three candidate paths, tested in one run against a POSITIVE CONTROL, a project
that verifiably carries the "Use of AI" section (More Terraforming Mars,
launched Sep 2023; its disclosure text is quoted in press coverage):

  A. dump field   does the Web Robots dump already carry an ai-disclosure key?
  B. raw HTML     does curl_cffi's fetched HTML contain the section at all?
  C. GraphQL      does Kickstarter's /graph endpoint answer the aiDisclosure query?

Run:  python run_00_diagnose.py
Paste the full output back; it decides the collection path definitively.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ksai import config, scrape

logger = logging.getLogger("run_00")

POSITIVE_CONTROL = "https://www.kickstarter.com/projects/strongholdgames/more-terraforming-mars"
AI_KEY_HINTS = ("ai", "disclos")


def _collect_keys(obj: Any, found: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                found.add(key)
            _collect_keys(value, found, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:3]:
            _collect_keys(item, found, depth + 1)


def scan_dump_for_ai_keys(max_lines: int = 50000) -> list[str]:
    """Part A: list every distinct JSON key in the dump that hints at AI/disclosure."""
    files = sorted(config.RAW.glob("*.json"))
    if not files:
        logger.warning("A) no dump in data/raw; skipping the field scan")
        return []
    keys: set[str] = set()
    with open(files[0], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                _collect_keys(json.loads(line), keys)
            except json.JSONDecodeError:
                continue
    hits = sorted(k for k in keys if any(h in k.lower() for h in AI_KEY_HINTS))
    # 'available', 'maintained' etc. can false-positive on 'ai'; show them anyway,
    # the eye filters faster than a regex argues.
    logger.info(
        "A) dump scanned (%s, first %d lines): %d distinct keys total",
        files[0].name,
        max_lines,
        len(keys),
    )
    logger.info("A) keys containing 'ai' or 'disclos': %s", hits or "NONE")
    return hits


def fetch_positive_control() -> str | None:
    """Part B: fetch the known-AI page; is 'use of ai' anywhere in the raw bytes?"""
    try:
        from curl_cffi import requests as cf
    except ImportError:
        logger.error("B) curl_cffi not installed; skipping")
        return None
    response = cf.get(POSITIVE_CONTROL, impersonate="chrome", timeout=30)
    html = response.text
    lower = html.lower()
    logger.info("B) positive control fetch: HTTP %s, %d bytes", response.status_code, len(html))
    logger.info("B) 'use of ai' in raw HTML: %s", "use of ai" in lower)
    logger.info("B) 'aidisclosure' in raw HTML: %s", "aidisclosure" in lower)
    text_hit, json_hit, _ = scrape.find_use_of_ai(html)
    logger.info("B) detector on this page: text_hit=%s json_hit=%s", text_hit, json_hit)
    out = Path(config.DERIVED) / "positive_control.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("B) raw HTML saved to %s for inspection", out)
    return html


def probe_graphql() -> None:
    """Part C: bootstrap a csrf token and run the aiDisclosure query on the control."""
    slug = scrape.slug_from_url(POSITIVE_CONTROL)
    try:
        session, token = scrape.graphql_session()
        logger.info("C) csrf token obtained: %s...", token[:12])
    except Exception as exc:  # noqa: BLE001
        logger.error("C) could not bootstrap a GraphQL session: %s", exc)
        return
    assert slug is not None
    result = scrape.query_ai_disclosure(session, token, slug)
    logger.info("C) GraphQL HTTP %s", result["http_status"])
    logger.info("C) GraphQL response:\n%s", json.dumps(result["body"], indent=2)[:2000])


def main() -> None:
    print("=" * 70)
    print("PART A: does the Web Robots dump already carry the flag?")
    print("=" * 70)
    scan_dump_for_ai_keys()

    print("=" * 70)
    print("PART B: is the section in the HTML curl_cffi receives? (positive control)")
    print("=" * 70)
    fetch_positive_control()

    print("=" * 70)
    print("PART C: does the GraphQL aiDisclosure query work?")
    print("=" * 70)
    probe_graphql()

    print("=" * 70)
    print("VERDICT GUIDE")
    print("=" * 70)
    print("A has an ai-disclosure key  -> best case: flag comes from the dump itself.")
    print("B says 'use of ai' True     -> raw-HTML path works after all; scale run_04 free.")
    print("C returns involvesAi: true  -> GraphQL path works; scale run_04 graphql.")
    print("all three fail              -> no free path retrieves it; revisit the approach.")
    print("Paste this whole output back.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
