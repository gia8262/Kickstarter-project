"""Retrieve the AI-disclosure flag for sampled projects.

Two retrieval paths, both free:

  collect_graphql     Kickstarter's own /graph endpoint. A csrf token read from
                      the homepage, plus the session cookies, answer a structured
                      ``aiDisclosure`` query that returns the flag and the
                      disclosure text without rendering JavaScript. FREE. This is
                      the path that collected the disclosures used in this study.
  probe_free          Fetch the page with a TLS-impersonating request (no browser)
                      and detect the "Use of AI" section in the markup or an
                      inlined JSON payload. FREE. Used for the pilot probe and the
                      run_00 diagnostic.

``find_use_of_ai`` (used by the probe and the diagnostic) parses the HTML with
BeautifulSoup and inspects inlined JSON by parsing it, not by matching raw text;
``AI_JSON_KEYS`` is the set of candidate keys it looks for.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

__all__ = [
    "find_use_of_ai",
    "pairs_from_sample",
    "probe_free",
    "collect_graphql",
    "graphql_session",
    "query_ai_disclosure",
    "AI_JSON_KEYS",
    "GRAPHQL_URL",
    "GRAPHQL_AI_QUERY",
]

#: Candidate keys the disclosure might use inside an embedded JSON blob.
#: Matching is case- and separator-insensitive ("aiDisclosure" == "ai_disclosure").
AI_JSON_KEYS: frozenset[str] = frozenset(
    {"useofai", "aidisclosure", "generatedbyai", "usesai", "aiconsent", "involvesai"}
)
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "section")
_OBJECT_RE = re.compile(r"\{.*\}", re.S)
_NOT_ALNUM_RE = re.compile(r"[^a-z0-9]")


def _norm_key(key: str) -> str:
    """Normalise a JSON key for matching: lowercase, separators stripped."""
    return _NOT_ALNUM_RE.sub("", key.lower())


def _json_contains_key(obj: Any, keys: frozenset[str]) -> bool:
    """Recursively test whether any of ``keys`` appears as a dict key in ``obj``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _norm_key(key) in keys:
                return True
            if _json_contains_key(value, keys):
                return True
    elif isinstance(obj, list):
        return any(_json_contains_key(item, keys) for item in obj)
    return False


def _try_load_json(text: str) -> Any | None:
    """Parse ``text`` as JSON, falling back to the first {...} object it contains."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = _OBJECT_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def find_use_of_ai(html: str) -> tuple[bool, bool, str | None]:
    """Detect the AI disclosure in a project page.

    Returns ``(text_hit, json_hit, snippet)``:
      text_hit  a visible 'Use of AI' heading or section is present
      json_hit  the disclosure appears inside an inlined JSON payload
      snippet   a short sample of that payload, to confirm the real field name
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) embedded JSON payloads (parse the <script> tags before stripping them).
    json_hit = False
    snippet: str | None = None
    for script in soup.find_all("script"):
        raw = script.string or script.get_text()
        if not raw or "ai" not in raw.lower():
            continue
        data = _try_load_json(raw.strip())
        if data is not None and _json_contains_key(data, AI_JSON_KEYS):
            json_hit = True
            snippet = raw.strip()[:400]
            break

    # 2) visible section: a heading whose text is 'Use of AI'.
    text_hit = False
    for tag in soup.find_all(_HEADING_TAGS):
        if "use of ai" in tag.get_text(" ", strip=True).lower():
            text_hit = True
            break

    # 3) fallback: substring over the visible text, with scripts/styles removed.
    if not text_hit:
        for node in soup(["script", "style"]):
            node.extract()
        text_hit = "use of ai" in soup.get_text(" ", strip=True).lower()

    return text_hit, json_hit, snippet


# ---------------------------------------------------------------------------
# Free path B: Kickstarter's own GraphQL API.
# The project page serves a csrf-token in a <meta> tag; with it (and the session
# cookies) the /graph endpoint answers structured queries, including the
# ``aiDisclosure`` object that backs the visible "Use of AI" section. This gives
# the flag AND the disclosure text without rendering JavaScript.
# ---------------------------------------------------------------------------

GRAPHQL_URL = "https://www.kickstarter.com/graph"

#: Disclosure fields requested per project. If the diagnostic shows a field is
#: named differently, fix it here and everything downstream follows.
GRAPHQL_AI_QUERY = """
query AiDisclosure($slug: String!) {
  project(slug: $slug) {
    pid
    state
    aiDisclosure {
      involvesAi
      involvesGeneration
      involvesFunding
      involvesOther
      generatedByAiDetails
      generatedByAiConsent
      otherAiDetails
    }
  }
}
"""

_SLUG_RE = re.compile(r"/projects/([^/?#]+)/([^/?#]+)")


def pairs_from_sample(
    sample: pd.DataFrame, limit: int | None = None, seed: int | None = None
) -> list[tuple[int, str]]:
    """(project_id, url) pairs from the sample frame.

    Rows without a url are dropped. Without ``seed`` the pairs come in frame
    order, truncated to the first ``limit`` (all of them if ``None``). With
    ``seed`` they are a reproducible random draw of ``limit`` rows (the whole
    frame, shuffled, if ``limit`` is ``None``).
    """
    rows = sample[["id", "url"]].dropna(subset=["url"])
    if seed is not None:
        rows = rows.sample(n=len(rows) if limit is None else limit, random_state=seed)
    elif limit is not None:
        rows = rows.head(limit)
    return [(int(pid), str(url)) for pid, url in rows.itertuples(index=False, name=None)]


def slug_from_url(url: str) -> str | None:
    """'https://www.kickstarter.com/projects/creator/name?x=1' -> 'creator/name'."""
    match = _SLUG_RE.search(url)
    return f"{match.group(1)}/{match.group(2)}" if match else None


def graphql_session() -> tuple[Any, str]:  # pragma: no cover - live network
    """Open a curl_cffi session and bootstrap a csrf token from the homepage."""
    from curl_cffi import requests as cf

    session: Any = cf.Session(impersonate="chrome")
    page = session.get("https://www.kickstarter.com/", timeout=30)
    soup = BeautifulSoup(page.text, "html.parser")
    meta = soup.find("meta", attrs={"name": "csrf-token"})
    token = str(meta.get("content", "")) if meta else ""
    if not token:
        raise RuntimeError("could not find csrf-token on the Kickstarter homepage")
    return session, token


#: HTTP statuses worth retrying with backoff (rate limiting and transient
#: server errors). Anything else is terminal: 200 is a success, and other
#: 4xx codes will not change on retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait from a Retry-After header (numeric-seconds form), if present.

    The HTTP-date form is not parsed; callers fall back to exponential backoff.
    """
    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _graphql_record(pid: int, url: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the cache record from a GraphQL result.

    A 200 yields a determined flag (0/1), plus the disclosure object when the
    project carries one. Any non-200 status, or a network error (status None),
    is recorded as a FAILURE: the status is kept and ``ai_disclosed`` is left
    None, never silently 0, so a rate-limited or blocked page is not mistaken
    for a genuinely non-disclosing one.
    """
    status = result.get("http_status")
    body = result.get("body") or {}
    rec: dict[str, Any] = {"id": int(pid), "url": url, "http_status": status}
    if result.get("error"):
        rec["error"] = str(result["error"])[:300]
    elif isinstance(body, dict) and "errors" in body:
        rec["error"] = json.dumps(body["errors"])[:300]
    if status != 200:
        rec["ai_disclosed"] = None
        return rec
    project = (body.get("data") or {}).get("project") or {} if isinstance(body, dict) else {}
    disclosure = project.get("aiDisclosure")
    rec["ai_disclosed"] = int(bool(disclosure and disclosure.get("involvesAi")))
    if disclosure:
        rec["disclosure"] = disclosure
    return rec


def _completed_urls(cache: Path) -> set[str]:
    """URLs already collected with a valid HTTP 200 record (the resume set).

    Only 200s count as done. Failure rows (non-200, ``ai_disclosed`` None) do
    not, so a rerun retries exactly the projects that have not been fetched.
    """
    if not cache.exists():
        return set()
    done: set[str] = set()
    with open(cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("http_status") == 200 and rec.get("url"):
                done.add(str(rec["url"]))
    return done


def query_ai_disclosure(
    session: Any, token: str, slug: str, timeout: float = 30.0
) -> dict[str, Any]:  # pragma: no cover - live network
    """Run the aiDisclosure query for one project slug; return the raw response.

    The returned dict carries ``http_status``, the parsed ``body`` (or the first
    400 chars of a non-JSON error page), and ``retry_after`` from the response
    headers when the server sends one.
    """
    response = session.post(
        GRAPHQL_URL,
        json={"query": GRAPHQL_AI_QUERY, "variables": {"slug": slug}},
        headers={"x-csrf-token": token, "content-type": "application/json"},
        timeout=timeout,
    )
    out: dict[str, Any] = {
        "http_status": response.status_code,
        "retry_after": _parse_retry_after(response.headers),
    }
    try:
        out["body"] = response.json()
    except Exception:  # noqa: BLE001 - body may be an HTML error page
        out["body"] = {"raw": response.text[:400]}
    return out


def _graphql_with_retry(
    session: Any,
    token: str,
    slug: str,
    *,
    max_retries: int = 4,
    backoff_base: float = 2.0,
    max_backoff: float = 120.0,
    timeout: float = 30.0,
) -> dict[str, Any]:  # pragma: no cover - live network
    """Query one slug, retrying retryable statuses and network errors with backoff.

    On a 429 the server's Retry-After header is honoured when present, otherwise
    the wait grows as ``backoff_base * 2**attempt`` (capped at ``max_backoff``)
    with a little jitter. A 200 or a non-retryable status returns immediately.
    """
    result: dict[str, Any] = {"http_status": None}
    for attempt in range(max_retries + 1):
        try:
            result = query_ai_disclosure(session, token, slug, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - network error: back off and retry
            result = {"http_status": None, "error": str(exc)[:200]}
        status = result.get("http_status")
        if status == 200 or (status is not None and status not in _RETRYABLE_STATUS):
            return result
        if attempt < max_retries:
            wait = result.get("retry_after") or min(max_backoff, backoff_base * 2.0**attempt)
            logger.info(
                "  %s on %s; backing off %.1fs (attempt %d/%d)",
                status,
                slug,
                wait,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait + random.uniform(0.0, 1.0))
    return result


def collect_graphql(
    items: Iterable[tuple[int, str]],
    cache: str | Path = "data/derived/ai_flags.jsonl",
    min_delay: float = 2.0,
    max_delay: float = 5.0,
    max_retries: int = 4,
    timeout: float = 30.0,
) -> Path:  # pragma: no cover - live network
    """FREE: collect AI flags via Kickstarter's GraphQL API. Resumable and polite.

    Only HTTP 200 responses are written to ``cache`` (one valid record per id),
    so a project with a 200 already present is skipped on a rerun. Each request
    waits a random ``min_delay``-``max_delay`` seconds; retryable statuses (429
    and 5xx) and network errors are retried with exponential backoff, honouring a
    Retry-After header on 429. A project that still fails after the retries is
    logged to a sibling ``*_failures.jsonl`` with ``ai_disclosed`` left None
    (never 0) and is NOT treated as done, so the next run retries it.
    """
    session, token = graphql_session()
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    failures = cache.with_name(f"{cache.stem}_failures.jsonl")
    done = _completed_urls(cache)
    todo = [(pid, url) for pid, url in items if url and url not in done]
    logger.info(
        "collecting %d projects via GraphQL; %d already have a 200 record", len(todo), len(done)
    )
    fetched = 0
    for i, (pid, url) in enumerate(todo, 1):
        slug = slug_from_url(url)
        if not slug:
            _append(
                failures,
                {
                    "id": int(pid),
                    "url": url,
                    "http_status": None,
                    "ai_disclosed": None,
                    "error": "no slug in url",
                },
            )
            continue
        result = _graphql_with_retry(session, token, slug, max_retries=max_retries, timeout=timeout)
        rec = _graphql_record(pid, url, result)
        if rec.get("http_status") == 200:
            _append(cache, rec)
            fetched += 1
        else:
            _append(failures, rec)
            logger.debug("graphql still failing for %s: status=%s", url, rec.get("http_status"))
        if i % 25 == 0:
            logger.info(
                "  progress: %d processed, %d fetched, %d remaining", i, fetched, len(todo) - i
            )
        time.sleep(random.uniform(min_delay, max_delay))
    logger.info(
        "graphql collection done: %d fetched (200), %d still failing -> %s",
        fetched,
        len(todo) - fetched,
        cache,
    )
    return cache


def _cached_urls(cache: Path) -> set[str]:
    if not cache.exists():
        return set()
    with open(cache) as f:
        return {json.loads(line)["url"] for line in f if line.strip()}


def _append(cache: Path, rec: dict[str, Any]) -> None:
    with open(cache, "a", encoding="utf-8") as out:
        out.write(json.dumps(rec) + "\n")


def probe_free(
    items: Iterable[tuple[int, str]],
    cache: str | Path = "data/derived/probe.jsonl",
    min_delay: float = 3.0,
    max_delay: float = 6.0,
) -> Path:  # pragma: no cover - requires live network
    """Fetch pages with a browser-impersonating request (no headless browser).

    ``items`` are (project_id, url) pairs. Each output record carries the project
    id, so results join back to the frame on ``id`` with no ambiguity. Reruns skip
    urls already done.
    """
    from curl_cffi import requests as cf  # lazy import so the module loads without it

    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = _cached_urls(cache)
    todo = [(pid, url) for pid, url in items if url and url not in done]
    logger.info("probing %d urls (free); %d already cached", len(todo), len(done))
    for i, (pid, url) in enumerate(todo, 1):
        rec: dict[str, Any] = {"id": int(pid), "url": url}
        try:
            response = cf.get(url, impersonate="chrome", timeout=30)
            text_hit, json_hit, _ = find_use_of_ai(response.text)
            rec.update(
                http_status=response.status_code,
                text_hit=text_hit,
                json_hit=json_hit,
                ai_disclosed=int(text_hit or json_hit),
                blocked=response.status_code in (403, 429, 503)
                or "just a moment" in response.text.lower(),
            )
        except Exception as exc:  # noqa: BLE001 - per-URL resilience; log and continue
            rec["error"] = str(exc)[:200]
            logger.debug("probe failed for %s: %s", url, exc)
        _append(cache, rec)
        if i % 25 == 0:
            logger.info("  probed %d/%d", i, len(todo))
        time.sleep(random.uniform(min_delay, max_delay))
    return cache
