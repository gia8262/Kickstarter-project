"""Tests for ksai.scrape.find_use_of_ai."""

from __future__ import annotations

import pandas as pd

from ksai import scrape


def test_visible_section_detected():
    html = "<html><body><h3>Use of AI</h3><p>Midjourney for art.</p></body></html>"
    text_hit, json_hit, _ = scrape.find_use_of_ai(html)
    assert text_hit is True
    assert json_hit is False


def test_no_section():
    html = "<html><body><h3>Risks and challenges</h3><p>None.</p></body></html>"
    text_hit, json_hit, snippet = scrape.find_use_of_ai(html)
    assert text_hit is False
    assert json_hit is False
    assert snippet is None


def test_embedded_json_detected():
    html = '<html><script>window.__d={"use_of_ai": "images generated"}</script></html>'
    text_hit, json_hit, snippet = scrape.find_use_of_ai(html)
    assert json_hit is True
    assert snippet and "use_of_ai" in snippet


def test_case_insensitive():
    html = "<div>USE OF AI: yes</div>"
    assert scrape.find_use_of_ai(html)[0] is True


def test_heading_nested_markup():
    html = "<section><div><h2><span>Use</span> of AI</h2></div><p>tool X</p></section>"
    text_hit, json_hit, _ = scrape.find_use_of_ai(html)
    assert text_hit is True
    assert json_hit is False


def test_application_json_script_block():
    html = (
        '<script type="application/json">'
        '{"project": {"ai_disclosure": {"text": "we used AI"}}}'
        "</script>"
    )
    text_hit, json_hit, snippet = scrape.find_use_of_ai(html)
    assert json_hit is True
    assert snippet is not None


def test_script_text_does_not_trigger_text_hit():
    # 'use_of_ai' (underscores) in a script must not be read as the visible section
    html = '<html><script>var x={"use_of_ai": 1}</script><body>Hello</body></html>'
    text_hit, json_hit, _ = scrape.find_use_of_ai(html)
    assert text_hit is False
    assert json_hit is True


def test_cache_roundtrip(tmp_path):
    cache = tmp_path / "c.jsonl"
    assert scrape._cached_urls(cache) == set()
    scrape._append(cache, {"url": "u1", "ai_disclosed": 1})
    scrape._append(cache, {"url": "u2"})
    assert scrape._cached_urls(cache) == {"u1", "u2"}


def test_json_hit_matches_camelcase_aidisclosure():
    html = """<html><body><script type="application/json">
    {"project": {"aiDisclosure": {"involvesAi": true}}}
    </script></body></html>"""
    text_hit, json_hit, _ = scrape.find_use_of_ai(html)
    assert json_hit is True


def test_slug_from_url():
    url = "https://www.kickstarter.com/projects/strongholdgames/more-terraforming-mars?ref=x"
    assert scrape.slug_from_url(url) == "strongholdgames/more-terraforming-mars"
    assert scrape.slug_from_url("https://example.com/nope") is None


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 22, 33, 44],
            "url": ["https://k/a/p1", None, "https://k/b/p3", "https://k/c/p4"],
            "category": ["Games", "Art", "Games", "Food"],
        }
    )


def test_pairs_from_sample_drops_missing_urls_and_keeps_order():
    pairs = scrape.pairs_from_sample(_sample_frame())
    assert pairs == [(11, "https://k/a/p1"), (33, "https://k/b/p3"), (44, "https://k/c/p4")]
    assert all(isinstance(pid, int) and isinstance(url, str) for pid, url in pairs)


def test_pairs_from_sample_limit():
    assert scrape.pairs_from_sample(_sample_frame(), limit=2) == [
        (11, "https://k/a/p1"),
        (33, "https://k/b/p3"),
    ]
    assert scrape.pairs_from_sample(_sample_frame(), limit=0) == []


def test_pairs_from_sample_default_is_unlimited():
    assert len(scrape.pairs_from_sample(_sample_frame(), limit=None)) == 3


def test_pairs_from_sample_seeded_draw_is_reproducible():
    frame = pd.DataFrame({"id": range(100), "url": [f"https://k/c/p{i}" for i in range(100)]})
    first = scrape.pairs_from_sample(frame, limit=10, seed=42)
    second = scrape.pairs_from_sample(frame, limit=10, seed=42)
    assert first == second
    assert len(first) == 10
    assert set(first) <= set(scrape.pairs_from_sample(frame))
    # a different seed gives a different draw
    assert first != scrape.pairs_from_sample(frame, limit=10, seed=7)
    # and the draw is not just the head slice
    assert first != scrape.pairs_from_sample(frame, limit=10)


def test_pairs_from_sample_seed_without_limit_shuffles_all():
    frame = pd.DataFrame({"id": range(20), "url": [f"https://k/c/p{i}" for i in range(20)]})
    shuffled = scrape.pairs_from_sample(frame, seed=42)
    assert len(shuffled) == 20
    assert sorted(shuffled) == sorted(scrape.pairs_from_sample(frame))


# --- GraphQL record building and resume logic ----------


def test_graphql_record_200_disclosing():
    result = {
        "http_status": 200,
        "body": {"data": {"project": {"aiDisclosure": {"involvesAi": True}}}},
    }
    rec = scrape._graphql_record(7, "https://k/a/p", result)
    assert rec == {
        "id": 7,
        "url": "https://k/a/p",
        "http_status": 200,
        "ai_disclosed": 1,
        "disclosure": {"involvesAi": True},
    }


def test_graphql_record_200_not_disclosing_has_zero_flag():
    # a 200 that genuinely carries no AI disclosure is a real 0, not a failure
    result = {"http_status": 200, "body": {"data": {"project": {"aiDisclosure": None}}}}
    rec = scrape._graphql_record(8, "https://k/a/p", result)
    assert rec["ai_disclosed"] == 0
    assert "disclosure" not in rec


def test_graphql_record_429_is_unknown_not_zero():
    # a rate-limited page is recorded as unknown, not as a non-disclosure
    result = {"http_status": 429, "body": {"raw": "<html>too many requests</html>"}}
    rec = scrape._graphql_record(9, "https://k/a/p", result)
    assert rec["http_status"] == 429
    assert rec["ai_disclosed"] is None
    assert "disclosure" not in rec


def test_graphql_record_network_error_is_unknown():
    rec = scrape._graphql_record(10, "https://k/a/p", {"http_status": None, "error": "timeout"})
    assert rec["ai_disclosed"] is None
    assert rec["http_status"] is None
    assert rec["error"] == "timeout"


def test_graphql_record_captures_graphql_errors_on_200():
    result = {"http_status": 200, "body": {"errors": [{"message": "bad slug"}]}}
    rec = scrape._graphql_record(11, "https://k/a/p", result)
    assert "bad slug" in rec["error"]
    assert rec["ai_disclosed"] == 0


def test_completed_urls_counts_only_200(tmp_path):
    cache = tmp_path / "ai_flags.jsonl"
    scrape._append(cache, {"id": 1, "url": "u1", "http_status": 200, "ai_disclosed": 1})
    scrape._append(cache, {"id": 2, "url": "u2", "http_status": 200, "ai_disclosed": 0})
    scrape._append(cache, {"id": 3, "url": "u3", "http_status": 429, "ai_disclosed": None})
    # only the two 200s are "done"; the 429 stays pending so a rerun retries it
    assert scrape._completed_urls(cache) == {"u1", "u2"}
    assert scrape._completed_urls(tmp_path / "missing.jsonl") == set()


def test_parse_retry_after():
    assert scrape._parse_retry_after({"Retry-After": "12"}) == 12.0
    assert scrape._parse_retry_after({"retry-after": "0"}) == 0.0
    assert scrape._parse_retry_after({}) is None
    assert scrape._parse_retry_after({"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}) is None
