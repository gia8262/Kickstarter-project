"""End-to-end integration test for the free-phase data flow.

Generates a tiny synthetic dump, then runs load -> frame -> region -> experience
-> stratified sample, asserting the pipeline holds together and the outputs are sane.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ksai import frame as F
from ksai import sampling, synthetic, webrobots


def test_full_free_phase(tmp_path):
    window = tmp_path / "window.json"
    archive = tmp_path / "archive.json"
    synthetic.write_dump(
        window,
        datetime(2024, 1, 1, tzinfo=UTC),
        months=6,
        per_month=200,
        pid_offset=1_000_000,
        seed=1,
    )
    synthetic.write_dump(
        archive,
        datetime(2019, 1, 1, tzinfo=UTC),
        months=6,
        per_month=200,
        pid_offset=2_000_000,
        seed=2,
    )

    raw = webrobots.load_dir(tmp_path, pattern="window.json")
    assert len(raw) == 1200

    fr = F.build_frame(raw, "2023-08-29")
    fr = F.attach_region(fr, {"US", "GB", "CA", "AU", "NZ", "IE"})

    history = pd.concat(
        [raw, webrobots.load_dir(tmp_path, pattern="archive.json")], ignore_index=True
    )
    fr = F.attach_experience(fr, history)

    # frame is well-formed
    for col in ("success", "english_majority", "first_time", "launch_quarter"):
        assert col in fr.columns
    assert fr["success"].isin([0, 1]).all()
    assert fr["english_majority"].isin([0, 1]).all()
    assert 0.0 < fr["english_majority"].mean() < 1.0  # mixed countries present

    # stratified sample is drawn cleanly
    sample = sampling.stratified_sample(fr, "category", n=300, seed=42, min_per_stratum=5)
    assert sample["id"].is_unique
    assert set(sample["category"]).issubset(set(fr["category"]))
    assert len(sample) <= len(fr)


def test_synthetic_generate(tmp_path):
    from types import SimpleNamespace

    (tmp_path / "raw").mkdir()
    (tmp_path / "archive").mkdir()
    cfg = SimpleNamespace(
        RAW=tmp_path / "raw",
        ARCHIVE=tmp_path / "archive",
        ensure_dirs=lambda: None,
    )
    synthetic.generate(cfg)
    assert (tmp_path / "raw" / "window_fake.json").exists()
    assert (tmp_path / "archive" / "archive_fake.json").exists()
