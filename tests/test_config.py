"""Tests for ksai.config."""

from __future__ import annotations

from ksai import config


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path / "raw")
    monkeypatch.setattr(config, "ARCHIVE", tmp_path / "archive")
    monkeypatch.setattr(config, "DERIVED", tmp_path / "derived")
    config.ensure_dirs()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "archive").is_dir()
    assert (tmp_path / "derived").is_dir()


def test_constants_are_sane():
    assert config.POLICY_DATE == "2023-08-29"
    assert "US" in config.ENGLISH_MAJORITY
    assert config.TARGET_SAMPLE_SIZE > config.PILOT_SIZE
    assert config.RANDOM_SEED == 42
