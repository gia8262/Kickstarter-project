"""Tests for ksai.sampling."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ksai import sampling


def test_wilson_ci_midpoint():
    p, lo, hi = sampling.wilson_ci(50, 100)
    assert p == pytest.approx(0.5)
    assert lo == pytest.approx(0.404, abs=0.005)
    assert hi == pytest.approx(0.596, abs=0.005)
    assert lo < p < hi


def test_wilson_ci_zero_and_empty():
    p, lo, hi = sampling.wilson_ci(0, 100)
    assert p == 0.0 and lo == 0.0 and hi > 0.0
    assert sampling.wilson_ci(0, 0) == (0.0, 0.0, 0.0)


def test_n_for_proportion():
    # p=0.5, e=0.01, z~1.96 -> ~9604
    assert sampling.n_for_proportion(0.5, 0.01) == 9604


def test_total_for_target_ai():
    assert sampling.total_for_target_ai(400, 0.04) == 10000
    assert sampling.total_for_target_ai(400, 0.0) == math.inf


def test_stratified_sample_shape_and_uniqueness():
    frame = pd.DataFrame(
        {
            "category": (["a"] * 600) + (["b"] * 300) + (["c"] * 100),
            "x": range(1000),
        }
    )
    s = sampling.stratified_sample(frame, "category", n=200, seed=1, min_per_stratum=10)
    # all strata represented
    assert set(s["category"].unique()) == {"a", "b", "c"}
    # no duplicates drawn
    assert s["x"].is_unique
    # size is close to n
    assert 180 <= len(s) <= 230
    # floor respected for the smallest stratum
    assert (s["category"] == "c").sum() >= 10


def test_stratified_sample_missing_column():
    with pytest.raises(KeyError):
        sampling.stratified_sample(pd.DataFrame({"a": [1]}), "category", n=1)
