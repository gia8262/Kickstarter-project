"""Tests for ksai.analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ksai import analysis


@pytest.fixture
def flagged_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 600
    ai = rng.random(n) < 0.2
    english = rng.random(n) < 0.7
    # build in a known pattern: AI hurts success, and more in the non-English group
    base = 0.6 - 0.15 * ai - 0.10 * (ai & ~english)
    success = rng.random(n) < base
    return pd.DataFrame(
        {
            "ai_disclosed": ai.astype(int),
            "english_majority": english.astype(int),
            "success": success.astype(int),
            "pledged_usd": rng.lognormal(8, 1, n) * (1 - 0.3 * ai),
            "pct_funded": rng.lognormal(0, 0.5, n),
            "backers": rng.poisson(50, n),
            "goal_usd": rng.lognormal(8.5, 1, n),
            "category": rng.choice(["Games", "Art", "Tech"], n),
        }
    )


def test_prevalence_by(flagged_frame):
    out = analysis.prevalence_by(flagged_frame, "category")
    assert set(out.columns) >= {"n", "ai_n", "rate", "ci_lo", "ci_hi"}
    assert out["n"].sum() == len(flagged_frame)
    assert ((out["ci_lo"] <= out["rate"]) & (out["rate"] <= out["ci_hi"])).all()


def test_odds_ratio_known_value():
    # 2x2 [[30,70],[10,90]] -> OR = (30*90)/(70*10) = 3.857...
    assert analysis.odds_ratio(30, 70, 10, 90) == pytest.approx(27 / 7)


def test_odds_ratio_zero_cell_haldane():
    assert np.isfinite(analysis.odds_ratio(0, 50, 10, 40))


def test_cliffs_delta_direction_and_bounds():
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = analysis.cliffs_delta(x, y)
    assert d == pytest.approx(1.0)  # complete separation
    assert analysis.cliffs_delta(y, x) == pytest.approx(-1.0)
    assert abs(analysis.cliffs_delta(x, x)) < 0.6


def test_cliffs_delta_empty_is_nan():
    assert np.isnan(analysis.cliffs_delta(np.array([]), np.array([1.0])))


def test_benjamini_hochberg_known_case():
    # classic: smallest p clearly significant, large ones not
    reject = analysis.benjamini_hochberg([0.001, 0.20, 0.04, 0.9], alpha=0.05)
    assert reject[0] is True or reject[0] == True  # noqa: E712 - numpy bool tolerance
    assert not reject[3]
    assert len(reject) == 4


def test_compare_outcomes_detects_built_in_penalty(flagged_frame):
    results = analysis.compare_outcomes(flagged_frame)
    by_name = {r.outcome: r for r in results}
    # success comparison present and direction matches the built-in pattern
    assert "success" in by_name
    assert by_name["success"].median_ai < by_name["success"].median_other
    assert by_name["success"].effect_name == "odds_ratio"
    # pledged comparison present, with Cliff's delta negative (AI pledged lower)
    assert by_name["pledged_usd"].effect_name == "cliffs_delta"
    assert by_name["pledged_usd"].effect < 0


def test_inclusion_split_gap_of_gaps(flagged_frame):
    out = analysis.inclusion_split(flagged_frame, "english_majority")
    assert len(out) == 2
    assert {"disclosure_rate", "representation_ratio", "success_gap"} <= set(out.columns)
    # built-in pattern: penalty larger in the non-English (0) group
    gap_non_english = out.loc[out["english_majority"] == 0, "success_gap"].iloc[0]
    gap_english = out.loc[out["english_majority"] == 1, "success_gap"].iloc[0]
    assert gap_non_english < gap_english
