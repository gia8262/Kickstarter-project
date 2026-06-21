"""Tests for ksai.matching.

The key test builds a synthetic confound (AI use concentrated in big-goal Games
projects) and verifies that matching removes the covariate imbalance (SMD drops
below the 0.10 threshold) while preserving a usable matched sample.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ksai import matching


@pytest.fixture
def confounded_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 1500
    category = rng.choice(["Games", "Art", "Music"], n, p=[0.4, 0.3, 0.3])
    goal = rng.lognormal(8.5, 1.0, n) * np.where(category == "Games", 2.0, 1.0)
    english = (rng.random(n) < 0.75).astype(int)
    first_time = (rng.random(n) < 0.5).astype(int)
    # confound: AI much more likely for big goals and Games
    logits = -2.5 + 0.4 * (np.log1p(goal) - 8.5) + 1.0 * (category == "Games")
    ai = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    success = (rng.random(n) < (0.55 - 0.10 * ai)).astype(int)
    return pd.DataFrame(
        {
            "ai_disclosed": ai,
            "category": category,
            "goal_usd": goal,
            "log_goal": np.log1p(goal),
            "duration_days": rng.choice([30, 45, 60], n).astype(float),
            "english_majority": english,
            "first_time": first_time,
            "success": success,
            "pledged_usd": rng.lognormal(8, 1, n),
            "pct_funded": rng.lognormal(0, 0.5, n),
            "backers": rng.poisson(40, n),
        }
    )


def test_propensity_scores_valid_and_ordered(confounded_frame):
    ps = matching.estimate_propensity(confounded_frame)
    assert ps.notna().all()
    assert ((ps > 0) & (ps < 1)).all()
    # built-in confound: treated units must have higher average propensity
    treated_mean = ps[confounded_frame["ai_disclosed"] == 1].mean()
    control_mean = ps[confounded_frame["ai_disclosed"] == 0].mean()
    assert treated_mean > control_mean


def test_match_balances_known_confound(confounded_frame):
    ps = matching.estimate_propensity(confounded_frame)
    result = matching.match(confounded_frame, ps, seed=3)

    # most treated units find a match
    assert result.n_matched >= 0.7 * result.n_treated
    # matched frame is balanced pairs: two rows per pair
    assert len(result.matched) == 2 * result.n_matched
    assert result.matched.groupby("pair_id").size().eq(2).all()

    # the built-in goal imbalance is visible before and removed after
    row = result.balance.set_index("covariate").loc["log_goal"]
    assert abs(row["smd_before"]) > 0.10
    assert abs(row["smd_after"]) < 0.10


def test_match_no_replacement(confounded_frame):
    ps = matching.estimate_propensity(confounded_frame)
    result = matching.match(confounded_frame, ps, seed=3)
    controls = result.matched[result.matched["ai_disclosed"] == 0]
    assert controls.index.is_unique  # each control used at most once


def test_smd_table_shape(confounded_frame):
    table = matching.smd_table(confounded_frame, confounded_frame.iloc[0:0])
    assert {"covariate", "smd_before", "smd_after", "balanced_after"} <= set(table.columns)


def test_propensity_converges_under_quasi_separation(confounded_frame):
    # a tiny category with zero treated units makes its dummy coefficient
    # diverge (quasi-separation); the model must pool it and converge silently
    frame = confounded_frame.copy()
    block = frame.index[:40]
    frame.loc[block, "category"] = "Dance"
    frame.loc[block, "ai_disclosed"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        ps = matching.estimate_propensity(frame)
    assert ps.notna().all()
    assert ((ps > 0) & (ps < 1)).all()


def test_propensity_pooling_affects_design_only(confounded_frame):
    # the caller's frame keeps its original categories after pooling
    frame = confounded_frame.copy()
    block = frame.index[:40]
    frame.loc[block, "category"] = "Dance"
    frame.loc[block, "ai_disclosed"] = 0
    matching.estimate_propensity(frame)
    assert (frame.loc[block, "category"] == "Dance").all()


def test_pool_sparse_levels_keeps_varying_levels():
    levels = pd.Series(["A"] * 40 + ["B"] * 40 + ["C"] * 12)
    flag = pd.Series([0, 1] * 20 + [0, 1] * 20 + [0] * 12)
    pooled = matching._pool_sparse_levels(levels, flag)
    assert set(pooled.unique()) == {"A", "B", "Other"}
    assert (pooled[levels == "C"] == "Other").all()
