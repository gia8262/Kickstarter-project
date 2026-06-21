"""Descriptive analysis: prevalence, outcome comparisons, and inclusion splits.

All statistics here are associational, matching the cross-sectional design:
prevalences with Wilson intervals, chi-square tests of independence, Mann-Whitney
U for skewed continuous outcomes, odds ratios and Cliff's delta as effect sizes,
and Benjamini-Hochberg control for the family of subgroup comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ksai.sampling import wilson_ci

logger = logging.getLogger(__name__)

__all__ = [
    "OutcomeComparison",
    "benjamini_hochberg",
    "cliffs_delta",
    "compare_outcomes",
    "inclusion_split",
    "odds_ratio",
    "prevalence_by",
]

#: Outcome columns compared between AI-disclosing and other projects.
OUTCOME_COLS: tuple[str, ...] = ("pledged_usd", "pct_funded", "backers", "goal_usd")


def prevalence_by(frame: pd.DataFrame, by: str, flag: str = "ai_disclosed") -> pd.DataFrame:
    """Disclosure prevalence within each level of ``by``, with Wilson 95% CIs."""
    rows = []
    for level, grp in frame.groupby(by, dropna=False):
        k, n = int(grp[flag].sum()), len(grp)
        p, lo, hi = wilson_ci(k, n)
        rows.append({by: level, "n": n, "ai_n": k, "rate": p, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """Odds ratio for a 2x2 table [[a, b], [c, d]], Haldane-corrected on zero cells."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5  # type: ignore[assignment]
    return (a * d) / (b * c)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta between samples ``x`` and ``y`` via the Mann-Whitney U relation.

    Ranges -1..1; positive means values in ``x`` tend to exceed those in ``y``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR procedure; returns per-test rejection decisions."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    k = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    return reject.tolist()


@dataclass(frozen=True)
class OutcomeComparison:
    """One outcome compared between the AI-disclosing and comparison groups.

    For the continuous outcomes ``median_ai`` and ``median_other`` are medians;
    for the binary ``success`` row they hold the group success rates (means),
    since a median of a 0/1 variable is uninformative.
    """

    outcome: str
    n_ai: int
    n_other: int
    median_ai: float
    median_other: float
    statistic: float
    p_value: float
    effect: float  # Cliff's delta for continuous, odds ratio for success
    effect_name: str


def compare_outcomes(frame: pd.DataFrame, flag: str = "ai_disclosed") -> list[OutcomeComparison]:
    """Compare success and the skewed outcomes between groups.

    Success uses a chi-square test with an odds ratio; continuous outcomes use
    Mann-Whitney U with Cliff's delta. All claims are associational.
    """
    ai = frame[frame[flag] == 1]
    other = frame[frame[flag] == 0]
    results: list[OutcomeComparison] = []

    table = np.array(
        [
            [int(ai["success"].sum()), int((1 - ai["success"]).sum())],
            [int(other["success"].sum()), int((1 - other["success"]).sum())],
        ]
    )
    if table.min() >= 0 and len(ai) and len(other):
        chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
        results.append(
            OutcomeComparison(
                outcome="success",
                n_ai=len(ai),
                n_other=len(other),
                median_ai=float(ai["success"].mean()),
                median_other=float(other["success"].mean()),
                statistic=float(chi2),
                p_value=float(p),
                effect=odds_ratio(*table.ravel().tolist()),
                effect_name="odds_ratio",
            )
        )

    for col in OUTCOME_COLS:
        if col not in frame.columns:
            continue
        x, y = ai[col].dropna().to_numpy(), other[col].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            continue
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        results.append(
            OutcomeComparison(
                outcome=col,
                n_ai=len(x),
                n_other=len(y),
                median_ai=float(np.median(x)),
                median_other=float(np.median(y)),
                statistic=float(u),
                p_value=float(p),
                effect=cliffs_delta(x, y),
                effect_name="cliffs_delta",
            )
        )
    return results


def inclusion_split(
    frame: pd.DataFrame, group_col: str, flag: str = "ai_disclosed"
) -> pd.DataFrame:
    """The inclusion layer: per group, the AI-vs-other success gap, plus access.

    Returns one row per group level with the disclosure rate (access), the
    representation ratio, the success rates of both subgroups, and the gap.
    The gap-of-gaps across rows is the heterogeneity finding.
    """
    overall_share = frame.groupby(group_col)[flag].count() / len(frame)
    ai_share = frame[frame[flag] == 1].groupby(group_col)[flag].count() / max(
        int(frame[flag].sum()), 1
    )
    rows = []
    for level, grp in frame.groupby(group_col, dropna=False):
        ai = grp[grp[flag] == 1]
        other = grp[grp[flag] == 0]
        k, n = int(grp[flag].sum()), len(grp)
        rate, lo, hi = wilson_ci(k, n)
        rows.append(
            {
                group_col: level,
                "n": n,
                "ai_n": k,
                "disclosure_rate": rate,
                "rate_ci_lo": lo,
                "rate_ci_hi": hi,
                "representation_ratio": float(
                    ai_share.get(level, 0.0) / overall_share.get(level, np.nan)
                ),
                "success_ai": float(ai["success"].mean()) if len(ai) else np.nan,
                "success_other": float(other["success"].mean()) if len(other) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["success_gap"] = out["success_ai"] - out["success_other"]
    return out
