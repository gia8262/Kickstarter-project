"""Propensity-score matching robustness layer.

Implements the standard observational-study remedy for confounding by observed
covariates: a logistic propensity model, 1:1 nearest-neighbour matching without
replacement on the logit of the propensity score with a caliper of 0.2 standard
deviations of that logit (Austin, 2011), and covariate balance assessed with
standardised mean differences (|SMD| < 0.10 treated as balanced).

Matching balances only observed covariates; hidden confounding and self-selection
into AI use remain, so matched results are reported as a robustness check beside
the descriptive comparison, not as causal effects.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

logger = logging.getLogger(__name__)

__all__ = ["MatchResult", "estimate_propensity", "match", "smd_table"]

#: A category level needs at least this many units in its rarer flag group to
#: keep its own dummy in the propensity design (events-per-variable logic,
#: Peduzzi et al., 1996); sparser levels pool into "Other" when separation is
#: diagnosed. Affects only the propensity model, never the analysis frame.
MIN_LEVEL_EVENTS: int = 10

#: Default covariates for the propensity model (categoricals are dummy-encoded).
DEFAULT_COVARIATES: tuple[str, ...] = (
    "log_goal",
    "duration_days",
    "english_majority",
    "first_time",
)
DEFAULT_CATEGORICAL: tuple[str, ...] = ("category",)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a matching run."""

    matched: pd.DataFrame  # matched rows, both groups, with 'pair_id'
    n_treated: int
    n_matched: int
    caliper: float
    balance: pd.DataFrame  # SMD before/after per covariate


def _design_matrix(
    frame: pd.DataFrame,
    covariates: tuple[str, ...],
    categorical: tuple[str, ...],
) -> pd.DataFrame:
    parts = [frame[list(covariates)].astype(float)]
    for col in categorical:
        if col in frame.columns:
            parts.append(pd.get_dummies(frame[col], prefix=col, drop_first=True, dtype=float))
    x = pd.concat(parts, axis=1)
    return sm.add_constant(x, has_constant="add")


def _separating_levels(levels: pd.Series, flag: pd.Series) -> list[str]:
    """Levels where the flag does not vary (all 0 or all 1): quasi-separation."""
    rate = flag.groupby(levels).mean()
    return [str(level) for level in rate.index[(rate == 0.0) | (rate == 1.0)]]


def _pool_sparse_levels(
    levels: pd.Series, flag: pd.Series, min_events: int = MIN_LEVEL_EVENTS
) -> pd.Series:
    """Pool levels with under ``min_events`` units in their rarer flag group.

    A level with zero treated (or zero control) units gives its dummy an
    unbounded ML coefficient, so the logit cannot converge; near-zero counts
    make the fit fragile the same way. The pooled "Other" level mixes enough
    of both groups to restore a finite optimum. This rewrites only the column
    handed to the propensity design matrix - the analysis frame and the SMD
    balance table keep the original categories.
    """
    counts = flag.groupby(levels).agg(["sum", "count"])
    minority = pd.concat([counts["sum"], counts["count"] - counts["sum"]], axis=1).min(axis=1)
    sparse = minority.index[minority < min_events]
    return levels.where(~levels.isin(sparse), "Other")


def _fit_logit(y: pd.Series, x: pd.DataFrame, **fit_kw: Any) -> Any:
    """Fit a Logit silently; callers decide what non-convergence means."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return sm.Logit(y, x).fit(disp=0, **fit_kw)


def _converged(fitted: Any) -> bool:
    return bool(fitted.mle_retvals.get("converged", False))


def estimate_propensity(
    frame: pd.DataFrame,
    flag: str = "ai_disclosed",
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
    categorical: tuple[str, ...] = DEFAULT_CATEGORICAL,
) -> pd.Series:
    """Fit a logistic propensity model P(flag=1 | covariates); return scores.

    Convergence ladder: when a categorical level quasi-separates the model
    (the flag does not vary inside it - e.g. a category with zero AI projects),
    sparse levels pool into "Other" before fitting, because no optimizer can
    converge on an unbounded likelihood; the pooling affects the propensity
    design only. An optimizer fallback (BFGS) then covers ordinary slow
    convergence, and a genuine failure still surfaces as a ConvergenceWarning.
    """
    data = frame.copy()
    if "log_goal" in covariates and "log_goal" not in data.columns:
        data["log_goal"] = np.log1p(data["goal_usd"])
    for col in categorical:
        if col not in data.columns:
            continue
        separating = _separating_levels(data[col], data[flag])
        if separating:
            pooled = _pool_sparse_levels(data[col], data[flag])
            merged = sorted(set(data[col].unique()) - set(pooled.unique()))
            logger.info(
                "propensity model: %r quasi-separates on %s (no flag variation); "
                "pooled sparse levels %s into 'Other' for the propensity design only",
                col,
                separating,
                merged,
            )
            data[col] = pooled
    x = _design_matrix(data, covariates, categorical)
    keep = x.notna().all(axis=1) & data[flag].notna()
    y = data.loc[keep, flag].astype(float)
    fitted = _fit_logit(y, x.loc[keep])
    if not _converged(fitted):
        fitted = _fit_logit(y, x.loc[keep], method="bfgs", maxiter=200)
    if not _converged(fitted):
        warnings.warn(
            "propensity logit did not converge after pooling and BFGS",
            ConvergenceWarning,
            stacklevel=2,
        )
    logger.info("propensity model fit on %d rows, %d parameters", int(keep.sum()), x.shape[1])
    scores = pd.Series(np.nan, index=frame.index, name="propensity")
    scores.loc[keep] = fitted.predict(x.loc[keep])
    return scores


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def match(
    frame: pd.DataFrame,
    propensity: pd.Series,
    flag: str = "ai_disclosed",
    caliper_sd: float = 0.2,
    seed: int = 42,
) -> MatchResult:
    """1:1 greedy nearest-neighbour matching without replacement on logit(PS).

    Treated units (flag=1) are matched to the nearest control within the caliper;
    treated units with no control inside the caliper are dropped (and counted).
    """
    data = frame.copy()
    data["propensity"] = propensity
    data = data.dropna(subset=["propensity"])
    data["ps_logit"] = _logit(data["propensity"].to_numpy())

    caliper = caliper_sd * float(data["ps_logit"].std())
    treated = data[data[flag] == 1].sample(frac=1.0, random_state=seed)  # shuffled greedy order
    controls = data[data[flag] == 0].sort_values("ps_logit")
    ctrl_logits = controls["ps_logit"].to_numpy()
    ctrl_index = controls.index.to_numpy()
    used = np.zeros(len(controls), dtype=bool)

    pairs: list[tuple[int, int]] = []
    for t_idx, t_logit in treated["ps_logit"].items():
        pos = int(np.searchsorted(ctrl_logits, t_logit))
        best, best_dist = -1, np.inf
        for p in (pos - 1, pos, pos + 1):
            lo, hi = max(p - 2, 0), min(p + 3, len(ctrl_logits))
            for q in range(lo, hi):
                if used[q]:
                    continue
                d = abs(ctrl_logits[q] - t_logit)
                if d < best_dist:
                    best, best_dist = q, d
        if best >= 0 and best_dist <= caliper:
            used[best] = True
            pairs.append((int(t_idx), int(ctrl_index[best])))  # type: ignore[call-overload]

    matched_rows = []
    for pair_id, (t, c) in enumerate(pairs):
        for idx in (t, c):
            row = data.loc[idx].copy()
            row["pair_id"] = pair_id
            matched_rows.append(row)
    matched = pd.DataFrame(matched_rows) if matched_rows else data.iloc[0:0].copy()

    n_treated = int((data[flag] == 1).sum())
    logger.info(
        "matched %d of %d treated within caliper %.4f (%d dropped)",
        len(pairs),
        n_treated,
        caliper,
        n_treated - len(pairs),
    )
    balance = smd_table(data, matched, flag)
    return MatchResult(
        matched=matched,
        n_treated=n_treated,
        n_matched=len(pairs),
        caliper=caliper,
        balance=balance,
    )


def _smd(t: pd.Series, c: pd.Series) -> float:
    t, c = t.astype(float), c.astype(float)
    pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((t.mean() - c.mean()) / pooled)


def smd_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    flag: str = "ai_disclosed",
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """Standardised mean differences per covariate, before and after matching."""
    rows = []
    for col in covariates:
        if col not in before.columns:
            continue
        smd_before = _smd(before.loc[before[flag] == 1, col], before.loc[before[flag] == 0, col])
        smd_after = (
            _smd(after.loc[after[flag] == 1, col], after.loc[after[flag] == 0, col])
            if len(after)
            else np.nan
        )
        rows.append({"covariate": col, "smd_before": smd_before, "smd_after": smd_after})
    table = pd.DataFrame(rows)
    table["balanced_after"] = table["smd_after"].abs() < 0.10
    return table
