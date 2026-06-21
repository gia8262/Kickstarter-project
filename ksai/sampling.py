"""Stratified random sampling and the sample-size arithmetic from the spec."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: z value for a two-sided 95% interval.
Z95: float = 1.959963985


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """95% Wilson score interval for a proportion k/n. Returns ``(p, lo, hi)``."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def n_for_proportion(p: float, e: float, z: float = Z95) -> int:
    """Sample size needed to estimate a proportion ``p`` with +/- ``e`` margin."""
    return math.ceil(z**2 * p * (1 - p) / e**2)


def total_for_target_ai(target_ai: int, prevalence: float) -> float:
    """Total sample needed to expect ``target_ai`` AI-disclosing projects."""
    if prevalence <= 0:
        return math.inf
    return math.ceil(target_ai / prevalence)


def stratified_sample(
    frame: pd.DataFrame,
    strata_col: str,
    n: int,
    seed: int = 42,
    min_per_stratum: int = 0,
) -> pd.DataFrame:
    """Proportional stratified random sample of approximately ``n`` rows.

    Each stratum gets ``round(n * stratum_share)`` rows, with an optional floor so
    small categories still yield analysable counts. Sampling within a stratum is
    simple random without replacement. The returned size is approximately ``n``
    because of rounding and the floor.
    """
    if strata_col not in frame.columns:
        raise KeyError(f"strata column {strata_col!r} not in frame")
    rng = np.random.default_rng(seed)
    total = len(frame)
    parts: list[pd.DataFrame] = []
    for _, grp in frame.groupby(strata_col, dropna=False):
        share = len(grp) / total
        take = max(min_per_stratum, round(n * share))
        take = min(take, len(grp))
        idx = rng.choice(grp.index.values, size=take, replace=False)
        parts.append(frame.loc[idx])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
