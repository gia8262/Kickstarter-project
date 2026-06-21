"""Presentation-quality figures for the dictionary content analysis.

Descriptive charts only, matplotlib only, one chart per file. Built on
``matplotlib.figure.Figure`` directly (no pyplot, no global backend state),
sized to stay readable in a two-column paper, saved as PNG at 150 dpi plus
SVG. Palette is Okabe-Ito (colourblind-safe); no chartjunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "OKABE_ITO",
    "plot_output_type_by_region",
    "plot_output_type_distribution",
    "plot_prevalence_by_quarter",
    "plot_success_by_output_type",
    "plot_top_tools",
]

#: Okabe & Ito (2008) colourblind-safe palette.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)

DPI: int = 150
FIGSIZE: tuple[float, float] = (4.8, 3.2)


def _new_axes(figsize: tuple[float, float] = FIGSIZE) -> tuple[Any, Any]:
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=DPI)
    ax = fig.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    return fig, ax


def _save(fig: Any, out_dir: str | Path, stem: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [out / f"{stem}.png", out / f"{stem}.svg"]
    for path in paths:
        fig.savefig(path, dpi=DPI)
    logger.info("figure written: %s (+ svg)", paths[0])
    return paths


def plot_output_type_distribution(dist: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Bar chart of stated output-type shares (% of disclosing projects).

    ``dist`` needs columns label / pct_of_disclosing (label_distribution output).
    """
    fig, ax = _new_axes()
    data = dist.sort_values("pct_of_disclosing", ascending=False)
    ax.bar(data["label"], data["pct_of_disclosing"], color=OKABE_ITO[0])
    ax.set_ylabel("% of AI-disclosing projects", fontsize=8)
    ax.set_title("Stated AI output types (multi-label, keyword-extracted)", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out_dir, "output_type_distribution")


def plot_output_type_by_region(crosstab: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Grouped bars: within each region group, share of its projects per label.

    ``crosstab`` is label_crosstab(..., "english_majority") output: a label
    column plus english_majority=<level> count columns.
    """
    count_cols = [c for c in crosstab.columns if c.startswith("english_majority=")]
    shares = crosstab[count_cols].div(crosstab[count_cols].sum(axis=0), axis=1) * 100.0
    fig, ax = _new_axes()
    width = 0.8 / len(count_cols)
    names = {"english_majority=0": "non-English-majority", "english_majority=1": "English-majority"}
    for i, col in enumerate(count_cols):
        offsets = [j + i * width for j in range(len(crosstab))]
        ax.bar(offsets, shares[col], width=width, color=OKABE_ITO[i], label=names.get(col, col))
    ax.set_xticks([j + width * (len(count_cols) - 1) / 2 for j in range(len(crosstab))])
    ax.set_xticklabels(crosstab["label"], rotation=30, fontsize=8)
    ax.set_ylabel("% of region's disclosing projects", fontsize=8)
    ax.set_title("Stated AI output types by platform-recorded region", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, out_dir, "output_type_by_region")


def plot_success_by_output_type(outcomes: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Success rate per stated output type, with N labelled on each bar.

    ``outcomes`` is outcomes_by_label output (label / n / success_rate).
    """
    fig, ax = _new_axes()
    data = outcomes.sort_values("success_rate", ascending=False)
    bars = ax.bar(data["label"], data["success_rate"] * 100.0, color=OKABE_ITO[2])
    for bar, n in zip(bars, data["n"], strict=True):
        ax.annotate(
            f"N={int(n)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("funding success rate (%)", fontsize=8)
    ax.set_title("Funding success by stated AI output type (descriptive)", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out_dir, "success_by_output_type")


def plot_top_tools(freq: pd.DataFrame, out_dir: str | Path, top_n: int = 15) -> list[Path]:
    """Horizontal bars: most-mentioned named AI tools (term_frequencies output)."""
    tools = freq[freq["kind"] == "tool"].nlargest(top_n, "n_projects").iloc[::-1]
    fig, ax = _new_axes(figsize=(4.8, 3.6))
    ax.barh(tools["term"], tools["n_projects"], color=OKABE_ITO[1])
    ax.set_xlabel("projects naming the tool", fontsize=8)
    ax.set_title("Named AI tools in disclosure texts", fontsize=9)
    return _save(fig, out_dir, "top_tools")


def plot_prevalence_by_quarter(prevalence: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Line chart of disclosure prevalence per launch quarter, with Wilson CI.

    ``prevalence`` is run_05's prevalence_by_quarter.csv (launch_quarter, rate,
    ci_lo, ci_hi).
    """
    data = prevalence.sort_values("launch_quarter")
    fig, ax = _new_axes()
    x = range(len(data))
    ax.fill_between(
        x, data["ci_lo"] * 100.0, data["ci_hi"] * 100.0, color=OKABE_ITO[0], alpha=0.2, lw=0
    )
    ax.plot(x, data["rate"] * 100.0, color=OKABE_ITO[0], marker="o", ms=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["launch_quarter"], rotation=45, fontsize=7)
    ax.set_ylabel("AI-disclosure prevalence (%)", fontsize=8)
    ax.set_title("AI-disclosure prevalence by launch quarter (95% Wilson CI)", fontsize=9)
    return _save(fig, out_dir, "prevalence_by_quarter")
