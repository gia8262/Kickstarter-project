"""Presentation-quality figures for the dictionary content analysis.

Descriptive charts only, matplotlib only, one chart per file. Built on
``matplotlib.figure.Figure`` directly (no pyplot, no global backend state),
sized to stay readable in a two-column paper, saved as PNG at 150 dpi plus
SVG. Palette is Okabe-Ito (colourblind-safe); no chartjunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "OKABE_ITO",
    "plot_disclosure_penalty",
    "plot_disclosure_uptake",
    "plot_matching_balance",
    "plot_outcome_penalty",
    "plot_output_type_by_region",
    "plot_output_type_distribution",
    "plot_output_type_pie",
    "plot_prevalence_by_category",
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


# --- publication style (serif, larger fonts, light grid; no in-figure title, the
# LaTeX \caption carries it). Used by the thesis-ready figures below. ---

#: rcParams applied (locally, via rc_context) to the publication figures so the
#: global/no-pyplot state of the module is left untouched.
_PUB_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "dejavuserif",
}

#: Human-readable category names (the raw labels use underscores).
_LABEL_NICE: dict[str, str] = {
    "audio_video": "audio / video",
    "functional_product": "functional / product",
    "language_translation": "language / translation",
}


def _nice(label: str) -> str:
    return _LABEL_NICE.get(label, label)


def _pub_axes(
    figsize: tuple[float, float], grid_axis: Literal["both", "x", "y"]
) -> tuple[Any, Any]:
    """A publication-styled Figure/axes: muted spines, ticks, and a light grid."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=DPI)
    ax = fig.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.4")
    ax.tick_params(colors="0.3", length=3)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="0.9", linewidth=0.6)
    return fig, ax


def plot_output_type_distribution(dist: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Horizontal bars of stated output-type shares (% of disclosing projects).

    ``dist`` needs columns label / pct_of_disclosing (label_distribution output).
    Publication-styled; the LaTeX caption supplies the title.
    """
    import matplotlib as mpl

    with mpl.rc_context(_PUB_RC):
        data = dist.sort_values("pct_of_disclosing", ascending=True)
        labels = [_nice(label) for label in data["label"]]
        pct = data["pct_of_disclosing"].to_numpy()
        y = list(range(len(labels)))
        fig, ax = _pub_axes((6.4, 3.6), grid_axis="x")
        ax.barh(y, pct, height=0.72, color=OKABE_ITO[0])
        for yi, value in zip(y, pct, strict=True):
            ax.annotate(
                f"{value:.0f}%",
                (value, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("% of AI-disclosing projects")
        ax.set_xlim(0, max(pct) * 1.12)
        return _save(fig, out_dir, "output_type_distribution")


def plot_output_type_pie(dist: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Pie of the stated output-type composition (share of all stated labels).

    ``dist`` is label_distribution output (label / count / pct_of_disclosing).
    NOTE: output_type is multi-label, so a project can state several types and the
    '% of disclosing projects' shares sum past 100%; a pie cannot show that. This
    pie therefore normalises the label *counts* to 100%, i.e. each type's share of
    all stated labels -- a different denominator from plot_output_type_distribution.
    Provided as an alternative format; the bar chart remains the primary figure.
    """
    data = dist.sort_values("count", ascending=False)
    labels = data["label"].tolist()
    counts = data["count"].to_numpy(dtype=float)
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(labels))]
    fig, ax = _new_axes(figsize=(5.8, 4.0))
    wedges, _texts, autotexts = ax.pie(
        counts,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%",
        pctdistance=0.78,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax.legend(
        wedges, labels, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5)
    )
    ax.set_title("Disclosed types of AI use", fontsize=9)
    ax.set_aspect("equal")
    return _save(fig, out_dir, "output_type_pie")


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
    ax.set_title("AI output types by region", fontsize=9)
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
    ax.set_title("Funding success by AI output type", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out_dir, "success_by_output_type")


def plot_top_tools(freq: pd.DataFrame, out_dir: str | Path, top_n: int = 15) -> list[Path]:
    """Horizontal bars of the most-named AI tools (term_frequencies output).

    Publication-styled; the LaTeX caption supplies the title.
    """
    import matplotlib as mpl

    with mpl.rc_context(_PUB_RC):
        tools = freq[freq["kind"] == "tool"].nlargest(top_n, "n_projects").iloc[::-1]
        counts = tools["n_projects"].to_numpy()
        y = list(range(len(tools)))
        fig, ax = _pub_axes((6.4, 4.4), grid_axis="x")
        ax.barh(y, counts, height=0.74, color=OKABE_ITO[1])
        for yi, value in zip(y, counts, strict=True):
            ax.annotate(
                f"{int(value)}",
                (value, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(tools["term"])
        ax.set_xlabel("projects naming the tool")
        ax.set_xlim(0, max(counts) * 1.10)
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


# --- structural / inclusion figures (written by run_05; sized for single column) ---

#: Display names for the inclusion subgroups, by (column, level).
_GROUP_LABELS: dict[tuple[str, int], str] = {
    ("english_majority", 0): "Outside English-majority",
    ("english_majority", 1): "English-majority",
    ("first_time", 1): "First-time creator",
    ("first_time", 0): "Repeat creator",
}

#: Display order of the four inclusion subgroups (region pair, then experience pair).
_GROUP_ORDER: tuple[tuple[str, int], ...] = (
    ("english_majority", 0),
    ("english_majority", 1),
    ("first_time", 1),
    ("first_time", 0),
)


def _inclusion_rows(region: pd.DataFrame, experience: pd.DataFrame) -> list[tuple[str, Any]]:
    """Pair each (label, row) for the four subgroups in display order."""
    frames = {"english_majority": region, "first_time": experience}
    rows = []
    for col, level in _GROUP_ORDER:
        sub = frames[col]
        rows.append((_GROUP_LABELS[(col, level)], sub[sub[col] == level].iloc[0]))
    return rows


def plot_disclosure_penalty(
    region: pd.DataFrame, experience: pd.DataFrame, out_dir: str | Path
) -> list[Path]:
    """Grouped bars of disclosing vs non-disclosing success rate within each group.

    ``region`` and ``experience`` are ``inclusion_split`` outputs (with the
    ``success_ai`` / ``success_other`` columns). Within each creator group the
    shorter AI-disclosing bar against the taller non-disclosing one is the
    disclosure funding penalty (small for repeat creators, large for first-timers).
    Colours match plot_outcome_penalty for a consistent AI-vs-non-AI reading.
    """
    import matplotlib as mpl

    with mpl.rc_context(_PUB_RC):
        rows = _inclusion_rows(region, experience)
        labels = [name for name, _ in rows]
        ai = [float(r["success_ai"]) * 100.0 for _, r in rows]
        other = [float(r["success_other"]) * 100.0 for _, r in rows]
        y = list(range(len(labels) - 1, -1, -1))  # first label on top
        h = 0.38
        fig, ax = _pub_axes((6.4, 3.6), grid_axis="x")
        bars_ai = ax.barh(
            [yi + h / 2 for yi in y], ai, height=h, color=OKABE_ITO[1], label="AI-disclosing"
        )
        bars_other = ax.barh(
            [yi - h / 2 for yi in y], other, height=h, color=OKABE_ITO[0], label="non-disclosing"
        )
        for bars in (bars_ai, bars_other):
            for bar in bars:
                ax.annotate(
                    f"{bar.get_width():.0f}",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                )
        ax.axhline(1.5, color="0.8", lw=0.8, zorder=0)  # region (top) | experience (bottom)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, len(labels) - 1 + 0.7)
        ax.set_xlabel("funding success rate (%)")
        ax.legend(frameon=False, loc="upper right")
        return _save(fig, out_dir, "disclosure_penalty_by_group")


def plot_disclosure_uptake(
    region: pd.DataFrame, experience: pd.DataFrame, out_dir: str | Path
) -> list[Path]:
    """Disclosure rate per group with Wilson CIs; representation ratio annotated.

    Bars past the dashed overall-rate line are groups that disclose AI more than
    the sample as a whole (representation ratio > 1).
    """
    import matplotlib as mpl

    with mpl.rc_context(_PUB_RC):
        rows = _inclusion_rows(region, experience)
        overall = float(region["ai_n"].sum()) / float(region["n"].sum()) * 100.0
        labels = [name for name, _ in rows]
        rate = [float(r["disclosure_rate"]) * 100.0 for _, r in rows]
        lo = [(float(r["disclosure_rate"]) - float(r["rate_ci_lo"])) * 100.0 for _, r in rows]
        hi = [(float(r["rate_ci_hi"]) - float(r["disclosure_rate"])) * 100.0 for _, r in rows]
        y = list(range(len(labels) - 1, -1, -1))
        fig, ax = _pub_axes((6.4, 3.4), grid_axis="x")
        ax.barh(
            y,
            rate,
            height=0.62,
            color=OKABE_ITO[2],
            xerr=[lo, hi],
            error_kw={"elinewidth": 1, "ecolor": "0.35"},
        )
        ax.axvline(overall, color="0.4", ls="--", lw=1)
        ax.annotate(
            f"overall {overall:.1f}%",
            (overall, max(y) + 0.6),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            color="0.3",
        )
        for yi, rt, h in zip(y, rate, hi, strict=True):
            ax.annotate(
                f"{rt:.1f}%",
                (rt + h, yi),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, max(y) + 1.1)
        ax.set_xlabel("AI-disclosure rate (%)")
        ax.margins(x=0.16)
        return _save(fig, out_dir, "disclosure_uptake_by_group")


def _success_row(outcomes: pd.DataFrame) -> tuple[float, float, float]:
    """Return (AI rate %, other rate %, odds ratio) from a compare_outcomes table."""
    row = outcomes[outcomes["outcome"] == "success"].iloc[0]
    return float(row["median_ai"]) * 100.0, float(row["median_other"]) * 100.0, float(row["effect"])


def plot_outcome_penalty(
    overall: pd.DataFrame, matched: pd.DataFrame, out_dir: str | Path
) -> list[Path]:
    """Grouped bars of funding success, AI-disclosing vs non, full and matched.

    ``overall`` / ``matched`` are ``compare_outcomes`` outputs; the ``success``
    row holds the two group success rates and the odds ratio as its effect. The
    near-identical gap in both panels shows the penalty survives matching.
    """
    ai_f, oth_f, or_f = _success_row(overall)
    ai_m, oth_m, or_m = _success_row(matched)
    groups = ["Full sample", "Matched"]
    ai_vals, oth_vals, ors = [ai_f, ai_m], [oth_f, oth_m], [or_f, or_m]
    x = list(range(len(groups)))
    width = 0.38
    fig, ax = _new_axes(figsize=(5.2, 3.4))
    b1 = ax.bar(
        [i - width / 2 for i in x], ai_vals, width, color=OKABE_ITO[1], label="AI-disclosing"
    )
    b2 = ax.bar(
        [i + width / 2 for i in x], oth_vals, width, color=OKABE_ITO[0], label="non-disclosing"
    )
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for i, orv in zip(x, ors, strict=True):
        ax.annotate(
            f"OR {orv:.2f}",
            (i, max(ai_vals[i], oth_vals[i])),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="0.3",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, max(oth_vals) + 22)
    ax.set_ylabel("funding success rate (%)", fontsize=8)
    ax.set_title("AI-disclosure funding penalty, before and after matching", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    return _save(fig, out_dir, "outcome_penalty_matched")


def plot_matching_balance(balance: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Love plot: absolute SMD per covariate before and after matching.

    ``balance`` is the matching balance table (covariate, smd_before, smd_after).
    The dashed line marks the 0.10 balance threshold (Austin, 2011); all points
    falling left of it after matching is the robustness check.
    """
    nice = {
        "log_goal": "log goal",
        "duration_days": "duration",
        "english_majority": "English-majority",
        "first_time": "first-time",
    }
    data = balance.copy()
    data["before"] = data["smd_before"].abs()
    data["after"] = data["smd_after"].abs()
    data = data.sort_values("before")
    y = list(range(len(data)))
    fig, ax = _new_axes(figsize=(5.2, 2.9))
    for yi, b, a in zip(y, data["before"], data["after"], strict=True):
        ax.plot([b, a], [yi, yi], color="0.7", lw=1.5, zorder=1)
    ax.scatter(data["before"], y, color=OKABE_ITO[5], s=40, zorder=2, label="before matching")
    ax.scatter(data["after"], y, color=OKABE_ITO[2], s=40, zorder=3, label="after matching")
    ax.axvline(0.10, color="0.4", ls="--", lw=1)
    ax.annotate("0.10", (0.10, len(data) - 0.4), ha="center", va="bottom", fontsize=7, color="0.3")
    ax.set_yticks(y)
    ax.set_yticklabels([nice.get(str(c), str(c)) for c in data["covariate"]], fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel("|standardised mean difference|", fontsize=8)
    ax.set_title("Covariate balance before and after matching", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.margins(x=0.10)
    return _save(fig, out_dir, "matching_balance_love")


def plot_prevalence_by_category(prevalence: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Horizontal bars of AI-disclosure rate per category, with Wilson CIs.

    ``prevalence`` is run_05's prevalence_by_category.csv (category, n, ai_n,
    rate, ci_lo, ci_hi). The dashed line marks the overall disclosure rate.
    """
    data = prevalence.sort_values("rate", ascending=True)
    overall = float(data["ai_n"].sum()) / float(data["n"].sum()) * 100.0
    y = list(range(len(data)))
    rate = data["rate"].to_numpy() * 100.0
    lo = (data["rate"] - data["ci_lo"]).to_numpy() * 100.0
    hi = (data["ci_hi"] - data["rate"]).to_numpy() * 100.0
    fig, ax = _new_axes(figsize=(6.5, 4.6))
    ax.barh(
        y,
        rate,
        height=0.72,
        color=OKABE_ITO[0],
        xerr=[lo, hi],
        error_kw={"elinewidth": 1, "ecolor": "0.4"},
    )
    ax.axvline(overall, color="0.4", ls="--", lw=1)
    ax.annotate(
        f"overall {overall:.1f}%",
        (overall, len(data) - 0.4),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.3",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(data["category"], fontsize=8)
    ax.set_xlabel("AI-disclosure rate (%)", fontsize=8)
    ax.set_title("AI-disclosure prevalence by category (95% Wilson CI)", fontsize=9)
    ax.set_ylim(-0.7, len(data) + 0.3)
    return _save(fig, out_dir, "prevalence_by_category")
