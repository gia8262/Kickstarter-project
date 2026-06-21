"""Three-way agreement between dictionary, blind-auto, and blind-human labels.

Validation tooling for the dictionary content analysis: the dictionary labels are
compared against two independent blind references (an automated re-labelling
from the text alone, and the author's blind read). Labels are multi-label sets
per project id, serialised pipe-separated. Descriptive agreement only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ksai.extract import OUTPUT_LABELS, UNCLASSIFIED

logger = logging.getLogger(__name__)

__all__ = [
    "ALL_LABELS",
    "cohens_kappa",
    "disagreement_table",
    "exact_match_rate",
    "label_sets",
    "per_label_agreement",
    "run_report",
]

#: Labels compared per-label: the five output types plus the unclassified bin.
ALL_LABELS: tuple[str, ...] = (*OUTPUT_LABELS, UNCLASSIFIED)


def label_sets(frame: pd.DataFrame, column: str) -> dict[int, frozenset[str]]:
    """id -> set of labels from a pipe-separated label column."""
    out: dict[int, frozenset[str]] = {}
    for pid, raw in zip(frame["id"], frame[column], strict=True):
        labels = (
            frozenset(part.strip() for part in str(raw).split("|") if part.strip())
            if pd.notna(raw)
            else frozenset()
        )
        out[int(pid)] = labels
    return out


def exact_match_rate(
    a: Mapping[int, frozenset[str]], b: Mapping[int, frozenset[str]]
) -> tuple[float, int]:
    """Share of common ids whose whole label sets are identical, plus n."""
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan"), 0
    matches = sum(a[i] == b[i] for i in common)
    return matches / len(common), len(common)


def cohens_kappa(x: Sequence[bool], y: Sequence[bool]) -> float:
    """Cohen's kappa for two binary ratings; NaN when chance agreement is 1."""
    if len(x) != len(y) or not x:
        raise ValueError("need two equal-length, non-empty rating vectors")
    n = len(x)
    po = sum(a == b for a, b in zip(x, y, strict=True)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def per_label_agreement(
    a: Mapping[int, frozenset[str]],
    b: Mapping[int, frozenset[str]],
    labels: Sequence[str] = ALL_LABELS,
) -> pd.DataFrame:
    """Binary presence/absence agreement and kappa per label over common ids."""
    common = sorted(set(a) & set(b))
    rows = []
    for label in labels:
        in_a = [label in a[i] for i in common]
        in_b = [label in b[i] for i in common]
        agreement = (
            sum(x == y for x, y in zip(in_a, in_b, strict=True)) / len(common)
            if common
            else float("nan")
        )
        kappa = cohens_kappa(in_a, in_b) if common else float("nan")
        rows.append(
            {
                "label": label,
                "n": len(common),
                "n_a": int(sum(in_a)),
                "n_b": int(sum(in_b)),
                "agreement": agreement,
                "kappa": kappa,
            }
        )
    return pd.DataFrame(rows)


def disagreement_table(sources: Mapping[str, Mapping[int, frozenset[str]]]) -> pd.DataFrame:
    """One row per id where any pair of available sources disagrees."""
    ids = sorted(set.intersection(*(set(s) for s in sources.values())))
    rows = []
    for i in ids:
        sets = {name: source[i] for name, source in sources.items()}
        if len({frozenset(s) for s in sets.values()}) > 1:
            rows.append({"id": i, **{f"{k}_labels": "|".join(sorted(v)) for k, v in sets.items()}})
    return pd.DataFrame(rows)


def run_report(
    sample_csv: str | Path,
    auto_csv: str | Path,
    human_csv: str | Path,
    out_dir: str | Path,
    private_dir: str | Path | None = None,
) -> None:
    """Compute and print the agreement report; write the two report CSVs.

    The safe per-category ``verification_report.csv`` goes to ``out_dir``; the
    per-project ``verification_disagreements.csv`` goes to ``private_dir`` (which
    defaults to ``out_dir`` when not given). Works before the human pass too: if
    ``human_csv`` is missing or has no filled human_labels, prints "human file
    not ready" and reports dictionary-vs-auto only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    private = Path(private_dir) if private_dir is not None else out
    private.mkdir(parents=True, exist_ok=True)
    dictionary = label_sets(pd.read_csv(sample_csv), "output_labels")
    auto = label_sets(pd.read_csv(auto_csv), "auto_labels")
    sources: dict[str, Mapping[int, frozenset[str]]] = {"dictionary": dictionary, "auto": auto}

    human: Mapping[int, frozenset[str]] | None = None
    human_path = Path(human_csv)
    if human_path.exists():
        human_frame = pd.read_csv(human_path)
        if "human_labels" in human_frame.columns:
            filled = label_sets(human_frame.dropna(subset=["human_labels"]), "human_labels")
            if filled:
                human = filled
                sources["human"] = filled
    if human is None:
        print("human file not ready - reporting dictionary-vs-auto only")

    pairs: list[tuple[str, Mapping[int, frozenset[str]], Mapping[int, frozenset[str]]]] = [
        ("dictionary_vs_auto", dictionary, auto)
    ]
    if human is not None:
        pairs += [("dictionary_vs_human", dictionary, human), ("auto_vs_human", auto, human)]

    report_rows = []
    for name, a, b in pairs:
        rate, n = exact_match_rate(a, b)
        print(f"\n== {name}: exact-match (whole label set) {rate:.1%} of {n} projects ==")
        table = per_label_agreement(a, b)
        table.insert(0, "comparison", name)
        print(table.to_string(index=False))
        report_rows.append(table)

    report = pd.concat(report_rows, ignore_index=True)
    report.to_csv(out / "verification_report.csv", index=False)
    disagreements = disagreement_table(sources)
    disagreements.to_csv(private / "verification_disagreements.csv", index=False)
    print(f"\ndisagreement rows: {len(disagreements)} (verification_disagreements.csv)")
    if len(disagreements):
        with pd.option_context("display.max_colwidth", 60):
            print(disagreements.to_string(index=False))
    logger.info("report written to %s", out)
