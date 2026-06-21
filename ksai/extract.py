"""Dictionary content analysis of the AI disclosure texts. Descriptive only.

METHOD: dictionary-based content analysis. A fixed, auditable list of terms is
applied to each disclosure to assign it to one or more output-type categories.
The researcher defines what each category captures and has verified the term
list; the method codes manifest content, the AI uses and tool names creators
state explicitly.

Two layers:
  (a) data-driven discovery: ranked frequencies of meaningful terms and named
      AI tools across all disclosure texts, which is how the dictionary was
      grounded in the corpus rather than in intuition alone;
  (b) categorisation: multi-label output-type labels assigned from the stated
      terms via the keyword dictionary at the top of this file, with the matched
      terms recorded per label.

Every dictionary lives here as plain data so the whole pipeline can be audited;
the share of texts no rule matches is reported as the unclassified residual.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ksai.disclosures import TEXT_FIELDS

logger = logging.getLogger(__name__)

__all__ = [
    "LABEL_KEYWORDS",
    "LANGUAGE_MOTIVE_TERMS",
    "OUTPUT_LABELS",
    "TOOL_ALIASES",
    "TOOL_LABELS",
    "UNCLASSIFIED",
    "Extraction",
    "classify",
    "explode_labels",
    "label_crosstab",
    "label_distribution",
    "outcomes_by_label",
    "term_frequencies",
]

#: Output-type labels, multi-label (pipe-separated when serialised).
OUTPUT_LABELS: tuple[str, ...] = (
    "images",
    "text",
    "language_translation",
    "audio_video",
    "functional_product",
)

#: Sentinel label when no keyword rule matches a non-empty text.
UNCLASSIFIED: str = "unclassified"

#: Keyword dictionary: stated terms -> output-type label. Matching is on word
#: boundaries, case-insensitive, against the disclosure text with the [field]
#: markers stripped. Deliberately explicit (full word forms, not stems) so the
#: dictionary is auditable line by line and tunable from the reported
#: unclassified rate.
LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "images": (
        "image",
        "images",
        "imagery",
        "illustration",
        "illustrations",
        "illustrated",
        "art",
        "artwork",
        "artworks",
        "concept art",
        "cover art",
        "drawing",
        "drawings",
        "painting",
        "paintings",
        "portrait",
        "portraits",
        "picture",
        "pictures",
        "photo",
        "photos",
        "photographs",
        "poster",
        "posters",
        "visual",
        "visuals",
        "graphic",
        "graphics",
        "graphic design",
        "icon",
        "icons",
        "sprite",
        "sprites",
        "texture",
        "textures",
        "render",
        "renders",
        "rendering",
        "mockup",
        "mockups",
        "logo",
        "logos",
        # approved corpus-grounded expansion (corpus-grounded scan);
        # es/de terms are forms actually present in the disclosures
        "sketch",
        "sketches",
        "backgrounds",
        "graphical",
        "pin-ups",
        "imágenes",
        "bilder",
        "bildern",
        "hintergründe",
        "portada",
        "visuales",
        "artes",
    ),
    "text": (
        "text",
        "texts",
        "copy",
        "copywriting",
        "description",
        "descriptions",
        "story",
        "stories",
        "storyline",
        "narrative",
        "writing",
        "write",
        "written",
        "wording",
        "blurb",
        "script",
        "scripts",
        "dialogue",
        "lore",
        "wording",
        "drafting",
        "drafts",
        "brainstorm",
        "brainstorming",
        "marketing copy",
        "marketing texts",
        # approved corpus-grounded expansion (corpus-grounded scan)
        "paraphrase",
        "storytelling",
        "writeups",
    ),
    "language_translation": (
        # the language/fluency MOTIVE terms double as this label's keywords
        "translate",
        "translated",
        "translating",
        "translation",
        "translations",
        "proofread",
        "proofreading",
        "proofreader",
        "grammar",
        "grammatical",
        "spelling",
        "spell check",
        "spellcheck",
        "readability",
        "fluency",
        "not my native",
        "not our native",
        "non-native",
        "native language",
        "native speaker",
        "native speakers",
        "english is not",
        "second language",
        # approved corpus-grounded expansion (corpus-grounded scan)
        "rechtschreibung",
    ),
    "audio_video": (
        "music",
        "soundtrack",
        "audio",
        "voice",
        "voices",
        "voiceover",
        "voice-over",
        "voice acting",
        "narration",
        "sound",
        "sounds",
        "sound effects",
        "song",
        "songs",
        "video",
        "videos",
        "animation",
        "animations",
        "animated",
        "trailer",
        "trailers",
        # approved corpus-grounded expansion (corpus-grounded scan)
        "audiobook",
        "audiobooks",
        "narrating",
    ),
    "functional_product": (
        "machine learning",
        "neural network",
        "neural networks",
        "language model",
        "ai model",
        "ai models",
        "llm",
        "llms",
        "chatbot",
        "chatbots",
        "algorithm",
        "algorithms",
        "recommendation engine",
        "image processing",
        "image recognition",
        "speech recognition",
        "object detection",
        "data extraction",
        "data analysis",
        "analytics",
        "automation",
        "ai-powered",
        "ai powered",
        "powered by ai",
        "ai features",
        "ai assistant",
        "ai engine",
        "ai app",
        "ai tool we are building",
        # approved corpus-grounded expansion (corpus-grounded scan)
        "recommendation",
        "recommendations",
        "ai-driven",
        "ai driven",
        "spam",
        "moderations",
        "classifier",
        "retrieval augmented",
        "vector database",
        "npc",
        "ai characters",
    ),
}

#: The language/fluency motive terms: when any of these appears, the
#: language_translation label takes priority over the generic text label
#: (the stated reason is a language barrier, not creative text generation).
LANGUAGE_MOTIVE_TERMS: tuple[str, ...] = LABEL_KEYWORDS["language_translation"]

#: Named AI tools: canonical name -> aliases matched in the text (word-bounded,
#: case-insensitive). Maintained list; the discovery layer also surfaces
#: frequent terms NOT on it, which is how new tools get added here.
TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "MidJourney": ("midjourney", "mid-journey", "mid journey"),
    "ChatGPT": ("chatgpt", "chat gpt", "gpt-4", "gpt-4o", "gpt-3", "gpt-5", "gpt"),
    "DALL-E": ("dall-e", "dalle", "dall e"),
    "Stable Diffusion": ("stable diffusion", "stablediffusion", "sdxl"),
    "Adobe Firefly": ("firefly", "adobe firefly"),
    "Photoshop AI": ("photoshop", "generative fill"),
    "Leonardo AI": ("leonardo",),
    "Ideogram": ("ideogram",),
    "Recraft": ("recraft",),
    "Krea": ("krea",),
    "Fotor": ("fotor",),
    "Canva": ("canva",),
    "NovelAI": ("novelai",),
    "Claude": ("claude",),
    "Gemini": ("gemini", "bard"),
    "Grok": ("grok",),
    "Copilot": ("copilot",),
    "ElevenLabs": ("elevenlabs", "eleven labs"),
    "Suno": ("suno",),
    "Udio": ("udio",),
    "Runway": ("runway", "runwayml"),
    "Sora": ("sora",),
    "Pika": ("pika",),
    "Kling": ("kling",),
    "Synthesia": ("synthesia",),
    "HeyGen": ("heygen",),
    "Whisper": ("whisper",),
    "DeepL": ("deepl",),
    "Grammarly": ("grammarly",),
    "ComfyUI": ("comfyui",),
    # approved corpus-grounded expansion (corpus-grounded scan)
    "DeepSeek": ("deepseek",),
    "VEED": ("veed",),
    "Langchain": ("langchain",),
    "OpenAI": ("openai",),  # general-purpose: discovery only, no output-type label
}

#: Tool -> output-type label, used by the categorisation layer (a named image
#: generator is a stated image use, etc.). Tools without a clear single output
#: type are counted in discovery but assign no label.
TOOL_LABELS: dict[str, str] = {
    "MidJourney": "images",
    "DALL-E": "images",
    "Stable Diffusion": "images",
    "Adobe Firefly": "images",
    "Photoshop AI": "images",
    "Leonardo AI": "images",
    "Ideogram": "images",
    "Recraft": "images",
    "Krea": "images",
    "Fotor": "images",
    "Canva": "images",
    "ChatGPT": "text",
    "Claude": "text",
    "Gemini": "text",
    "Grok": "text",
    "NovelAI": "text",
    "ElevenLabs": "audio_video",
    "Suno": "audio_video",
    "Udio": "audio_video",
    "Runway": "audio_video",
    "Sora": "audio_video",
    "Pika": "audio_video",
    "Kling": "audio_video",
    "Synthesia": "audio_video",
    "HeyGen": "audio_video",
    "Whisper": "audio_video",
    "DeepL": "language_translation",
    "Grammarly": "language_translation",
    # approved corpus-grounded expansion (corpus-grounded scan)
    "DeepSeek": "text",
    "Copilot": "text",
    "VEED": "audio_video",
    "Langchain": "functional_product",
}

#: English stopwords plus boilerplate so the discovery ranking stays readable.
_STOPWORD_BLOCK = """
    a about above after again against all also am an and any are aren't as at be
    because been before being below between both but by can cannot could couldn't
    did didn't do does doesn't doing don't down during each few for from further
    had hadn't has hasn't have haven't having he he'd he'll he's her here here's
    hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't
    it it's its itself let's may me might more most mustn't my myself no nor not
    of off on once only or other ought our ours ourselves out over own same shan't
    she she'd she'll she's should shouldn't so some such than that that's the
    their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we
    we'd we'll we're we've were weren't what what's when when's where where's
    which while who who's whom why why's will with won't would wouldn't you you'd
    you'll you're you've your yours yourself yourselves
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_BLOCK.split())

_MARKER_RE = re.compile("|".join(re.escape(f"[{field}]") for field in TEXT_FIELDS))
_TOKEN_RE = re.compile(r"[a-z][a-z0-9'\-]{2,}")


def _word_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_LABEL_RES: dict[str, re.Pattern[str]] = {
    label: _word_pattern(terms) for label, terms in LABEL_KEYWORDS.items()
}
_TOOL_RES: dict[str, re.Pattern[str]] = {
    tool: _word_pattern(aliases) for tool, aliases in TOOL_ALIASES.items()
}
_ALIAS_WORDS: frozenset[str] = frozenset(
    word for aliases in TOOL_ALIASES.values() for alias in aliases for word in alias.split()
)


def _strip_markers(text: str) -> str:
    return _MARKER_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class Extraction:
    """One project's extracted output-type labels and their evidence."""

    labels: tuple[str, ...]  # OUTPUT_LABELS members, or (UNCLASSIFIED,)
    matched: dict[str, tuple[str, ...]]  # label -> stated terms that matched
    source: str  # "keywords" (text-based) or "flags_only" (no text)

    @property
    def label_str(self) -> str:
        """Pipe-separated label list, the sheet/CSV serialisation."""
        return "|".join(self.labels)


def classify(
    text: str,
    category: str | None = None,
    flags: Mapping[str, object] | None = None,
) -> Extraction:
    """Assign multi-label output types from the stated terms in ``text``.

    Dictionary classification under an assumption of truthful self-report: a label is
    assigned only when its dictionary terms (or a named tool mapped to it) are
    stated in the disclosure, and the matching terms are recorded as evidence.
    The language/fluency motive takes priority over the generic text label.

    Empty texts fall back to the platform's structured flags and the project
    category (``source="flags_only"``): a disclosure whose only content is
    "AI is involved in the funded project / its operation" is classed
    functional_product. Anything still unmatched is UNCLASSIFIED.
    """
    cleaned = _strip_markers(text or "")
    if not cleaned:
        labels: list[str] = []
        flags_hit = bool(flags and (flags.get("involvesFunding") or flags.get("involvesOther")))
        if flags_hit or category == "Technology":
            labels.append("functional_product")
        return Extraction(tuple(labels) or (UNCLASSIFIED,), {}, "flags_only")

    matched: dict[str, tuple[str, ...]] = {}
    for label, pattern in _LABEL_RES.items():
        hits = sorted({hit.lower() for hit in pattern.findall(cleaned)})
        if hits:
            matched[label] = tuple(hits)
    for tool, pattern in _TOOL_RES.items():
        tool_label = TOOL_LABELS.get(tool)
        if tool_label and pattern.search(cleaned):
            matched[tool_label] = tuple(sorted({*matched.get(tool_label, ()), tool}))

    # stated language/fluency motive outranks generic text generation
    if "language_translation" in matched:
        matched.pop("text", None)

    labels = [label for label in OUTPUT_LABELS if label in matched]
    return Extraction(tuple(labels) or (UNCLASSIFIED,), matched, "keywords")


def term_frequencies(texts: Iterable[str], min_projects: int = 2) -> pd.DataFrame:
    """Layer (a): ranked named-tool and term frequencies across ``texts``.

    Returns one row per named tool (kind="tool", canonical name, aliases
    pooled) and per remaining token (kind="term", stopwords and tool-alias
    words excluded), with total mentions, the number of projects mentioning
    it, and that count as a share of all texts. Terms appearing in fewer than
    ``min_projects`` projects are dropped to keep the table readable.
    """
    texts = [_strip_markers(t or "").lower() for t in texts]
    n_texts = len(texts)
    mentions: Counter[tuple[str, str]] = Counter()
    projects: Counter[tuple[str, str]] = Counter()
    for text in texts:
        if not text:
            continue
        for tool, pattern in _TOOL_RES.items():
            hits = pattern.findall(text)
            if hits:
                mentions[("tool", tool)] += len(hits)
                projects[("tool", tool)] += 1
        tokens = [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]
        tokens = [t for t in tokens if t not in _ALIAS_WORDS]
        counts = Counter(tokens)
        for token, count in counts.items():
            mentions[("term", token)] += count
            projects[("term", token)] += 1

    rows = [
        {
            "kind": kind,
            "term": term,
            "n_mentions": mentions[(kind, term)],
            "n_projects": n_projects,
            "pct_projects": 100.0 * n_projects / max(n_texts, 1),
        }
        for (kind, term), n_projects in projects.items()
        if kind == "tool" or n_projects >= min_projects
    ]
    out = pd.DataFrame(rows, columns=["kind", "term", "n_mentions", "n_projects", "pct_projects"])
    return out.sort_values(
        ["n_projects", "n_mentions", "term"], ascending=[False, False, True]
    ).reset_index(drop=True)


def explode_labels(frame: pd.DataFrame, label_col: str = "output_labels") -> pd.DataFrame:
    """One row per (project, label) from the pipe-separated label column."""
    out = frame.copy()
    out[label_col] = out[label_col].str.split("|")
    out = out.explode(label_col).rename(columns={label_col: "label"})
    return out.reset_index(drop=True)


def label_distribution(labelled: pd.DataFrame, total: int) -> pd.DataFrame:
    """Label counts and shares of all ``total`` disclosing projects.

    Multi-label: shares sum to more than 100 by design.
    """
    counts = explode_labels(labelled)["label"].value_counts()
    out = counts.rename_axis("label").reset_index(name="count")
    out["pct_of_disclosing"] = 100.0 * out["count"] / max(total, 1)
    return out


def label_crosstab(labelled: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Label x group counts with row percentages (share of label per group)."""
    rows = explode_labels(labelled)
    table = pd.crosstab(rows["label"], rows[group_col])
    pct = table.div(table.sum(axis=1), axis=0) * 100.0
    out = table.add_prefix(f"{group_col}=").join(pct.add_prefix(f"rowpct_{group_col}=").round(2))
    return out.reset_index()


def outcomes_by_label(merged: pd.DataFrame) -> pd.DataFrame:
    """Funding outcomes per output-type label (descriptive bridge table).

    ``merged`` is the exploded label table joined to the sample outcomes by
    id; one project contributes to every label it stated. All figures are
    descriptive associations, not effects.
    """
    rows = []
    for label, grp in merged.groupby("label"):
        rows.append(
            {
                "label": label,
                "n": len(grp),
                "success_rate": float(grp["success"].mean()),
                "median_pledged_usd": float(np.median(grp["pledged_usd"])),
                "median_pct_funded": float(np.median(grp["pct_funded"])),
                "median_backers": float(np.median(grp["backers"])),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
