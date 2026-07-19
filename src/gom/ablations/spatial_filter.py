"""Question-level dataset filters for the ablation runner.

Currently provides a single filter — ``spatial`` — that keeps only the VQA
examples whose *question* asks about spatial configuration (relative position,
orientation, direction, containment). It is meant for building a
spatial-reasoning subset so a VLM comparison measures spatial ability rather
than generic VQA accuracy.

Design: the detector is a transparent, dependency-free lexicon match on the
question text (word-boundary, case-insensitive). It is deliberately
*high-precision* rather than exhaustive:

  * Ubiquitous, weakly-spatial words (``on``, ``in``, ``at``, ``by``) are
    **excluded** — on their own they are almost never a spatial cue — in
    favour of the unambiguous multi-word phrases below (``on top of``,
    ``to the left of`` …), which ARE matched.
  * Single-word cues are matched on word boundaries so ``left`` does not fire
    inside ``cleft`` and ``over`` does not fire inside ``discover``.

It is a heuristic, not a classifier, so a few false positives are expected and
accepted (e.g. "what is *left* on the plate?" where ``left`` means *remaining*,
or "is that *right*?" meaning *correct*). The lexicon is overridable from the
config (``keywords`` to replace, ``extra_keywords`` to extend) for anyone who
wants to tighten or widen it. See ``main.py`` for the wiring and the
``dataset_filter`` config block.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Pattern, Sequence

from gom.vqa.runner import VQAExample

# High-precision spatial cues. Multi-word entries are matched as phrases
# (whitespace-flexible); single words are matched on word boundaries. Kept as a
# flat, alphabetically-grouped tuple so it is easy to read and extend.
DEFAULT_SPATIAL_TERMS: tuple[str, ...] = (
    # --- relative-position phrases (preferred over ambiguous bare prepositions) ---
    "on top of", "in front of", "next to", "close to", "far from",
    "to the left", "to the right", "left of", "right of",
    "in the middle", "in the center", "in the centre", "in the corner",
    # --- single-word position cues ---
    "left", "right", "above", "below", "under", "underneath", "beneath",
    "behind", "beside", "between", "among", "amongst", "near", "nearby",
    "nearest", "closest", "farthest", "furthest", "adjacent", "opposite",
    "against", "atop", "top", "bottom", "middle", "center", "centre",
    "corner", "edge", "inside", "outside", "surrounding", "around", "beyond",
    "front", "back", "side", "upper", "lower",
    # --- orientation / direction ---
    "vertical", "vertically", "horizontal", "horizontally", "upside",
    "direction", "facing", "leftmost", "rightmost", "topmost",
    "north", "south", "east", "west",
    # --- spatial interrogatives ---
    "where",
)


def build_spatial_pattern(
    keywords: Optional[Sequence[str]] = None,
    extra_keywords: Optional[Iterable[str]] = None,
) -> Pattern[str]:
    """Compile the spatial-lexicon into a single case-insensitive regex.

    ``keywords`` REPLACES the default lexicon when given; ``extra_keywords`` is
    APPENDED to whichever base lexicon is in effect. Blank/empty terms are
    ignored. Raises ``ValueError`` if the resulting lexicon is empty (an empty
    pattern would match everything, silently disabling the filter).
    """
    terms = list(keywords) if keywords else list(DEFAULT_SPATIAL_TERMS)
    if extra_keywords:
        terms += list(extra_keywords)

    parts: list[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        term = term.strip().lower()
        if not term:
            continue
        if " " in term:
            # Phrase: escape each word, join with flexible whitespace, and
            # anchor the whole phrase on word boundaries.
            inner = r"\s+".join(re.escape(word) for word in term.split())
            parts.append(rf"\b{inner}\b")
        else:
            parts.append(rf"\b{re.escape(term)}\b")

    if not parts:
        raise ValueError(
            "Spatial lexicon is empty after normalization — refusing to build a "
            "pattern that would match every question. Check dataset_filter.keywords."
        )
    return re.compile("|".join(parts), re.IGNORECASE)


def is_spatial_question(question: Optional[str], pattern: Pattern[str]) -> bool:
    """True if ``question`` contains at least one spatial cue from ``pattern``."""
    if not question:
        return False
    return pattern.search(question) is not None


def filter_spatial_examples(
    examples: Sequence[VQAExample],
    keywords: Optional[Sequence[str]] = None,
    extra_keywords: Optional[Iterable[str]] = None,
) -> list[VQAExample]:
    """Return only the examples whose question is spatial, order preserved.

    Order preservation matters: the caller subsamples ``num_images`` /
    ``questions_per_image`` in first-seen order right after this, so keeping the
    original order keeps that subsample (and cross-run comparability)
    deterministic.
    """
    pattern = build_spatial_pattern(keywords, extra_keywords)
    return [ex for ex in examples if is_spatial_question(ex.question, pattern)]
