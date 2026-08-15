#!/usr/bin/env python3
"""Appearance-question filter for the Table 2 subset re-score.

Implements the filter recorded in CLAUDE.md "Known confounds": drop an image's
canonical question when it asks about surface appearance that the mark overlay
destroys — color, material/texture/pattern, text-in-image — or when a color
word appears in the question or in >=50% of the gold answers — plus subjective
questions the image cannot answer ("Have you visited this zoo?"). RefCOCOg is
never filtered (its "question" is a synthetic target-description string).

Stdlib only; runs on the host like the rest of reproduction/.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_question_type():
    # Load the file directly: importing the gom package pulls in torch.
    path = ROOT / "src" / "gom" / "question_intent.py"
    spec = importlib.util.spec_from_file_location("_gom_question_intent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module._question_type


question_type = _load_question_type()

COLOR_WORDS = frozenset(
    """
    black white red green blue yellow orange purple pink brown gray grey
    tan beige cream ivory khaki gold golden silver bronze maroon navy teal
    turquoise violet cyan magenta lavender lilac peach salmon crimson
    scarlet indigo olive mint aqua blond blonde brunette redhead multicolored
    multicolor rainbow
    """.split()
)

_MATERIAL_RE = re.compile(
    r"\b(material|made (?:of|out of|from)|texture|pattern(?:ed)?|striped?|"
    r"plaid|checkered|polka|fabric|wooden|metal(?:lic)?|leather|plastic|"
    r"cloth|denim|wool|cotton|brick|marble)\b"
)
_TEXT_IN_IMAGE_RE = re.compile(
    r"\b(says?|written|writing|letters?|words?|brand|name on|number (?:is )?on|"
    r"spell(?:ed)?)\b"
)
_TOKEN_RE = re.compile(r"[a-z]+")
_SUBJECTIVE_RE = re.compile(
    r"\b(?:have|would) you\b|\bdo you (?!see\b)|\bcould (?:these|this|it) be\b|"
    r"\b(?:does|do) .{0,30}\b(?:like|want|enjoy)\b|\bsame day\b|"
    r"\bdo (?:you|they) think\b"
)


def _has_color_word(text: str) -> bool:
    return any(token in COLOR_WORDS for token in _TOKEN_RE.findall(text.lower()))


def appearance_reason(question: str, answers) -> str | None:
    """Return why this question is appearance-bound, or None to keep it."""
    q = question.lower()
    if question_type(q) == "color":
        return "color_question"
    if _MATERIAL_RE.search(q):
        return "material_texture_pattern"
    if _TEXT_IN_IMAGE_RE.search(q):
        return "text_in_image"
    if _has_color_word(q):
        return "color_word_in_question"
    answers = [a for a in (answers or []) if a]
    if answers and 2 * sum(_has_color_word(a) for a in answers) >= len(answers):
        return "color_word_in_answers"
    if _SUBJECTIVE_RE.search(q):
        return "subjective"
    return None


def keep_ids(rows) -> set:
    """question_ids of rows that survive the filter (rows: canonical eval rows)."""
    kept = set()
    for row in rows:
        answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
        if appearance_reason(row["question"], answers) is None:
            kept.add(row["question_id"])
    return kept


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import first_row_per_image

    for path in map(Path, sys.argv[1:]):
        rows = first_row_per_image(
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        )
        reasons = {}
        for row in rows:
            answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
            reason = appearance_reason(row["question"], answers)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        dropped = sum(reasons.values())
        print(f"{path}: kept {len(rows) - dropped}/{len(rows)}  dropped by reason: {reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
