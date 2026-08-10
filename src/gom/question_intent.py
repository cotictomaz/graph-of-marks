"""Deterministic VQA question parsing for question-guided preprocessing.

The parser intentionally uses a small visual ontology.  Unrestricted lexical
resources such as WordNet produce many non-visual senses and make detector
queries both slow and inaccurate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple


_ALIASES = {
    "people": "person",
    "persons": "person",
    "men": "person",
    "man": "person",
    "women": "person",
    "woman": "person",
    "children": "person",
    "child": "person",
    "kids": "person",
    "kid": "person",
    "dogs": "dog",
    "cats": "cat",
    "giraffes": "giraffe",
    "turtles": "turtle",
    "pigs": "pig",
    "buses": "bus",
    "cars": "car",
    "chairs": "chair",
    "tables": "table",
    "televisions": "tv",
    "television": "tv",
    "monitors": "tv",
    "monitor": "tv",
}

_CATEGORY_MEMBERS = {
    "animal": (
        "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "turtle", "pig", "rabbit", "fish",
        "stuffed animal", "stuffed giraffe", "stuffed turtle", "stuffed pig",
    ),
    "vehicle": (
        "car", "truck", "bus", "motorcycle", "bicycle", "train", "boat",
        "airplane",
    ),
    "furniture": (
        "chair", "couch", "bed", "table", "desk", "bench",
    ),
    "food": (
        "fruit", "vegetable", "pizza", "sandwich", "cake", "donut", "hot dog",
    ),
}

_VISUAL_OBJECTS = {
    # COCO detector ontology.
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
    # Common open-vocabulary anchors used by spatial VQA questions.
    "door", "doorway", "window", "wall", "roof", "building", "house",
    "street", "road", "sidewalk", "bridge", "fence", "table", "desk",
    "counter", "shelf", "cabinet", "plate", "chopsticks", "soup", "food",
    "fruit", "vegetable", "shirt", "pants", "dress", "jacket", "coat", "hat",
    "helmet", "shoe", "sky", "snow", "water", "grass", "floor", "ground",
    "tree", "flower", "sign", "screen", "animal",
}

_COMPOUND_ALIASES = {
    "traffic lights": "traffic light",
    "fire hydrants": "fire hydrant",
    "dining tables": "dining table",
    "hot dogs": "hot dog",
    "tennis rackets": "tennis racket",
    "cell phones": "cell phone",
    "wine glasses": "wine glass",
}

_RELATION_PHRASES = (
    ("in front of", "in_front_of"),
    ("to the left of", "left_of"),
    ("to the right of", "right_of"),
    ("on top of", "on_top_of"),
    ("next to", "next_to"),
    ("adjacent to", "next_to"),
    ("close to", "near"),
    ("far from", "far_from"),
    ("inside of", "inside"),
    ("out of", "outside"),
    ("left of", "left_of"),
    ("right of", "right_of"),
    ("beside", "next_to"),
    ("alongside", "next_to"),
    ("behind", "behind"),
    ("above", "above"),
    ("below", "below"),
    ("under", "below"),
    ("beneath", "below"),
    ("inside", "inside"),
    ("outside", "outside"),
    ("touching", "touching"),
    ("holding", "holding"),
    ("wearing", "wearing"),
    ("near", "near"),
)

_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "could", "did", "do",
    "does", "for", "from", "has", "have", "how", "in", "is", "it", "many",
    "much", "of", "on", "or", "that", "the", "there", "these", "this", "those",
    "to", "what", "when", "where", "which", "who", "why", "with", "would",
    "between", "he", "her", "hers", "him", "his", "its", "she", "their",
    "theirs", "them", "they", "we", "you", "your",
}

_ATTRIBUTE_WORDS = {
    "color", "coloured", "colored", "kind", "type", "number", "name", "doing",
    "laying", "lying", "sitting", "standing", "stuffed",
}

_PERSON_PRONOUNS = {"he", "her", "hers", "him", "his", "she"}


def canonical_object_label(label: str) -> str:
    value = re.sub(r"[_-]+", " ", str(label).strip().lower())
    value = re.sub(r"\s+", " ", value)
    value = _COMPOUND_ALIASES.get(value, _ALIASES.get(value, value))
    if value.endswith("ies") and f"{value[:-3]}y" in _VISUAL_OBJECTS:
        return f"{value[:-3]}y"
    if value.endswith("es") and value[:-2] in _VISUAL_OBJECTS:
        return value[:-2]
    if value.endswith("s") and value[:-1] in _VISUAL_OBJECTS:
        return value[:-1]
    return value


@dataclass(frozen=True)
class QuestionIntent:
    question: str
    question_type: str
    object_terms: FrozenSet[str]
    anchor_terms: FrozenSet[str]
    relation_source_terms: FrozenSet[str]
    relation_anchor_terms: FrozenSet[str]
    target_categories: FrozenSet[str]
    relation_terms: FrozenSet[str]
    detector_queries: Tuple[str, ...]

    @property
    def needs_depth(self) -> bool:
        return bool(self.relation_terms & {"in_front_of", "behind"})


def _question_type(q: str) -> str:
    if re.search(r"\bhow many\b", q):
        return "count"
    if re.search(r"\bwhat colou?r\b", q):
        return "color"
    if re.match(r"^(is|are|does|do|did|can|could|has|have)\b", q):
        return "yes_no"
    if re.search(r"\bwhere\b", q):
        return "spatial"
    if re.search(r"\b(who|what (?:animal|vehicle|food|furniture|kind|type))\b", q):
        return "identity"
    return "open"


def _relations(q: str) -> FrozenSet[str]:
    padded = f" {q} "
    found = set()
    for phrase, relation in _RELATION_PHRASES:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", padded):
            found.add(relation)
    return frozenset(found)


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    output = []
    for value in values:
        value = canonical_object_label(value)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def parse_question_intent(question: str) -> QuestionIntent:
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    if not q:
        return QuestionIntent(
            "", "open", frozenset(), frozenset(), frozenset(), frozenset(),
            frozenset(), frozenset(), (),
        )

    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", q)
    categories = {canonical_object_label(t) for t in tokens if canonical_object_label(t) in _CATEGORY_MEMBERS}

    # Remove matched relation words before extracting concrete object anchors.
    relation_words = {
        word
        for phrase, _ in _RELATION_PHRASES
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q)
        for word in phrase.split()
    }
    anchors = {
        canonical_object_label(token)
        for token in tokens
        if token not in _STOPWORDS
        and token not in _ATTRIBUTE_WORDS
        and token not in relation_words
        and canonical_object_label(token) not in categories
        and canonical_object_label(token) in _VISUAL_OBJECTS
    }
    if _PERSON_PRONOUNS.intersection(tokens):
        anchors.add("person")

    # Keep common compound nouns as detector queries without generating arbitrary
    # n-grams.  These are visual phrases that occur directly in the question.
    compounds = []
    compounds_to_match = {
        **{phrase: phrase for phrase in (
            "traffic light", "fire hydrant", "dining table", "hot dog",
            "tennis racket", "cell phone", "wine glass",
        )},
        **_COMPOUND_ALIASES,
    }
    for phrase, canonical in compounds_to_match.items():
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q):
            compounds.append(canonical)
            anchors.difference_update(phrase.split())
            anchors.add(canonical)

    expanded = [member for category in sorted(categories) for member in _CATEGORY_MEMBERS[category]]
    queries = _ordered_unique([*sorted(anchors), *sorted(categories), *expanded, *compounds])
    object_terms = frozenset(set(anchors) | set(categories) | set(expanded))

    relation_anchors = set()
    relation_sources = set()
    for phrase, _ in _RELATION_PHRASES:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q)
        if not match:
            continue
        if categories:
            relation_sources.update(expanded)
        else:
            prefix_tokens = re.findall(r"[a-z]+", q[:match.start()])
            if _PERSON_PRONOUNS.intersection(prefix_tokens):
                relation_sources.add("person")
            relation_sources.update(
                canonical_object_label(token)
                for token in prefix_tokens
                if token not in _STOPWORDS
                and token not in _ATTRIBUTE_WORDS
                and canonical_object_label(token) not in categories
                and canonical_object_label(token) in _VISUAL_OBJECTS
            )
        tail_tokens = re.findall(r"[a-z]+", q[match.end():])
        for token in tail_tokens:
            canonical = canonical_object_label(token)
            if (
                token not in _STOPWORDS
                and token not in _ATTRIBUTE_WORDS
                and canonical in _VISUAL_OBJECTS
            ):
                relation_anchors.add(canonical)
                break

    return QuestionIntent(
        question=q,
        question_type=_question_type(q),
        object_terms=object_terms,
        anchor_terms=frozenset(anchors),
        relation_source_terms=frozenset(relation_sources),
        relation_anchor_terms=frozenset(relation_anchors),
        target_categories=frozenset(categories),
        relation_terms=_relations(q),
        detector_queries=queries,
    )


__all__ = ["QuestionIntent", "canonical_object_label", "parse_question_intent"]
