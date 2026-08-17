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

# Real objects that happen to end in -ing/-ed, exempt from the participle filter.
_PARTICIPLE_OBJECTS = {
    "building", "buildings", "ceiling", "painting", "paintings", "awning",
    "clothing", "railing", "siding", "shed", "sled", "seed", "weed", "bed",
    "hedge", "wing", "wings", "ring", "rings", "string", "swing", "curtaining",
}

# Words that survive the stopword/attribute filters but are not visual objects.
# Only used to keep open-vocabulary detector queries clean; the closed-set
# anchors above are unaffected.
_META_WORDS = {
    # scene/meta nouns
    "photo", "photograph", "image", "picture", "pic", "scene", "shot", "view",
    "side", "sides", "part", "parts", "piece", "top", "bottom", "edge", "corner",
    "background", "foreground", "center", "centre", "middle", "place", "area",
    "position", "row", "line", "group", "pair", "half", "end", "front", "back",
    # question verbs / auxiliaries
    "see", "seen", "appear", "appears", "look", "looks", "looking", "seem",
    "seems", "show", "shows", "shown", "showing", "contain", "contains", "use",
    "used", "using", "made", "make", "makes", "eat", "eats", "eating", "drink",
    "drinking", "ride", "riding", "play", "playing", "watch", "watching",
    "carry", "carrying", "call", "called", "put", "placed", "located", "visible",
    # quantifiers / determiners / adjectives
    "any", "some", "all", "both", "either", "neither", "other", "another",
    "same", "different", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "first", "second", "third", "last", "next", "least",
    "small", "smaller", "smallest", "large", "larger", "largest", "big",
    "bigger", "biggest", "little", "tall", "taller", "short", "shorter", "long",
    "longer", "dark", "darker", "light", "lighter", "bright", "old", "older",
    "young", "younger", "new", "antique", "open", "closed", "empty", "full",
    "clean", "dirty", "wet", "dry", "hot", "cold", "far", "close", "closer",
    "very", "quite", "really", "just", "also", "not", "no", "yes", "than",
    "if", "so", "as", "but", "while", "about", "into", "onto", "off", "out",
    "indoors", "outdoors", "indoor", "outdoor", "daytime", "nighttime",
    "up", "down", "here", "now", "then", "get", "got", "give", "given",
    # bare direction words (relation phrases are stripped separately, but these
    # also occur standalone: "to the left or to the right of ...")
    "left", "right", "above", "below", "under", "over", "behind", "beside",
    "upper", "lower", "leftmost", "rightmost",
}


def canonical_object_label(label: str) -> str:
    value = re.sub(r"[_-]+", " ", str(label).strip().lower())
    value = re.sub(r"\s+", " ", value)
    value = _COMPOUND_ALIASES.get(value, _ALIASES.get(value, value))
    if value.endswith("ies") and f"{value[:-3]}y" in _VISUAL_OBJECTS:
        return f"{value[:-3]}y"
    if value.endswith("ves"):
        for singular in (f"{value[:-3]}f", f"{value[:-3]}fe"):
            if singular in _VISUAL_OBJECTS:
                return singular
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
    open_terms: Tuple[str, ...] = ()

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

    # Open-vocabulary queries: content nouns the closed visual ontology does not
    # know ("cheeseburger", "van", "towel", "guitar"). Without these the object a
    # question is *about* is often never detected, so Algorithm 3 marks unrelated
    # objects instead. The queries are only handed to the open-vocabulary detector
    # (OWLv2), which scores them against the image, so a non-object word simply
    # returns nothing above threshold.
    covered = set(anchors) | set(categories) | set(expanded)
    covered |= {word for phrase in compounds for word in phrase.split()}
    def _is_participle(token: str) -> bool:
        # Participles and predicate adjectives must never become marks: the
        # question "unpeeled or peeled" rendered `peeled_1`/`unpeeled_2` onto
        # arbitrary fruit - the answer options drawn on the image - and
        # "throwing" labelled a frisbee `throwing_1`.
        if token in _PARTICIPLE_OBJECTS:
            return False
        return token.endswith("ing") or token.endswith("ed")

    open_candidates = [
        token
        for token in tokens
        if len(token) > 2
        and token not in _STOPWORDS
        and token not in _ATTRIBUTE_WORDS
        and token not in relation_words
        and token not in _META_WORDS
        and not _is_participle(token)
        and token not in covered
        and canonical_object_label(token) not in covered
        and canonical_object_label(token) not in _VISUAL_OBJECTS
    ]
    # A modifier followed by a head noun becomes a phrase query ("teddy bear",
    # "toy car"): the head may be a known visual object, which is exactly the
    # case where the bare modifier alone would be a useless query.
    open_bigrams = []
    modifiers = set()
    for left, right in zip(tokens, tokens[1:]):
        if left in open_candidates and (
            right in open_candidates or canonical_object_label(right) in covered
        ):
            open_bigrams.append(f"{left} {canonical_object_label(right)}")
            # "creamy"/"teddy" alone are useless detector queries; the phrase carries
            # the meaning, so the modifier is not emitted on its own.
            modifiers.add(left)
    open_terms = _ordered_unique(
        [*open_bigrams, *(t for t in open_candidates if t not in modifiers)]
    )[:8]

    # Bare category words are dropped from the detector queries: OWLv2 labels a
    # detection with the query string, so querying "animal" yields marks labelled
    # "animal_1" instead of "cow_1". The category's members are queried instead.
    queries = _ordered_unique(
        [*sorted(anchors), *expanded, *compounds, *open_terms]
    )
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
        open_terms=open_terms,
    )


__all__ = ["QuestionIntent", "canonical_object_label", "parse_question_intent"]
