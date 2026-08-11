"""Paper-declared Graph-of-Marks relation estimation and query filtering.

This module intentionally implements only Algorithms 2 and 3 from the GoM
supplement.  The general-purpose relation inferencer contains additional
heuristics; keeping this implementation separate makes the reproduction path
auditable and prevents those heuristics from leaking into paper runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


PAPER_RELATIONS = frozenset(
    {"above", "below", "left_of", "right_of", "in_front_of", "behind", "near"}
)
PAPER_MODIFIERS = frozenset({"touching", "very_close", "close"})

_RELATION_ALIASES: Mapping[str, Set[str]] = {
    "above": {"above", "over", "on top of", "higher than"},
    "below": {"below", "under", "underneath", "lower than"},
    "left_of": {"left", "left of", "to the left of"},
    "right_of": {"right", "right of", "to the right of"},
    "in_front_of": {"in front", "in front of", "front of", "closer than"},
    "behind": {"behind", "in back of", "at the back of"},
    "near": {"near", "nearby", "next to", "beside", "close to", "adjacent to"},
    "touching": {"touching", "in contact with"},
    "very_close": {"very close", "extremely close"},
    "close": {"close", "close to"},
}


@dataclass(frozen=True)
class PaperRelationConfig:
    direction_margin: float = 20.0
    depth_threshold: float = 0.10
    near_threshold: float = 5000.0
    touching_iou_threshold: float = 0.10
    touching_gap_threshold: float = 3.0
    very_close_threshold: float = 0.05
    close_threshold: float = 0.12
    query_object_threshold: float = 0.50
    query_relation_threshold: float = 0.50
    top_k_per_head: int = 3


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _edge_gap(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(float(a[0]) - float(b[2]), float(b[0]) - float(a[2]), 0.0)
    dy = max(float(a[1]) - float(b[3]), float(b[1]) - float(a[3]), 0.0)
    return math.hypot(dx, dy)


def _normalized_distance(
    a: Tuple[float, float], b: Tuple[float, float], image_size: Tuple[int, int]
) -> float:
    width, height = image_size
    dx = (b[0] - a[0]) / max(1.0, float(width))
    dy = (b[1] - a[1]) / max(1.0, float(height))
    return math.hypot(dx, dy)


def infer_paper_relations(
    boxes: Sequence[Sequence[float]],
    *,
    labels: Optional[Sequence[str]] = None,
    depths: Optional[Sequence[float]],
    image_size: Tuple[int, int],
    config: PaperRelationConfig = PaperRelationConfig(),
) -> List[Dict[str, Any]]:
    """Implement Algorithm 2 for every ordered object pair.

    Relations use the natural edge direction: ``source relation target``.  A
    positive x displacement therefore means the source is left of the target;
    a positive y displacement means the source is above the target.  Depth is
    normalized with larger values closer to the camera.
    """
    centers = [_center(box) for box in boxes]
    valid_depths = list(depths) if depths is not None and len(depths) == len(boxes) else None
    relationships: List[Dict[str, Any]] = []

    for source, source_box in enumerate(boxes):
        for target, target_box in enumerate(boxes):
            if source == target:
                continue
            sx, sy = centers[source]
            tx, ty = centers[target]
            dx, dy = tx - sx, ty - sy
            distance = math.hypot(dx, dy)
            pair_relations: List[Tuple[str, str]] = []

            directional: Optional[str] = None
            if abs(dy) >= abs(dx) and abs(dy) > config.direction_margin:
                directional = "above" if dy > 0.0 else "below"
            elif abs(dx) > config.direction_margin:
                directional = "left_of" if dx > 0.0 else "right_of"

            modifier: Optional[str] = None
            if directional:
                different_classes = (
                    labels is not None
                    and len(labels) == len(boxes)
                    and _normalize(labels[source]) != _normalize(labels[target])
                )
                if different_classes and (
                    _iou(source_box, target_box) > config.touching_iou_threshold
                    or _edge_gap(source_box, target_box) <= config.touching_gap_threshold
                ):
                    modifier = "touching"
                elif different_classes:
                    normalized = _normalized_distance(centers[source], centers[target], image_size)
                    if normalized < config.very_close_threshold:
                        modifier = "very_close"
                    elif normalized < config.close_threshold:
                        modifier = "close"
                pair_relations.append((directional, modifier or ""))

            if valid_depths is not None:
                depth_delta = float(valid_depths[target]) - float(valid_depths[source])
                if abs(depth_delta) > config.depth_threshold:
                    pair_relations.append(
                        ("behind" if depth_delta > 0.0 else "in_front_of", "")
                    )

            if not pair_relations and distance < config.near_threshold:
                pair_relations.append(("near", ""))

            for relation, relation_modifier in pair_relations:
                item: Dict[str, Any] = {
                    "src_idx": source,
                    "tgt_idx": target,
                    "relation": relation,
                    "distance": distance,
                    "source": "paper_algorithm2",
                }
                if relation_modifier:
                    item["modifier"] = relation_modifier
                    item["display_relation"] = f"{relation_modifier}_{relation}"
                relationships.append(item)

    return relationships


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalize(text)} "
    normalized_phrase = _normalize(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def label_aliases(label: str) -> Set[str]:
    """Return lexical aliases for a detector label using WordNet when present."""
    clean = _normalize(re.sub(r"_\d+$", "", str(label)))
    aliases = {clean, clean.replace("_", " ")}
    try:
        from nltk.corpus import wordnet as wn

        for candidate in {clean, clean.replace(" ", "_")}:
            for synset in wn.synsets(candidate):
                for lemma in synset.lemma_names():
                    aliases.add(_normalize(lemma.replace("_", " ")))
    except Exception:
        pass
    return {alias for alias in aliases if alias}


class FastTextSimilarity:
    """Lazy mmap-backed FastText similarity over a prebuilt gensim KeyedVectors file."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path).expanduser() if path else None
        self._vectors = None

    @property
    def available(self) -> bool:
        return self.path is not None and self.path.is_file()

    def _load(self):
        if not self.available:
            return None
        if self._vectors is None:
            try:
                from gensim.models import KeyedVectors
            except ImportError as exc:
                raise RuntimeError(
                    "paper_aaai26 semantic filtering requires gensim and a converted "
                    "cc.en.300 KeyedVectors file"
                ) from exc
            self._vectors = KeyedVectors.load(str(self.path), mmap="r")
        return self._vectors

    def __call__(self, left: str, right: str) -> float:
        """Similarity of the best-matching query token to the label.

        The left side is a whole question and the right side a short object label,
        so averaging every left token dilutes the query with function words:
        "Are there either any helmets or horses in this image?" against "horse"
        scores 0.27 as a sentence mean but 0.84 token-wise. Under the mean, no
        pair ever reached the paper's 0.5 query-object threshold, so Algorithm 3's
        semantic half never fired and only exact lexical hits survived - which
        also miss plurals ("horses" does not contain "horse" as a whole word).
        Taking the max over query tokens still separates cleanly: unrelated pairs
        ("What color is the bus?" vs "horse") stay near 0.20.
        """
        vectors = self._load()
        if vectors is None:
            return 0.0
        left_tokens = [token for token in _normalize(left).split() if token in vectors]
        right_tokens = [token for token in _normalize(right).split() if token in vectors]
        if not left_tokens or not right_tokens:
            return 0.0
        import numpy as np

        right_vec = np.mean([vectors[token] for token in right_tokens], axis=0)
        right_norm = float(np.linalg.norm(right_vec))
        if not right_norm:
            return 0.0
        best = 0.0
        for token in left_tokens:
            left_vec = vectors[token]
            left_norm = float(np.linalg.norm(left_vec))
            if left_norm:
                best = max(
                    best,
                    float(np.dot(left_vec, right_vec) / (left_norm * right_norm)),
                )
        return best


def _relation_relevant(
    relation_terms: Sequence[str],
    relation: str,
    *,
    semantic_similarity: Optional[Callable[[str, str], float]],
    threshold: float,
) -> bool:
    components = [relation]
    modifier = ""
    for candidate_modifier in PAPER_MODIFIERS:
        prefix = f"{candidate_modifier}_"
        if relation.startswith(prefix):
            modifier = candidate_modifier
            components = [candidate_modifier, relation[len(prefix):]]
            break
    if "near" in relation_terms and modifier in {"touching", "very_close", "close"}:
        return True
    aliases = set()
    for component in components:
        aliases.update(
            _RELATION_ALIASES.get(component, {component.replace("_", " ")})
        )
    if any(
        _contains_phrase(term, alias) or _contains_phrase(alias, term)
        for term in relation_terms
        for alias in aliases
    ):
        return True
    return bool(
        semantic_similarity
        and relation_terms
        and max(
            semantic_similarity(term, alias)
            for term in relation_terms
            for alias in aliases
        ) > threshold
    )


def _query_relation_terms(question: str) -> Tuple[str, ...]:
    try:
        from gom.question_intent import parse_question_intent

        parsed = parse_question_intent(question)
        canonical = {
            "next_to": "near",
            "on_top_of": "above",
        }
        return tuple(
            sorted(canonical.get(term, term) for term in parsed.relation_terms)
        )
    except Exception:
        return tuple(
            alias
            for aliases in _RELATION_ALIASES.values()
            for alias in aliases
            if _contains_phrase(question, alias)
        )


def filter_paper_graph(
    labels: Sequence[str],
    relationships: Sequence[Mapping[str, Any]],
    *,
    question: str,
    config: PaperRelationConfig = PaperRelationConfig(),
    semantic_similarity: Optional[Callable[[str, str], float]] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Implement Algorithm 3 and remap relation endpoints to kept-object indices."""
    matched: List[int] = []
    for index, label in enumerate(labels):
        aliases = label_aliases(label)
        lexical = any(_contains_phrase(question, alias) for alias in aliases)
        semantic = bool(
            not lexical
            and semantic_similarity
            and max(semantic_similarity(question, alias) for alias in aliases)
            > config.query_object_threshold
        )
        if lexical or semantic:
            matched.append(index)

    if not matched:
        kept = list(range(len(labels)))
        candidate_edges = [dict(edge) for edge in relationships]
    elif len(matched) == 1:
        head = matched[0]
        candidate_edges = [
            dict(edge) for edge in relationships if int(edge["src_idx"]) == head
        ]
        kept = sorted({head, *(int(edge["tgt_idx"]) for edge in candidate_edges)})
    else:
        matched_set = set(matched)
        kept = sorted(matched_set)
        candidate_edges = [
            dict(edge)
            for edge in relationships
            if int(edge["src_idx"]) in matched_set and int(edge["tgt_idx"]) in matched_set
        ]

    by_head: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for edge in candidate_edges:
        by_head[int(edge["src_idx"])].append(edge)

    relation_terms = _query_relation_terms(question)
    selected: List[Dict[str, Any]] = []
    for head in kept:
        ranked = sorted(
            by_head.get(head, []),
            key=lambda edge: (
                0
                if _relation_relevant(
                    relation_terms,
                    str(edge.get("display_relation", edge["relation"])),
                    semantic_similarity=semantic_similarity,
                    threshold=config.query_relation_threshold,
                )
                else 1,
                float(edge.get("distance", math.inf)),
            ),
        )
        selected.extend(ranked[: max(0, int(config.top_k_per_head))])

    deduplicated: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    for edge in selected:
        pair = tuple(sorted((int(edge["src_idx"]), int(edge["tgt_idx"]))))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduplicated.append(edge)

    remap = {old: new for new, old in enumerate(kept)}
    remapped: List[Dict[str, Any]] = []
    for edge in deduplicated:
        source, target = int(edge["src_idx"]), int(edge["tgt_idx"])
        if source not in remap or target not in remap:
            continue
        item = dict(edge)
        item["src_idx"], item["tgt_idx"] = remap[source], remap[target]
        item["source"] = "paper_algorithm3"
        remapped.append(item)
    return kept, remapped


def relation_digest(relationships: Iterable[Mapping[str, Any]]) -> str:
    """Stable digest shared by graph, triples, render metadata, and inference."""
    canonical = [
        {
            "src_idx": int(edge["src_idx"]),
            "tgt_idx": int(edge["tgt_idx"]),
            "relation": str(edge["relation"]),
            "modifier": str(edge.get("modifier", "")),
        }
        for edge in relationships
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_paper_relation(edge: Mapping[str, Any]) -> None:
    relation = str(edge.get("relation", ""))
    modifier = str(edge.get("modifier", ""))
    if relation not in PAPER_RELATIONS:
        raise ValueError(f"Unsupported paper relation label: {relation!r}")
    if modifier and modifier not in PAPER_MODIFIERS:
        raise ValueError(f"Unsupported paper relation modifier: {modifier!r}")
