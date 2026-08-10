#!/usr/bin/env python3
"""Audit whether graph, triples, and rendered arrows exploit the same edge set."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gom.question_intent import canonical_object_label, parse_question_intent
from gom.relations.paper import PAPER_RELATIONS, relation_digest


RELATION_MAP = {
    "next_to": "near",
    "on_top_of": "above",
}


def edge_represents_requested_relation(edge: dict, requested: set[str]) -> bool:
    relation = edge["relation"]
    modifier = edge.get("modifier", "")
    if relation in requested or modifier in requested:
        return True
    return "near" in requested and modifier in {"touching", "very_close", "close"}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def graph_edges(data: dict) -> list[dict]:
    edges = []
    for edge in data.get("links", []):
        relation = edge.get("relation")
        if relation not in PAPER_RELATIONS:
            continue
        edges.append(
            {
                "src_idx": int(edge["source"]),
                "tgt_idx": int(edge["target"]),
                "relation": relation,
                "modifier": edge.get("modifier", ""),
                "display_relation": edge.get("display_relation", relation),
            }
        )
    return edges


def base_label(value: str) -> str:
    import re

    return canonical_object_label(re.sub(r"_\d+$", "", str(value)))


def audit(
    eval_path: Path, preprocessing: Path, granularity: str = "image"
) -> dict:
    rows = read_jsonl(eval_path)
    image_counts: dict[str, int] = defaultdict(int)
    counts: Counter = Counter(total_questions=len(rows))
    failures: list[dict] = []

    for row in rows:
        stem = Path(row["image_path"]).stem
        if granularity == "image":
            index = 1
        else:
            image_counts[stem] += 1
            index = image_counts[stem]
        prefix = f"{stem}_q{index}"
        graph_path = preprocessing / f"{prefix}_graph.json"
        triples_path = preprocessing / f"{prefix}_graph_triples.txt"
        render_meta_path = preprocessing / f"{prefix}_render_variants.json"
        if not graph_path.is_file() or not triples_path.is_file() or not render_meta_path.is_file():
            counts["missing_artifacts"] += 1
            failures.append({"question_id": row["question_id"], "error": "missing_artifacts"})
            continue

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        edges = graph_edges(graph)
        digest = relation_digest(edges)
        graph_meta = graph.get("graph", {})
        if graph_meta.get("edge_digest") != digest or graph_meta.get("edge_count") != len(edges):
            counts["graph_digest_mismatch"] += 1
        expected_triples = ["Triples:"]
        labels = {int(node["id"]): str(node.get("label", "")) for node in graph.get("nodes", [])}
        for edge in edges:
            expected_triples.append(
                f"{labels[edge['src_idx']]} -({edge['display_relation']})-> "
                f"{labels[edge['tgt_idx']]}"
            )
        actual_triples = [
            line.strip()
            for line in triples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # The graph serializer and the human-readable triples writer may use
        # different stable traversal orders.  Validate the represented edge
        # multiset rather than treating harmless ordering differences as data
        # corruption.
        if Counter(actual_triples) != Counter(expected_triples):
            counts["triples_mismatch"] += 1

        render_meta = json.loads(render_meta_path.read_text(encoding="utf-8"))
        for variant, metadata in render_meta.items():
            expected_rendered = len(edges) if metadata.get("display_relationships") else 0
            if (
                metadata.get("edge_digest") != digest
                or metadata.get("edge_count") != len(edges)
                or metadata.get("rendered_edge_count") != expected_rendered
            ):
                counts["render_metadata_mismatch"] += 1
                failures.append(
                    {
                        "question_id": row["question_id"],
                        "variant": variant,
                        "error": "render_metadata_mismatch",
                    }
                )

        if edges:
            counts["questions_with_graph_edges"] += 1
        intent = parse_question_intent(row["question"])
        requested = {RELATION_MAP.get(value, value) for value in intent.relation_terms}
        requested &= PAPER_RELATIONS | {"touching"}
        if not requested:
            continue
        counts["explicit_supported_relation_questions"] += 1
        matching_edges = [
            edge
            for edge in edges
            if edge_represents_requested_relation(edge, requested)
        ]
        if matching_edges:
            counts["requested_relation_present_anywhere"] += 1

        mentioned = {
            *intent.relation_source_terms,
            *intent.relation_anchor_terms,
            *intent.anchor_terms,
        }
        matching_nodes = {
            node_id for node_id, label in labels.items() if base_label(label) in mentioned
        }
        if mentioned:
            counts["relation_questions_with_endpoint_terms"] += 1
        if matching_nodes:
            counts["relation_questions_with_detected_endpoint"] += 1
        if matching_nodes and any(
            edge["src_idx"] in matching_nodes or edge["tgt_idx"] in matching_nodes
            for edge in matching_edges
        ):
            counts["requested_relation_incident_to_detected_endpoint"] += 1

    hard_errors = sum(
        counts[key]
        for key in (
            "missing_artifacts",
            "graph_digest_mismatch",
            "triples_mismatch",
            "render_metadata_mismatch",
        )
    )
    report = {
        "schema_version": 1,
        "counts": dict(counts),
        "hard_consistency_errors": hard_errors,
        "definitions": {
            "explicit_supported_relation_questions": "Questions whose parsed relation is in the seven-relation paper ontology or touching modifier.",
            "requested_relation_present_anywhere": "Eligible questions with at least one graph edge representing the requested relation, including touching/very_close/close modifiers as proximity evidence for next_to/near.",
            "requested_relation_incident_to_detected_endpoint": "Eligible questions where that edge touches a detected object explicitly mentioned in the query.",
        },
        "failures": failures[:100],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-granularity", choices=("image", "question"), default="image"
    )
    args = parser.parse_args()
    report = audit(args.eval, args.preprocessing, args.artifact_granularity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    return 1 if report["hard_consistency_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
