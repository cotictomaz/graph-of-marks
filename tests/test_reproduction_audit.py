import json

from gom.relations.paper import relation_digest
from reproduction.audit_relations import audit


def test_audit_counts_touching_direction_as_next_to_representation(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "image_path": "/dataset/example.jpg",
                "question": "What animal is next to the dog?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    preprocessing = tmp_path / "preprocessing"
    preprocessing.mkdir()
    edge = {
        "src_idx": 0,
        "tgt_idx": 1,
        "relation": "left_of",
        "modifier": "touching",
        "display_relation": "touching_left_of",
    }
    digest = relation_digest([edge])
    graph = {
        "graph": {"edge_digest": digest, "edge_count": 1},
        "nodes": [
            {"id": 0, "label": "dog_1"},
            {"id": 1, "label": "giraffe_1"},
        ],
        "links": [
            {
                "source": 0,
                "target": 1,
                "relation": "left_of",
                "modifier": "touching",
                "display_relation": "touching_left_of",
            }
        ],
    }
    (preprocessing / "example_q1_graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )
    (preprocessing / "example_q1_graph_triples.txt").write_text(
        "Triples:\ndog_1 -(touching_left_of)-> giraffe_1\n", encoding="utf-8"
    )
    (preprocessing / "example_q1_render_variants.json").write_text(
        json.dumps(
            {
                "gom_text_labeled": {
                    "display_relationships": True,
                    "edge_digest": digest,
                    "edge_count": 1,
                    "rendered_edge_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    report = audit(eval_path, preprocessing)

    assert report["hard_consistency_errors"] == 0
    assert report["counts"]["explicit_supported_relation_questions"] == 1
    assert report["counts"]["requested_relation_present_anywhere"] == 1
    assert report["counts"]["requested_relation_incident_to_detected_endpoint"] == 1
