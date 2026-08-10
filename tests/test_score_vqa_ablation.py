import json

from data_paper.score_vqa_ablation import (
    cluster_bootstrap_ci,
    density_bin,
    graph_relations,
)


def test_cluster_bootstrap_resamples_whole_images():
    ci = cluster_bootstrap_ci(
        {"image_a": [1.0, 1.0], "image_b": [-1.0]},
        samples=200,
        seed=42,
    )
    assert ci[0] <= 0.0 <= ci[1]


def test_relation_density_bins_match_paper_analysis():
    assert density_bin(0) == "0"
    assert density_bin(1) == "1-3"
    assert density_bin(3) == "1-3"
    assert density_bin(4) == "4-16"
    assert density_bin(16) == "4-16"
    assert density_bin(17) == "17+"


def test_graph_relations_normalizes_displayed_predicates(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "links": [
                    {"relation": "Next To"},
                    {"relation": "in_front_of"},
                    {"relation": None},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert graph_relations({"graph_json": str(graph)}) == {
        "next_to",
        "in_front_of",
    }
