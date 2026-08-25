import pytest

from gom.relations.paper import (
    PaperRelationConfig,
    filter_paper_graph,
    infer_paper_relations,
    relation_digest,
    validate_paper_relation,
)


def edge(relations, source, target, relation):
    return next(
        item
        for item in relations
        if item["src_idx"] == source
        and item["tgt_idx"] == target
        and item["relation"] == relation
    )


def test_algorithm2_uses_natural_edge_direction_and_depth_sign():
    relations = infer_paper_relations(
        [[0, 0, 10, 10], [40, 0, 50, 10]],
        labels=["cat", "table"],
        depths=[0.2, 0.8],
        image_size=(100, 100),
    )
    assert edge(relations, 0, 1, "left_of")
    assert edge(relations, 0, 1, "behind")
    assert edge(relations, 1, 0, "right_of")
    assert edge(relations, 1, 0, "in_front_of")


def test_algorithm2_modifiers_only_apply_to_different_classes():
    boxes = [[0, 0, 10, 10], [12, 0, 22, 10]]
    config = PaperRelationConfig(direction_margin=1)
    same = infer_paper_relations(
        boxes, labels=["cat", "cat"], depths=None, image_size=(100, 100), config=config
    )
    different = infer_paper_relations(
        boxes, labels=["cat", "dog"], depths=None, image_size=(100, 100), config=config
    )
    assert "modifier" not in edge(same, 0, 1, "left_of")
    assert edge(different, 0, 1, "left_of")["modifier"] == "touching"
    assert edge(different, 0, 1, "left_of")["display_relation"] == "touching_left_of"


def test_algorithm3_prioritizes_requested_relation_then_distance():
    relationships = [
        {"src_idx": 0, "tgt_idx": 1, "relation": "above", "distance": 1},
        {"src_idx": 0, "tgt_idx": 2, "relation": "near", "distance": 10},
    ]
    kept, filtered = filter_paper_graph(
        ["pig", "person", "chair"],
        relationships,
        question="What is next to the pig?",
        config=PaperRelationConfig(top_k_per_head=1),
    )
    assert kept == [0, 1, 2]
    assert len(filtered) == 1
    assert filtered[0]["relation"] == "near"
    assert filtered[0]["tgt_idx"] == 2


def test_algorithm3_treats_proximity_modifier_as_next_to_evidence():
    relationships = [
        {"src_idx": 0, "tgt_idx": 1, "relation": "above", "distance": 1},
        {
            "src_idx": 0,
            "tgt_idx": 2,
            "relation": "left_of",
            "modifier": "touching",
            "display_relation": "touching_left_of",
            "distance": 10,
        },
    ]
    _, filtered = filter_paper_graph(
        ["pig", "person", "chair"],
        relationships,
        question="What is next to the pig?",
        config=PaperRelationConfig(top_k_per_head=1),
    )
    assert len(filtered) == 1
    assert filtered[0]["display_relation"] == "touching_left_of"


def test_algorithm3_remaps_endpoints_after_multi_object_filter():
    relationships = [
        {"src_idx": 1, "tgt_idx": 3, "relation": "left_of", "distance": 2},
        {"src_idx": 3, "tgt_idx": 1, "relation": "right_of", "distance": 2},
    ]
    kept, filtered = filter_paper_graph(
        ["sky", "pig", "tree", "chair"],
        relationships,
        question="Is the pig left of the chair?",
    )
    assert kept == [1, 3]
    assert [(item["src_idx"], item["tgt_idx"]) for item in filtered] == [(0, 1)]


def test_paper_path_rejects_conceptnet_labels():
    with pytest.raises(ValueError, match="Unsupported paper relation"):
        validate_paper_relation({"relation": "cnet_synonym"})


def test_relation_digest_includes_modifier():
    plain = [{"src_idx": 0, "tgt_idx": 1, "relation": "left_of"}]
    touching = [
        {
            "src_idx": 0,
            "tgt_idx": 1,
            "relation": "left_of",
            "modifier": "touching",
        }
    ]
    assert relation_digest(plain) != relation_digest(touching)


def test_both_axes_are_emitted_when_both_clear_the_margin():
    """A left/right question about a vertically-dominant pair used to get an
    `above` edge and no left/right edge at all."""
    from gom.relations.paper import infer_paper_relations

    # target is far below and clearly to the right: |dy| dominates, so only
    # `above` used to be emitted. Convention: positive dx => source is left_of target.
    boxes = [[0.0, 0.0, 10.0, 10.0], [100.0, 400.0, 110.0, 410.0]]
    rels = infer_paper_relations(
        boxes, labels=["dog", "cat"], depths=None, image_size=(500, 500)
    )
    out = {(r["src_idx"], r["tgt_idx"], r["relation"]) for r in rels}
    assert (0, 1, "above") in out
    assert (0, 1, "left_of") in out


def test_relation_digest_ignores_edge_order():
    """The graph records the digest from the selection list; the renderer records
    it after a NetworkX round-trip, which re-orders edges by source node. Those
    two agreed only by coincidence, and inference hard-fails on a mismatch."""
    from gom.relations.paper import relation_digest

    edges = [
        {"src_idx": 2, "tgt_idx": 0, "relation": "above"},
        {"src_idx": 0, "tgt_idx": 1, "relation": "left_of", "modifier": "touching"},
        {"src_idx": 1, "tgt_idx": 2, "relation": "near"},
    ]
    assert relation_digest(edges) == relation_digest(list(reversed(edges)))
    assert relation_digest(edges) != relation_digest(edges[:2])
