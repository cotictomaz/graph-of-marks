from gom.relations.inference import RelationInferencer


def test_in_front_enforcement_uses_requested_source_and_target_endpoints():
    inferencer = RelationInferencer(enable_parallel=False)
    # person_0 overlaps/occludes the doorway; person_1 is elsewhere.
    boxes = [
        [50, 40, 72, 112],
        [0, 35, 20, 105],
        [40, 20, 82, 100],
    ]

    relations = inferencer.enforce_question_relations(
        [],
        boxes,
        question_rel_terms={"in_front_of"},
        question_subject_idxs={2},
        question_candidate_idxs={0, 1},
    )

    assert len(relations) == 1
    assert relations[0]["src_idx"] == 0
    assert relations[0]["tgt_idx"] == 2
    assert relations[0]["relation"] == "in_front_of"


def test_existing_relation_from_wrong_source_does_not_block_enforcement():
    inferencer = RelationInferencer(enable_parallel=False)
    boxes = [
        [50, 40, 72, 112],
        [0, 35, 20, 105],
        [40, 20, 82, 100],
    ]
    wrong = {
        "src_idx": 1,
        "tgt_idx": 2,
        "relation": "in_front_of",
        "distance": 1.0,
    }

    relations = inferencer.enforce_question_relations(
        [wrong],
        boxes,
        question_rel_terms={"in_front_of"},
        question_subject_idxs={2},
        question_candidate_idxs={0},
    )

    assert any(
        relation["src_idx"] == 0
        and relation["tgt_idx"] == 2
        and relation["relation"] == "in_front_of"
        for relation in relations
    )


def test_unknown_relation_target_selects_nearest_supported_object():
    inferencer = RelationInferencer(enable_parallel=False)
    boxes = [
        [40, 0, 80, 80],
        [30, 75, 90, 90],
        [0, 100, 120, 140],
    ]

    relations = inferencer.enforce_question_relations(
        [],
        boxes,
        question_rel_terms={"on_top_of"},
        question_subject_idxs=None,
        question_candidate_idxs={0},
    )

    assert len(relations) == 1
    assert relations[0]["src_idx"] == 0
    assert relations[0]["tgt_idx"] == 1
    assert relations[0]["relation"] == "on_top_of"
