from gom.pipeline.preprocessor import rank_paper_relations


def test_paper_ranked_relations_prioritize_question_then_fall_back():
    relations = [
        {"src_idx": 0, "tgt_idx": 1, "relation": "near", "confidence": 0.9},
        {"src_idx": 2, "tgt_idx": 1, "relation": "left_of", "confidence": 0.4},
        {"src_idx": 0, "tgt_idx": 1, "relation": "left_of", "confidence": 0.3},
    ]

    ranked = rank_paper_relations(
        relations,
        question_rel_terms={"left_of"},
        question_subject_idxs={1},
        question_candidate_idxs={0},
    )

    assert ranked[0]["relation"] == "left_of"
    assert ranked[0]["src_idx"] == 0
    assert {relation["relation"] for relation in ranked} == {"left_of", "near"}


def test_paper_ranked_relations_caps_existing_pool_without_synthesis():
    relations = [
        {"src_idx": index, "tgt_idx": index + 1, "relation": "near"}
        for index in range(20)
    ]

    ranked = rank_paper_relations(relations, max_total=16)

    assert len(ranked) == 16
    assert all(relation in relations for relation in ranked)
