"""Algorithm 3 must actually filter — and the harness must feed it a question.

A whole Table 2 run once completed with query-based filtering silently inactive:
the preprocessing input carried no question, `filter_paper_graph` matched nothing,
and its `if not matched` branch kept every detected object. Every render marked the
whole scene instead of the query-relevant subgraph, and nothing failed.
"""
from gom.relations.paper import filter_paper_graph

LABELS = ["dog", "giraffe", "tree", "car", "person", "sky", "grass", "fence"]
RELATIONS = [
    {"src_idx": i, "tgt_idx": (i + 1) % len(LABELS), "relation": "left_of", "score": 1.0}
    for i in range(len(LABELS))
]


def test_query_prunes_to_mentioned_objects():
    kept, _ = filter_paper_graph(
        LABELS, RELATIONS, question="Is the dog to the left of the giraffe?"
    )
    assert {LABELS[i] for i in kept} == {"dog", "giraffe"}


def test_empty_question_keeps_everything():
    """Documents the degenerate branch, so the behaviour is deliberate not accidental."""
    kept, _ = filter_paper_graph(LABELS, RELATIONS, question="")
    assert len(kept) == len(LABELS)


def test_filtering_is_strictly_narrower_than_no_question():
    """The regression that mattered: a real question must mark fewer objects."""
    with_query, _ = filter_paper_graph(
        LABELS, RELATIONS, question="What colour is the car?"
    )
    without_query, _ = filter_paper_graph(LABELS, RELATIONS, question="")
    assert len(with_query) < len(without_query)


def test_zero_match_top_k_bounds_the_degenerate_branch():
    """The mark-storm fix: unmatchable query -> a few salient marks, not all."""
    boxes = [[0, 0, 10 * (i + 1), 10 * (i + 1)] for i in range(len(LABELS))]
    scores = [1.0] * len(LABELS)
    kept, _ = filter_paper_graph(
        LABELS,
        RELATIONS,
        question="Is there a zebra?",
        boxes=boxes,
        scores=scores,
        zero_match_top_k=2,
    )
    assert kept == [6, 7], "equal scores: the two largest boxes win"
    default_kept, _ = filter_paper_graph(LABELS, RELATIONS, question="Is there a zebra?")
    assert len(default_kept) == len(LABELS), "published keep-all stays the default"


def test_plural_question_matches_singular_label():
    """'shelves' in the question must match a detected 'shelf'."""
    kept, _ = filter_paper_graph(
        ["shelf", "bottle"], [], question="On which side are the shelves?"
    )
    assert kept == [0]


def test_single_match_keeps_its_neighbours():
    kept, edges = filter_paper_graph(LABELS, RELATIONS, question="Where is the car?")
    assert LABELS.index("car") in kept
    assert len(kept) > 1, "a lone match should retain the objects it relates to"
    assert edges


def test_semantic_similarity_uses_best_query_token(tmp_path):
    """Averaging the whole question hid every real match below the 0.5 threshold."""
    import numpy as np
    from gom.relations.paper import FastTextSimilarity

    class FakeVectors(dict):
        def __contains__(self, key):
            return dict.__contains__(self, key)

    vectors = FakeVectors({
        "horses": np.array([1.0, 0.0]),
        "horse": np.array([1.0, 0.0]),
        "are": np.array([0.0, 1.0]),
        "there": np.array([0.0, 1.0]),
        "any": np.array([0.0, 1.0]),
    })
    stub = tmp_path / "vectors.kv"
    stub.write_bytes(b"")  # only needs to exist; _vectors is injected below
    sim = FastTextSimilarity(str(stub))
    sim._vectors = vectors
    score = sim("are there any horses", "horse")
    assert score > 0.9, f"best matching token should dominate, got {score}"


def test_person_mark_matches_a_question_that_says_man():
    """The silent failure that broke relation selection for every human mark.

    Detector labels are canonicalized (man/woman/child -> person), so a question
    saying "man" never lexically matched a `person` mark, Algorithm 3 fell through
    to its zero-match branch, and the arrow it drew joined whichever two objects
    happened to be nearest instead of the pair the question named.
    """
    labels = ["person", "tie", "speaker", "sky"]
    relations = [
        {"src_idx": 0, "tgt_idx": 2, "relation": "right_of"},
        {"src_idx": 1, "tgt_idx": 3, "relation": "below"},
    ]
    kept, edges = filter_paper_graph(
        labels, relations, question="Is the speaker to the left of the man?"
    )
    assert {labels[i] for i in kept} == {"person", "speaker"}
    assert [e["relation"] for e in edges] == ["right_of"]


def test_arrow_prefers_the_queried_pair_over_a_nearer_stranger():
    """Ranking on relation type alone pointed a query object's arrow at whatever
    neighbour was closest; 38% of left/right questions got an unrelated pair."""
    labels = ["dog", "giraffe", "tree"]
    relations = [
        # the tree is much nearer, but it is not what the question is about
        {"src_idx": 0, "tgt_idx": 2, "relation": "left_of", "distance": 5.0},
        {"src_idx": 0, "tgt_idx": 1, "relation": "left_of", "distance": 400.0},
    ]
    _, edges = filter_paper_graph(
        labels, relations, question="Is the dog to the left of the giraffe?"
    )
    assert [(labels[e["src_idx"]], labels[e["tgt_idx"]]) for e in edges] == [
        ("dog", "giraffe")
    ]


def test_max_total_relations_bounds_the_whole_render():
    """top_k_per_head bounds arrows per source, not per image."""
    from gom.relations.paper import PaperRelationConfig

    labels = ["dog", "cat", "bird", "horse", "sheep", "cow"]
    relations = [
        {"src_idx": i, "tgt_idx": (i + 1) % len(labels), "relation": "left_of",
         "distance": float(i)}
        for i in range(len(labels))
    ]
    question = "Are the dog, cat, bird, horse, sheep and cow to the left of each other?"
    _, uncapped = filter_paper_graph(labels, relations, question=question)
    _, capped = filter_paper_graph(
        labels, relations, question=question,
        config=PaperRelationConfig(top_k_per_head=1, max_total_relations=2),
    )
    assert len(uncapped) > 2
    assert len(capped) == 2


def test_left_right_question_makes_both_directions_relevant():
    """"to the left or to the right of X" is one question about one axis, but the
    phrase matcher returns a single term, so half of them ranked the asked-about
    edge as irrelevant."""
    from gom.relations.paper import _query_relation_terms

    assert set(_query_relation_terms("Is A to the left or to the right of B?")) == {
        "left_of", "right_of"
    }
    assert set(_query_relation_terms("Is A to the right of B?")) == {
        "left_of", "right_of"
    }
