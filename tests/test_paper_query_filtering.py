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
