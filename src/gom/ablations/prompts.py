"""
Prompt template builders for the prompting techniques experiment.

Each public function returns a string with a single {question} placeholder,
compatible with run_vqa's `prompt_tpl.format(question=ex.question)` call.

The scene graph text is prepended to the template by run_vqa before the
format call, so every template can reference "the scene graph above" without
needing to include it explicitly.

Answer parsing in runner.py uses:
    ans.rsplit("Answer:", 1)[-1].strip().strip('"')
which extracts everything after the LAST "Answer:" token.  This is already
CoT-compatible, but requires templates to end with "Answer: [concise answer]"
to avoid verbose post-answer text degrading exact-match scores.
"""

from typing import Any, Dict

# -----------------------------------------------------------------------
# Few-shot example pool (textual only — no images).
# Format mirrors the GoM scene graph style for consistency.
# The pool is ordered; build_few_shot_template takes the first n_shots entries.
# -----------------------------------------------------------------------
FEW_SHOT_POOL = [
    {
        "context": (
            "Object 1 (dog) is RIGHT OF Object 2 (child). "
            "Object 3 (ball) is IN FRONT OF Object 2 (child)."
        ),
        "question": "Who is on the right of the child?",
        "answer": "dog",
    },
    {
        "context": (
            "Object 1 (cup) is ON TOP OF Object 2 (table). "
            "Object 3 (book) is NEXT TO Object 2 (table)."
        ),
        "question": "What is on the table?",
        "answer": "cup",
    },
    {
        "context": (
            "Object 1 (bicycle) is IN FRONT OF Object 2 (building). "
            "Object 3 (person) is ON TOP OF Object 1 (bicycle)."
        ),
        "question": "What is the person riding?",
        "answer": "bicycle",
    },
    {
        "context": (
            "Object 1 (cat) is LEFT OF Object 2 (dog). "
            "Object 3 (sofa) is BEHIND Object 1 (cat)."
        ),
        "question": "What animal is on the left?",
        "answer": "cat",
    },
    {
        "context": (
            "Object 1 (sandwich) is ON TOP OF Object 2 (plate). "
            "Object 3 (glass) is NEXT TO Object 2 (plate)."
        ),
        "question": "What is on the plate?",
        "answer": "sandwich",
    },
    {
        "context": (
            "Object 1 (airplane) is ABOVE Object 2 (airport). "
            "Object 3 (car) is IN FRONT OF Object 2 (airport)."
        ),
        "question": "What is above the airport?",
        "answer": "airplane",
    },
]


# -----------------------------------------------------------------------
# Template builders (private)
# -----------------------------------------------------------------------

def _baseline_template() -> str:
    return (
        "Answer the question based on the spatial configuration in the image "
        "and the graph description.\n\n"
        "Question: {question}"
    )


def _few_shot_template(n_shots: int) -> str:
    n = min(n_shots, len(FEW_SHOT_POOL))
    if n == 0:
        return _baseline_template()

    shot_lines = []
    for i, ex in enumerate(FEW_SHOT_POOL[:n], 1):
        shot_lines.append(
            f"EXAMPLE {i}: "
            f"CONTEXT: {ex['context']} "
            f"QUESTION: {ex['question']} "
            f"ANSWER: {ex['answer']}"
        )
    shots_text = "\n".join(shot_lines)

    # {{question}} in an f-string produces the literal {question} placeholder
    # that run_vqa will later fill with the actual question text.
    return (
        f"Here are examples of how to answer visual questions using scene graph descriptions:\n\n"
        f"{shots_text}\n\n"
        f"Now, given the image and the scene graph above, answer the following question:\n"
        f"QUESTION: {{question}}\n"
        f"ANSWER:"
    )


def _chain_of_thought_template() -> str:
    # Ends with "Answer: [one word or short phrase]" so the existing
    # rsplit("Answer:", 1) parser always extracts a concise answer.
    return (
        "Look at the image and the scene graph description carefully. "
        "Reason step by step about the visible objects and their spatial relationships, "
        "then provide a concise final answer.\n\n"
        "Question: {question}\n\n"
        "Think step by step, then write your final answer on the last line "
        "using this exact format (one word or short phrase only):\n"
        "Answer: [your concise answer]"
    )


def _graph_guided_template() -> str:
    # Instructs the model to explicitly reference the scene graph before answering.
    # Complementary to CoT: more structured, less open-ended reasoning.
    return (
        "Using the scene graph description above, identify all objects and spatial "
        "relationships that are relevant to the question. "
        "Then answer the question in one word or a short phrase.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

VALID_STRATEGIES = ("baseline", "few_shot", "chain_of_thought", "graph_guided")


def build_prompt_template(strategy: str, strategy_cfg: Dict[str, Any]) -> str:
    """
    Return a prompt template string with a single {question} placeholder.

    Args:
        strategy: One of "baseline", "few_shot", "chain_of_thought", "graph_guided".
        strategy_cfg: The strategy's config dict from the YAML file.
                      Only "few_shot" reads a value ("n_shots").

    Returns:
        A string containing exactly one {question} placeholder, ready to be
        passed as prompt_tpl to run_vqa.

    Raises:
        ValueError: If strategy is not recognised.
    """
    if strategy == "baseline":
        return _baseline_template()
    elif strategy == "few_shot":
        n_shots = int(strategy_cfg.get("n_shots", 3))
        return _few_shot_template(n_shots)
    elif strategy == "chain_of_thought":
        return _chain_of_thought_template()
    elif strategy == "graph_guided":
        return _graph_guided_template()
    else:
        raise ValueError(
            f"Unknown prompting strategy: {strategy!r}. "
            f"Valid values: {VALID_STRATEGIES}"
        )
