import pytest

from gom.vqa.prompts import build_vqa_prompt


def test_raw_prompt_is_invariant_across_profiles():
    prompts = {
        build_vqa_prompt("raw", "What color?", profile=profile)
        for profile in (
            "neutral_concise",
            "supplementary_concise",
            "visual_aid_concise",
            "relation_explicit_concise",
        )
    }
    assert len(prompts) == 1
    system, user = prompts.pop()
    assert system == "You are a helpful visual assistant."
    assert user.endswith("using a single word or phrase.")


def test_relation_prompt_defines_direction_without_requesting_ids():
    system, user = build_vqa_prompt(
        "visual",
        "Where is the pig?",
        profile="relation_explicit_concise",
    )
    assert "from object A to object B means A R B" in system
    assert "Do not mention object IDs" in system
    assert user.endswith("Answer using a single word or phrase.")


def test_visual_textual_requires_graph_text():
    with pytest.raises(ValueError, match="scene_graph"):
        build_vqa_prompt(
            "visual_textual",
            "Where?",
            profile="supplementary_concise",
        )


def test_paper_declared_prompt_is_verbatim_and_not_short_answer_modified():
    system, user = build_vqa_prompt(
        "visual", "Is the chair left of the table?", profile="paper_declared"
    )
    assert system.startswith("You are a multimodal assistant with spatial reasoning")
    assert user == (
        "Answer the question based on the spatial configuration in the image.\n"
        "Question: Is the chair left of the table?"
    )
    assert "single word" not in user


def test_released_artifact_prompt_is_separate_profile():
    _, user = build_vqa_prompt(
        "visual", "What color?", profile="released_artifact_bare"
    )
    assert user == "Answer with only one word.\nQuestion: What color?\nAnswer:"
