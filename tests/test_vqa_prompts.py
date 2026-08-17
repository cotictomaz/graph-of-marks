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


def test_direct_concise_forbids_reasoning_on_every_condition():
    for mode in ("raw", "visual"):
        _, user = build_vqa_prompt(mode, "How many dogs?", profile="direct_concise")
        assert "Do not describe your approach" in user
        assert "single word or phrase" in user


def test_gom_v2_concise_raw_matches_direct_concise_raw():
    assert build_vqa_prompt(
        "raw", "How many dogs?", profile="gom_v2_concise"
    ) == build_vqa_prompt("raw", "How many dogs?", profile="direct_concise")


def test_gom_v2_concise_marked_explains_marks_and_bans_leakage():
    system, user = build_vqa_prompt(
        "visual", "Where is the dog?", profile="gom_v2_concise"
    )
    assert "hints, not part of the scene" in system
    assert "trust the photograph" in system
    assert "Never use a reference ID" in system
    assert "never above or below alone" in system
    assert "color of an outline" in system
    assert user.startswith("Question: Where is the dog?")
    assert user.endswith("Do not describe your approach or explain your reasoning.")


def test_gom_v3_concise_adds_presence_and_examples():
    raw_v3 = build_vqa_prompt("raw", "How many dogs?", profile="gom_v3_concise")
    assert raw_v3 == build_vqa_prompt("raw", "How many dogs?", profile="direct_concise")
    system, user = build_vqa_prompt("visual", "Who is that?", profile="gom_v3_concise")
    # presence assertion (the yes->no existence-denial fix)
    assert "really present in the photograph" in system
    # few-shot exemplars (the person_1 ID-leak fix)
    assert "A: woman" in system and "never person_1" in system
    assert "never above or below" in system
    # user turn stays the plain direct instruction (measured: repeating the ban
    # there cost accuracy without fixing the leak)
    assert user.startswith("Question: Who is that?")
    assert user.endswith("Do not describe your approach or explain your reasoning.")


def test_gom_v3b_concise_never_prints_a_forbidden_token():
    """Naming a banned token in a prohibition also makes it available: v3's
    explicit bans raised the very answers they forbade."""
    system, user = build_vqa_prompt("visual", "Who is that?", profile="gom_v3b_concise")
    for token in ("person_1", "above", "below", "never"):
        assert token not in system.lower(), f"{token!r} must not appear in the prompt"
    assert "hints" in system and "photograph itself is the evidence" in system
    assert build_vqa_prompt("raw", "Q?", profile="gom_v3b_concise") == build_vqa_prompt(
        "raw", "Q?", profile="direct_concise"
    )
