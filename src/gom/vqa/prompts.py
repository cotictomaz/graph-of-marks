"""Versioned VQA prompt profiles used by paper reproduction experiments."""
from __future__ import annotations

from typing import Optional, Tuple


PROMPT_PROFILES = (
    "paper_declared",
    "released_artifact_bare",
    "neutral_concise",
    "supplementary_concise",
    "visual_aid_concise",
    "relation_explicit_concise",
    "direct_concise",
    "gom_v2_concise",
    "gom_v3_concise",
    "gom_v3b_concise",
    "gom_v4_concise",
)

SYSTEM_RAW = "You are a helpful visual assistant."
USER_RAW = "Question: {question}\nAnswer the question using a single word or phrase."
SYSTEM_RELEASED_ARTIFACT = "You are a helpful visual assistant."
USER_RELEASED_ARTIFACT = "Answer with only one word.\nQuestion: {question}\nAnswer:"

SYSTEM_SUPPLEMENTARY_VISUAL = (
    "You are a multimodal assistant with spatial reasoning capabilities. "
    "Use the visual scene graph in the image to interpret spatial relations "
    "and answer questions grounded in the visual layout."
)
SYSTEM_SUPPLEMENTARY_VISUAL_TEXTUAL = (
    "You are a multimodal assistant capable of understanding both visual and "
    "textual scene graphs. Use the image and the accompanying graph description "
    "to answer the question accurately."
)
USER_SUPPLEMENTARY_VISUAL = (
    "Answer the question based on the spatial configuration in the image.\n"
    "Question: {question}\n"
    "Answer using a single word or phrase."
)
USER_SUPPLEMENTARY_VISUAL_TEXTUAL = (
    "Answer the question based on the spatial configuration in the image and "
    "the graph description.\n"
    "Scene Graph (Textual):\n{scene_graph}\n"
    "Question: {question}\n"
    "Answer using a single word or phrase."
)
USER_PAPER_VISUAL = (
    "Answer the question based on the spatial configuration in the image.\n"
    "Question: {question}"
)
USER_PAPER_VISUAL_TEXTUAL = (
    "Answer the question based on the spatial configuration in the image and "
    "the graph description.\n"
    "Scene Graph (Textual):\n{scene_graph}\n"
    "Question: {question}"
)

VISUAL_AID_SUFFIX = (
    " The colored outlines, object labels, and arrows are optional visual aids. "
    "If a mark conflicts with visible image content, rely on the image."
)
RELATION_EXPLICIT_SUFFIX = (
    " An arrow labeled R from object A to object B means A R B. "
    "Use directed arrows only when relevant. Do not mention object IDs or relation "
    "descriptions unless the question asks for them."
)
# direct_concise: supplementary_concise wording with an anti-plan answer
# instruction on every condition. Targets models (LlamaV-o1) that otherwise
# emit a reasoning plan instead of an answer when marks are present.
DIRECT_ANSWER_INSTRUCTION = (
    "Answer with the final answer only, using a single word or phrase. "
    "Do not describe your approach or explain your reasoning."
)
_CONCISE_INSTRUCTIONS = (
    "Answer the question using a single word or phrase.",
    "Answer using a single word or phrase.",
)

# gom_v2_concise: the flip-case audit showed marked-condition answers leaking the
# annotation vocabulary ("person_1", "Right of", "green"). This profile explains
# the marks once, demotes them to optional aids, and bans them as answers. Raw is
# byte-identical to direct_concise so raw predictions stay comparable across runs.
SYSTEM_GOM_V2 = (
    "You are a helpful visual assistant. The photograph has been annotated: "
    "colored outlines mark detected objects, tags such as person_1 or bare "
    "numbers are object reference IDs, and labeled arrows show spatial "
    "relations between the outlined objects. The annotations are hints, not "
    "part of the scene; if they conflict with what the photograph shows, trust "
    "the photograph. Answer about the photograph itself, in plain natural "
    "language. Never use a reference ID as an answer: if asked who someone is, "
    "say what you see, such as woman, man, or boy - never person_1 or a "
    "number. Never use an arrow's relation word as an answer: if asked where "
    "something is, name the place in the photograph, such as on the ground or "
    "in the car - never above or below alone. Never answer with the color of "
    "an outline."
)
USER_GOM_V2 = "Question: {question}\n" + DIRECT_ANSWER_INSTRUCTION

# gom_v3_concise: the flip audit (reproduction/FLIP_AUDIT_GOM_V2.md) showed the
# v2 wording ban was not enough on its own - Qwen and LlamaV still answered
# "person_1" to "who ...?" questions (83/2975), and marked scenes still drew
# yes->no existence denials. This profile keeps the v2 explanation, adds an
# explicit presence assertion, and shows the behaviour instead of only naming it.
SYSTEM_GOM_V3 = (
    "You are a helpful visual assistant. The photograph has been annotated: "
    "colored outlines mark detected objects, tags such as person_1 or bare "
    "numbers are object reference IDs, and labeled arrows show spatial "
    "relations between the outlined objects. The annotations are hints, not "
    "part of the scene; if they conflict with what the photograph shows, trust "
    "the photograph. Every outlined object is really present in the "
    "photograph, and objects with no outline are present too - the marks cover "
    "only some of the scene.\n"
    "Answer about the photograph itself, in plain natural language. Never "
    "answer with a reference ID, an arrow's relation word, or the color of an "
    "outline. For example:\n"
    "Q: Who is wearing a jacket? A: woman   (never person_1)\n"
    "Q: Are there any benches near the sidewalk? A: yes\n"
    "Q: Where is the dog? A: in the car   (never above or below)"
)
# Measured on the audit set: repeating the ban in the user turn as well cut
# Qwen's text-tag leaks only 13->12 while costing Gemma 8-10 accuracy points, so
# the user turn stays the plain direct instruction. Text-tag ID leakage on "who"
# questions is a model behaviour we cannot prompt away; the numeric-ID
# conditions (1-3 leaks vs 12-14) are the mitigation and both run in the eval.
USER_GOM_V3 = USER_GOM_V2

# gom_v3b_concise: same intent as v3, positive framing only. Measured on
# data_v6: v3's explicit bans ("never person_1", "never above or below") RAISED
# the very answers they forbid - Gemma's bare-relation-word answers went 1 -> 31
# and LlamaV's plan-mode came back - because naming a token in a prohibition
# also makes it available. This version never prints a forbidden token.
SYSTEM_GOM_V3B = (
    "You are a helpful visual assistant. The photograph has been annotated with "
    "colored outlines around detected objects, short tags naming them, and "
    "arrows showing how those objects are arranged. The annotations are hints "
    "added on top; the photograph itself is the evidence, and it also contains "
    "things that were not annotated. Answer the question about the photograph, "
    "using the ordinary everyday word for what you see."
)


# gom_v4_concise: gom_v2_concise (the measured best) plus two positive-framed
# sentences for the two mechanisms the gom_v3 flip sweep quantified and no render
# fix can reach.
#
#   1. The mark set is read as an existence oracle, in both directions. On GQA
#      existence questions the flip rate when the questioned noun IS marked and the
#      gold is "no" is 34.5% (gemma) / 34.1% (llamav) against 6.0% / 24.2% when it
#      is not; symmetrically, gold "yes" with the noun UNmarked flips 18.4% (gemma)
#      / 17.6% (qwen) against 7.6% / 13.5% when marked. Detector confidence does
#      not separate the false-positive marks from the true ones (0.462 vs 0.474),
#      so this cannot be fixed by a threshold - only by saying the marks are
#      partial and fallible.
#   2. Arrows connect whatever Algorithm 3 selected, which is often not the pair
#      the question names.
#
# Positive framing is not a style choice: v3's explicit bans RAISED the answers
# they forbade (RESULTS.md gom_v3), so nothing here names a forbidden token.
SYSTEM_GOM_V4 = SYSTEM_GOM_V2 + (
    " The outlines cover only part of the scene, and an outline can be wrong: "
    "decide what is present or absent by looking at the photograph itself. When "
    "the question names two things, find those two in the photograph and answer "
    "about them, whichever objects the arrows happen to connect."
)
USER_GOM_V4 = USER_GOM_V2


def _directify(user: str) -> str:
    for sentence in _CONCISE_INSTRUCTIONS:
        user = user.replace(sentence, DIRECT_ANSWER_INSTRUCTION)
    return user


def build_vqa_prompt(
    mode: str,
    question: str,
    *,
    scene_graph: Optional[str] = None,
    profile: str = "gom_v2_concise",
) -> Tuple[str, str]:
    """Return system and user text for a controlled VQA condition.

    Default is ``gom_v2_concise``: the empirically best mark-aware prompt. It
    tells the model the drawn object-ID tags (``person_1``, bare numbers) and
    relation-arrow words are pointers, not answers -- so the VLM does not copy
    the label tags. Pass ``profile=`` to override (e.g. ``paper_declared`` for
    verbatim paper reproduction).
    """
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt profile: {profile}")
    if mode not in {"raw", "visual", "visual_textual"}:
        raise ValueError(f"Unknown VQA mode: {mode}")

    if profile == "released_artifact_bare":
        return SYSTEM_RELEASED_ARTIFACT, USER_RELEASED_ARTIFACT.format(
            question=question
        )

    if mode == "raw" or profile == "neutral_concise":
        user = USER_RAW.format(question=question)
        if profile in {
            "direct_concise",
            "gom_v2_concise",
            "gom_v3_concise",
            "gom_v3b_concise",
            "gom_v4_concise",
        }:
            user = _directify(user)
        return SYSTEM_RAW, user

    if profile in {
        "gom_v2_concise", "gom_v3_concise", "gom_v3b_concise", "gom_v4_concise"
    }:
        if mode == "visual_textual":
            # text_graph sends a clean image + triples: the mark explanation
            # would be false there, so fall back to the direct textual prompt.
            if scene_graph is None:
                raise ValueError("scene_graph is required for visual_textual mode")
            user = _directify(
                USER_SUPPLEMENTARY_VISUAL_TEXTUAL.format(
                    scene_graph=scene_graph, question=question
                )
            )
            return SYSTEM_SUPPLEMENTARY_VISUAL_TEXTUAL, user
        if profile == "gom_v3_concise":
            return SYSTEM_GOM_V3, USER_GOM_V3.format(question=question)
        if profile == "gom_v3b_concise":
            return SYSTEM_GOM_V3B, USER_GOM_V2.format(question=question)
        if profile == "gom_v4_concise":
            return SYSTEM_GOM_V4, USER_GOM_V4.format(question=question)
        return SYSTEM_GOM_V2, USER_GOM_V2.format(question=question)

    if mode == "visual_textual":
        if scene_graph is None:
            raise ValueError("scene_graph is required for visual_textual mode")
        system = SYSTEM_SUPPLEMENTARY_VISUAL_TEXTUAL
        template = (
            USER_PAPER_VISUAL_TEXTUAL
            if profile == "paper_declared"
            else USER_SUPPLEMENTARY_VISUAL_TEXTUAL
        )
        user = template.format(scene_graph=scene_graph, question=question)
    else:
        system = SYSTEM_SUPPLEMENTARY_VISUAL
        template = (
            USER_PAPER_VISUAL
            if profile == "paper_declared"
            else USER_SUPPLEMENTARY_VISUAL
        )
        user = template.format(question=question)

    if profile == "visual_aid_concise":
        system += VISUAL_AID_SUFFIX
    elif profile == "relation_explicit_concise":
        system += RELATION_EXPLICIT_SUFFIX
    elif profile == "direct_concise":
        user = _directify(user)
    return system, user
