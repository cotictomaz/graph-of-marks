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


def build_vqa_prompt(
    mode: str,
    question: str,
    *,
    scene_graph: Optional[str] = None,
    profile: str = "supplementary_concise",
) -> Tuple[str, str]:
    """Return system and user text for a controlled VQA condition."""
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt profile: {profile}")
    if mode not in {"raw", "visual", "visual_textual"}:
        raise ValueError(f"Unknown VQA mode: {mode}")

    if profile == "released_artifact_bare":
        return SYSTEM_RELEASED_ARTIFACT, USER_RELEASED_ARTIFACT.format(
            question=question
        )

    if mode == "raw" or profile == "neutral_concise":
        return SYSTEM_RAW, USER_RAW.format(question=question)

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
    return system, user
