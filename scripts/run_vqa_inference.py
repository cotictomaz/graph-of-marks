#!/usr/bin/env python3
"""
VQA Inference Script using vLLM

Run VQA on GoM-processed images.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Load .env (HF_HOME, HF_TOKEN, ...) before importing vllm/torch so cache
# and auth environment variables take effect.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
try:
    from gom.utils.env import load_dotenv

    load_dotenv(_REPO_ROOT / ".env" if (_REPO_ROOT / ".env").is_file() else None)
except Exception:
    pass

from vllm import LLM, SamplingParams
from gom.vqa.prompts import PROMPT_PROFILES, build_vqa_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Prompt templates verbatim from the AAAI26 supplementary (Fig 1 / Fig 2).
SYSTEM_VISUAL = (
    "You are a multimodal assistant with spatial reasoning capabilities. "
    "Use the visual scene graph in the image to interpret spatial relations "
    "and answer questions grounded in the visual layout."
)

SYSTEM_VISUAL_TEXTUAL = (
    "You are a multimodal assistant capable of understanding both visual and "
    "textual scene graphs. Use the image and the accompanying graph description "
    "to answer the question accurately."
)

USER_VISUAL = (
    "Answer the question based on the spatial configuration in the image.\n"
    "Question: {question}"
)

USER_VISUAL_TEXTUAL = (
    "Answer the question based on the spatial configuration in the image "
    "and the graph description.\n"
    "Scene Graph (Textual):\n{scene_graph}\n"
    "Question: {question}"
)

# Raw baseline: plain image + question (lmms-eval VQA convention: single word/phrase).
SYSTEM_RAW = "You are a helpful visual assistant."

USER_RAW = "Question: {question}\nAnswer the question using a single word or phrase."


def load_dataset(path: str | Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_prompt(
    question: str,
    mode: str,
    scene_graph_text: Optional[str] = None,
    profile: str = "gom_v2_concise",
) -> tuple[str, str]:
    return build_vqa_prompt(
        mode,
        question,
        scene_graph=scene_graph_text,
        profile=profile,
    )


def format_messages(system: str, user: str, image_path: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": user},
            ],
        },
    ]


def run_inference(
    llm: LLM,
    examples: list[dict],
    image_dir: Optional[str],
    mode: str,
    sampling_params: SamplingParams,
    prompt_profile: str = "gom_v2_concise",
) -> list[dict]:
    results = []

    for i, ex in enumerate(examples):
        if mode == "raw":
            # Baseline always uses the unprocessed image
            image_path = ex.get("image_path") or ex.get("gom_image_path")
        else:
            image_path = ex.get("gom_image_path") or ex.get("image_path")
        full_path = Path(image_dir) / image_path if image_dir else Path(image_path)

        if not full_path.exists():
            log.warning(f"Image not found: {full_path}")
            continue

        question = ex["question"]
        try:
            system, user = build_prompt(
                question,
                mode,
                ex.get("scene_graph_text"),
                prompt_profile,
            )
        except ValueError:
            continue

        messages = format_messages(system, user, f"file://{full_path.resolve()}")
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        prediction = outputs[0].outputs[0].text.strip()

        results.append({
            "image_path": str(image_path),
            "question": question,
            "prediction": prediction,
            "answer": ex.get("answer"),
            "image_id": ex.get("image_id"),
        })

        if (i + 1) % 50 == 0:
            log.info(f"Processed {i + 1}/{len(examples)}")

    return results


def save_results(results: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="VQA inference with vLLM")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--mode", choices=["raw", "visual", "visual_textual"], default="visual_textual")
    parser.add_argument(
        "--prompt-profile",
        choices=PROMPT_PROFILES,
        default="gom_v2_concise",
        help=(
            "Default gom_v2_concise: mark-aware prompt that tells the model the "
            "drawn ID tags and relation words are pointers, not answers. Use "
            "paper_declared for verbatim paper reproduction."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)  # greedy: lmms-eval VQA standard
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[:args.limit]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    results = run_inference(
        llm,
        examples,
        args.image_dir,
        args.mode,
        sampling_params,
        args.prompt_profile,
    )
    save_results(results, args.output)

    # Simple accuracy if ground truth available
    with_gt = [r for r in results if r.get("answer")]
    if with_gt:
        correct = sum(1 for r in with_gt if r["prediction"].lower() == r["answer"].lower())
        log.info(f"Accuracy: {correct}/{len(with_gt)} ({100*correct/len(with_gt):.1f}%)")


if __name__ == "__main__":
    main()
