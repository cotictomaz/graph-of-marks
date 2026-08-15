#!/usr/bin/env python3
"""Run one Table 2 model over every dataset, condition, and decoding setting."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PAPER_SPEC, first_row_per_image, load_yaml, sha256_file
from gom.vqa.prompts import PROMPT_PROFILES, build_vqa_prompt


REC_SYSTEM = "You are a multimodal assistant capable of understanding visual scene graphs."
REC_USER = (
    "Identify the object ID(s) for the following description(s) based on the "
    "scene graph visualization in the image.\n"
    "Target object description(s): {descriptions}\n"
    "Respond with only the ID(s)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prompt-profile", choices=PROMPT_PROFILES, default="paper_declared")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help=(
            "Cap concurrent sequences. Needed for models whose worst-case "
            "multimodal profile run otherwise exhausts VRAM; 0 keeps vLLM's default."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--one-per-image",
        action="store_true",
        help="Use the first canonical question for each image.",
    )
    parser.add_argument(
        "--single-setting",
        action="store_true",
        help="Run the saved best setting: seed=0, temperature=0.2, top_p=0.9.",
    )
    parser.add_argument(
        "--setting",
        default="",
        help="Run one explicit decoding setting as SEED,TEMPERATURE,TOP_P.",
    )
    parser.add_argument(
        "--artifact-granularity", choices=("image", "question"), default="image"
    )
    parser.add_argument(
        "--fold-system-into-user",
        action="store_true",
        help=(
            "Prepend the system prompt to the user turn. Required by chat "
            "templates that reject a system role alongside an image (Mllama)."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question IDs")
    return rows


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def commit_id() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def attach_paths(
    rows: list[dict],
    preprocessing: Path,
    variant: str | None,
    granularity: str = "image",
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        stem = Path(row["image_path"]).stem
        if granularity == "image":
            index = 1
        else:
            counts[stem] += 1
            index = counts[stem]
        row["render_index"] = index
        row["graph_path"] = str(preprocessing / f"{stem}_q{index}_graph.json")
        row["render_metadata_path"] = str(
            preprocessing / f"{stem}_q{index}_render_variants.json"
        )
        row["inference_image"] = (
            row["image_path"]
            if variant is None
            else str(
                preprocessing
                / "renders"
                / variant
                / f"{stem}_q{index}_output.jpg"
            )
        )


def validate_artifact(row: dict, condition: str, variant: str | None) -> dict[str, Any]:
    image_path = Path(row["inference_image"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if variant is None:
        return {"image_sha256": sha256_file(image_path), "edge_digest": None}
    metadata_path = Path(row["render_metadata_path"])
    graph_path = Path(row["graph_path"])
    if not metadata_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError(metadata_path if not metadata_path.is_file() else graph_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if variant not in metadata:
        raise RuntimeError(f"{metadata_path}: no {variant!r} variant")
    render = metadata[variant]
    if render.get("output_sha256") != sha256_file(image_path):
        raise RuntimeError(f"{condition}: render hash mismatch for {image_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_meta = graph.get("graph", {})
    if render.get("edge_digest") != graph_meta.get("edge_digest"):
        raise RuntimeError(f"{condition}: graph/render edge digest mismatch")
    expected = graph_meta.get("edge_count") if render.get("display_relationships") else 0
    if render.get("rendered_edge_count") != expected:
        raise RuntimeError(f"{condition}: rendered edge count mismatch")
    return {
        "image_sha256": render["output_sha256"],
        "edge_digest": render["edge_digest"],
        "edge_count": render["edge_count"],
        "rendered_edge_count": render["rendered_edge_count"],
    }


def prompt_for(row: dict, dataset: str, condition: str, profile: str) -> tuple[str, str]:
    if dataset == "refcocog":
        descriptions = row.get("descriptions") or [row["question"]]
        return REC_SYSTEM, REC_USER.format(
            descriptions=json.dumps(descriptions, ensure_ascii=False)
        )
    mode = "raw" if condition == "raw" else "visual"
    return build_vqa_prompt(mode, row["question"], profile=profile)


def messages(
    system: str, user: str, image: Path, fold_system: bool = False
) -> list[dict]:
    text = f"{system}\n\n{user}" if fold_system else user
    turn = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{image.resolve()}"}},
            {"type": "text", "text": text},
        ],
    }
    if fold_system:
        # Mllama's chat template raises "Prompting with images is incompatible
        # with system messages", so the system instruction has to ride along in
        # the user turn. Recorded in each prediction sidecar.
        return [turn]
    return [{"role": "system", "content": system}, turn]


def output_path(
    data_root: Path,
    model_key: str,
    dataset: str,
    condition: str,
    seed: int,
    temperature: float,
    top_p: float,
    prompt_profile: str,
) -> Path:
    # The profile owns its own tree: prompt wording changes the generations, so
    # two profiles must never write to the same prediction file.
    return (
        data_root
        / "predictions"
        / prompt_profile
        / model_key
        / dataset
        / condition
        / f"seed{seed}_temp{temperature:.1f}_top_p{top_p:.2f}.jsonl"
    )


def write_checkpoint(path: Path, rows: list[dict], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_existing(path: Path, metadata: dict) -> dict[str, dict]:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not sidecar.is_file():
        return {}
    actual = json.loads(sidecar.read_text(encoding="utf-8"))
    if actual != metadata:
        raise RuntimeError(f"Cannot resume {path}: metadata changed")
    return {row["question_id"]: row for row in read_jsonl(path)}


def main() -> int:
    args = parse_args()
    spec = load_yaml(PAPER_SPEC)
    model = spec["models"].get(args.model_key)
    if model is None:
        raise ValueError(f"Unknown model key {args.model_key!r}")
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    decode = spec["published_decoding"]
    if args.setting:
        seed, temperature, top_p = args.setting.split(",")
        settings = [(int(seed), float(temperature), float(top_p))]
    elif args.single_setting:
        settings = [(0, 0.2, 0.9)]
    else:
        settings = list(
            itertools.product(decode["seeds"], decode["temperatures"], decode["top_p"])
        )

    from vllm import LLM, SamplingParams

    engine_options = {}
    if args.max_num_seqs:
        engine_options["max_num_seqs"] = args.max_num_seqs
    llm = LLM(
        model=model["id"],
        revision=model["revision"],
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path="/",
        enforce_eager=True,
        seed=42,
        **engine_options,
    )
    for dataset in datasets:
        eval_path = args.data_root / "prepared" / dataset / "eval.jsonl"
        template_rows = read_jsonl(eval_path)
        if args.one_per_image:
            template_rows = first_row_per_image(template_rows)
        if args.limit:
            template_rows = template_rows[: args.limit]
        preprocessing = args.data_root / "artifacts" / dataset / "preprocessing"
        for condition, condition_spec in spec["conditions"].items():
            if dataset == "refcocog" and condition in {"raw", "segmented"}:
                continue
            variant = condition_spec["render_variant"]
            rows = [dict(row) for row in template_rows]
            attach_paths(rows, preprocessing, variant, args.artifact_granularity)
            artifact_provenance = {
                row["question_id"]: validate_artifact(row, condition, variant)
                for row in rows
            }
            for seed, temperature, top_p in settings:
                output = output_path(
                    args.data_root,
                    args.model_key,
                    dataset,
                    condition,
                    seed,
                    temperature,
                    top_p,
                    args.prompt_profile,
                )
                metadata = {
                    "schema_version": 1,
                    "git_commit": commit_id(),
                    "paper_spec_sha256": sha256_file(PAPER_SPEC),
                    "dataset_eval_sha256": sha256_file(eval_path),
                    "dataset": dataset,
                    "condition": condition,
                    "render_variant": variant,
                    "model_key": args.model_key,
                    "model_id": model["id"],
                    "model_revision": model["revision"],
                    "prompt_profile": args.prompt_profile,
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": decode["max_tokens"],
                    "row_count": len(rows),
                    "one_per_image": args.one_per_image,
                    "single_setting": args.single_setting,
                }
                if args.fold_system_into_user:
                    # Added only when set, so sidecars written before this option
                    # existed still compare equal and stay resumable.
                    metadata["fold_system_into_user"] = True
                existing = load_existing(output, metadata) if args.resume else {}
                completed: list[dict] = []
                for row in rows:
                    if row["question_id"] in existing:
                        completed.append(existing[row["question_id"]])
                pending = [row for row in rows if row["question_id"] not in existing]
                sampling = SamplingParams(
                    max_tokens=decode["max_tokens"],
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
                for batch in batched(pending, args.batch_size):
                    batch_messages = []
                    for row in batch:
                        system, user = prompt_for(
                            row, dataset, condition, args.prompt_profile
                        )
                        batch_messages.append(
                            messages(
                                system,
                                user,
                                Path(row["inference_image"]),
                                args.fold_system_into_user,
                            )
                        )
                    generated = llm.chat(
                        messages=batch_messages,
                        sampling_params=sampling,
                        use_tqdm=False,
                    )
                    for row, result in zip(batch, generated):
                        completed.append(
                            {
                                "question_id": row["question_id"],
                                "paper_row_index": row["paper_row_index"],
                                "prediction": result.outputs[0].text.strip(),
                                "artifact": artifact_provenance[row["question_id"]],
                            }
                        )
                    completed.sort(key=lambda value: value["paper_row_index"])
                    write_checkpoint(output, completed, metadata)
                print(
                    f"{args.model_key} {dataset} {condition} seed={seed} "
                    f"temp={temperature} top_p={top_p}: {len(completed)}/{len(rows)}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
