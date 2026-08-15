#!/usr/bin/env python3
"""Fail-closed, single-command reproduction harness for paper Table 2."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    MANIFEST_REGISTRY,
    PAPER_SPEC,
    REPRODUCTION_ROOT,
    ROOT,
    canonical_rows,
    first_row_per_image,
    load_yaml,
    sha256_file,
    write_jsonl,
)


ALL_DATASETS = ("gqa", "vqav1", "vqav2", "refcocog")


def absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "datasets",
            "prepare",
            "preprocess",
            "audit",
            "inference",
            "score",
            "vqav2",
            "table2",
            "plan",
        ),
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated override; table2 defaults to all four datasets",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("GOM_PAPER_DATA", str(REPRODUCTION_ROOT / "data"))
        ),
    )
    parser.add_argument(
        "--dataset-archive",
        type=Path,
        default=Path(
            os.environ.get("GOM_DATASET_ARCHIVE", str(ROOT / "data_paper/gom_datasets.zip"))
        ),
    )
    parser.add_argument(
        "--dataset-archive-url",
        default=os.environ.get("GOM_DATASET_ARCHIVE_URL"),
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        default=[],
        metavar="DATASET=PATH",
    )
    parser.add_argument(
        "--fasttext",
        type=Path,
        default=(
            Path(os.environ["GOM_FASTTEXT"])
            if os.environ.get("GOM_FASTTEXT")
            else None
        ),
        help="Converted cc.en.300 gensim KeyedVectors file",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(os.environ.get("GOM_MODEL_CACHE", "~/.cache/gom-paper")),
        help="Persistent host cache mounted into both reproduction containers",
    )
    parser.add_argument("--models", default="gemma3_4b,qwen25_vl_7b,llamav_o1_11b")
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=256,
        help="Requests per vLLM chat call (default: 256, the previous proven run).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM fraction of total VRAM; lower it when the GPU is shared.",
    )
    parser.add_argument(
        "--one-per-image",
        action="store_true",
        help="Score the first canonical question per image only.",
    )
    parser.add_argument(
        "--single-setting",
        action="store_true",
        help="Run seed=0, temperature=0.2, top_p=0.9 instead of the 27-setting grid.",
    )
    parser.add_argument(
        "--fold-system-into-user",
        action="store_true",
        help="Fold the system prompt into the user turn (Mllama chat templates).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="Cap vLLM concurrent sequences; 0 keeps the engine default.",
    )
    parser.add_argument(
        "--container-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment for the container, e.g. VLLM_ATTENTION_BACKEND=XFORMERS.",
    )
    parser.add_argument("--prompt-profile", default="paper_declared")
    parser.add_argument(
        "--setting",
        default="",
        help="Forward one explicit decoding setting (SEED,TEMPERATURE,TOP_P) to inference.",
    )
    parser.add_argument(
        "--render-profile",
        default="paper_aaai26",
        help="Preprocessing render profile (paper_aaai26 or a *_outline/*_lowfill variant).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Forward a row limit to inference (pilot runs against partial artifacts).",
    )
    parser.add_argument(
        "--extra-conditions",
        default="",
        help="Forward extra inference conditions (e.g. text_graph).",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help=(
            "Concurrent full-model preprocessing workers. Use 2 on a GPU with "
            "at least ~24 GB VRAM; image-level artifacts are required."
        ),
    )
    parser.add_argument(
        "--artifact-granularity",
        choices=("image", "question"),
        default="image",
        help=(
            "Preprocessing artifact contract. The paper renders one graph per image; "
            "use question only for the slower question-conditioned ablation."
        ),
    )
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> tuple[str, ...]:
    if args.datasets:
        values = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    elif args.command == "vqav2":
        values = ("vqav2",)
    else:
        values = ALL_DATASETS
    unknown = sorted(set(values) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return values


def image_overrides(args: argparse.Namespace) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in args.image_dir:
        if "=" not in value:
            raise ValueError("--image-dir must use DATASET=PATH")
        dataset, path = value.split("=", 1)
        result[dataset.strip()] = Path(path).expanduser().resolve()
    return result


def registry_entry(dataset: str) -> dict[str, Any]:
    return load_yaml(MANIFEST_REGISTRY)["datasets"][dataset]


def manifest_path(entry: dict[str, Any]) -> Path:
    return REPRODUCTION_ROOT / str(entry["manifest"])


def dataset_image_dir(
    dataset: str, args: argparse.Namespace, entry: dict[str, Any]
) -> Path:
    override = image_overrides(args).get(dataset)
    if override is not None:
        return override
    return (args.data_root / "images" / str(entry["image_subdir"])).resolve()


def preflight(
    args: argparse.Namespace,
    datasets: tuple[str, ...],
    *,
    require_images: bool,
    require_fasttext: bool,
) -> None:
    problems: list[str] = []
    for dataset in datasets:
        entry = registry_entry(dataset)
        path = manifest_path(entry)
        if entry.get("status") != "exact_author_image_split":
            problems.append(
                f"{dataset}: exact author image split is unavailable "
                f"({entry.get('status')})"
            )
            continue
        if not path.is_file():
            problems.append(f"{dataset}: missing manifest {path}")
            continue
        expected_hash = entry.get("sha256")
        actual_hash = sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            problems.append(
                f"{dataset}: manifest hash {actual_hash} != registered {expected_hash}"
            )
        source = json.loads(path.read_text(encoding="utf-8"))
        if len(source) != entry.get("source_rows"):
            problems.append(
                f"{dataset}: {len(source)} source rows != {entry.get('source_rows')}"
            )
        if require_images:
            directory = dataset_image_dir(dataset, args, entry)
            if not directory.is_dir():
                problems.append(f"{dataset}: missing image directory {directory}")
    if require_fasttext:
        if args.fasttext is None or not args.fasttext.expanduser().is_file():
            problems.append(
                "paper_aaai26: --fasttext must point to converted cc.en.300 KeyedVectors"
            )
        elif args.fasttext.suffix != ".kv":
            problems.append("paper_aaai26: --fasttext must use the .kv conversion format")
        else:
            vectors = Path(str(args.fasttext.expanduser()) + ".vectors.npy")
            if not vectors.is_file():
                problems.append(
                    f"paper_aaai26: missing mmap vector companion {vectors}"
                )
    if problems:
        formatted = "\n  - ".join(problems)
        raise RuntimeError(
            "Reproduction preflight failed; substitutions are forbidden:\n  - "
            + formatted
        )


def prepare(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    preflight(args, datasets, require_images=True, require_fasttext=False)
    for dataset in datasets:
        entry = registry_entry(dataset)
        rows, duplicate_count = canonical_rows(
            dataset,
            manifest_path(entry),
            dataset_image_dir(dataset, args, entry),
        )
        expected = entry.get("canonical_rows")
        if expected is not None and len(rows) != expected:
            raise RuntimeError(f"{dataset}: {len(rows)} canonical rows != {expected}")
        if len({row["image_id"] for row in rows}) != entry.get("unique_images"):
            raise RuntimeError(f"{dataset}: unique-image count does not match registry")
        output = args.data_root / "prepared" / dataset
        write_jsonl(rows, output / "eval.jsonl")
        if args.artifact_granularity == "image":
            # One render per image, but conditioned on that image's first canonical
            # question. Dropping the question is NOT equivalent: Algorithm 3
            # (filter_paper_graph) matches object labels against the query, so an
            # empty question matches nothing, falls into its `if not matched` branch,
            # and silently keeps every detected object. Keeping every question
            # instead would rerun SAM/depth per question, which the paper's
            # one-render-per-image contract does not do.
            #
            # RefCOCOg is deliberately excluded: its "question" is the referring
            # expression itself, so conditioning the graph on it would prune the
            # render down to the referent and leak the answer the REC task asks for.
            condition_on_question = dataset != "refcocog"
            if condition_on_question and not args.one_per_image:
                raise ValueError(
                    f"{dataset}: --artifact-granularity image conditions each render on "
                    "the image's first canonical question, which is only valid when "
                    "inference scores one question per image. Pass --one-per-image, or "
                    "use --artifact-granularity question."
                )
            # Same selector inference uses, so each render is conditioned on exactly
            # the question that will be scored against it.
            preproc = []
            for row in first_row_per_image(rows):
                entry_row = {"image_path": row["image_path"]}
                if condition_on_question:
                    entry_row["question"] = row["question"]
                    entry_row["question_id"] = row["question_id"]
                preproc.append(entry_row)
        else:
            preproc = [
                {
                    "image_path": row["image_path"],
                    "question": row["question"],
                    "question_id": row["question_id"],
                }
                for row in rows
            ]
        (output / "preproc_input.json").write_text(
            json.dumps(preproc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "dataset": dataset,
            "manifest": manifest_path(entry).as_posix(),
            "manifest_sha256": sha256_file(manifest_path(entry)),
            "source_rows": len(rows) + duplicate_count,
            "canonical_rows": len(rows),
            "preprocessing_rows": len(preproc),
            "artifact_granularity": args.artifact_granularity,
            "exact_duplicates_removed": duplicate_count,
            "unique_images": len({row["image_id"] for row in rows}),
            "image_split_status": entry["status"],
            "query_status": entry["query_status"],
            "paper_reported_queries_per_image": entry.get(
                "paper_reported_queries_per_image"
            ),
            "observed_queries_per_image": entry.get(
                "observed_queries_per_image"
            ),
            "note": entry.get("note"),
        }
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        print(f"prepared {dataset}: {len(rows)} rows")


def install_datasets(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    overridden = set(image_overrides(args))
    install = tuple(dataset for dataset in datasets if dataset not in overridden)
    if not install:
        return
    command = [
        sys.executable,
        str(REPRODUCTION_ROOT / "prepare_datasets.py"),
        "--archive",
        str(args.dataset_archive.expanduser().resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--datasets",
        ",".join(install),
    ]
    if args.dataset_archive_url:
        command.extend(["--archive-url", args.dataset_archive_url])
    run(command, dry_run=args.dry_run)


def run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def docker_base(image: str, args: argparse.Namespace) -> list[str]:
    model_cache = args.model_cache.expanduser().resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{ROOT}:{ROOT}",
        "-v", f"{args.data_root.resolve()}:{args.data_root.resolve()}",
        "-v", f"{model_cache}:/model-cache",
        "-w", str(ROOT),
        "-e", f"PYTHONPATH={ROOT / 'src'}",
        "-e", "PYTHONUNBUFFERED=1",
    ]
    env_file = ROOT / ".env"
    if env_file.is_file():
        command.extend(["--env-file", str(env_file)])
    command.extend(
        [
            "-e", "HF_HOME=/model-cache",
            "-e", "TORCH_HOME=/model-cache/torch_cache",
            "-e", "FVCORE_CACHE=/model-cache/torch_cache/iopath_cache",
        ]
    )
    for value in args.container_env:
        if "=" not in value:
            raise ValueError("--container-env must use KEY=VALUE")
        command.extend(["-e", value])
    for path in image_overrides(args).values():
        command.extend(["-v", f"{path}:{path}:ro"])
    if args.fasttext is not None:
        fasttext = absolute_without_resolving(args.fasttext)
        fasttext_vectors = Path(str(fasttext) + ".vectors.npy")
        command.extend(["-v", f"{fasttext}:{fasttext}:ro"])
        command.extend(
            ["-v", f"{fasttext_vectors}:{fasttext_vectors}:ro"]
        )
    return command + [image]


def build_images(
    args: argparse.Namespace, *, inference: bool, preprocess: bool = True
) -> None:
    if args.no_build:
        return
    if preprocess:
        run(
            [
                "docker", "build", "-f",
                "reproduction/docker/preprocess.Dockerfile",
                "-t", "gom-paper-preprocess:1", ".",
            ],
            dry_run=args.dry_run,
        )
    if inference:
        run(
            [
                "docker", "build", "-f",
                "reproduction/docker/inference.Dockerfile",
                "-t", "gom-paper-inference:1", ".",
            ],
            dry_run=args.dry_run,
        )


def preprocess(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    preflight(args, datasets, require_images=True, require_fasttext=True)
    prepare(args, datasets)
    fasttext = absolute_without_resolving(args.fasttext)
    fasttext_vectors = Path(str(fasttext) + ".vectors.npy")
    fasttext_provenance = {
        "format": "gensim KeyedVectors",
        "source_model": "cc.en.300.vec",
        "metadata_path": str(fasttext),
        "metadata_sha256": sha256_file(fasttext),
        "vectors_path": str(fasttext_vectors),
        "vectors_sha256": sha256_file(fasttext_vectors),
    }
    provenance_path = args.data_root / "artifacts" / "fasttext.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(fasttext_provenance, indent=2) + "\n", encoding="utf-8"
    )
    build_images(args, inference=False)
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    if args.preprocess_workers == 1:
        for dataset in datasets:
            source = args.data_root / "prepared" / dataset / "preproc_input.json"
            output = args.data_root / "artifacts" / dataset / "preprocessing"
            command = docker_base("gom-paper-preprocess:1", args) + [
                "python3", "src/image_preprocessor.py",
                "--profile", args.render_profile,
                "--json_file", str(source.resolve()),
                "--output_folder", str(output.resolve()),
                "--paper_fasttext_path", str(absolute_without_resolving(args.fasttext)),
                "--output_format", "jpg",
                "--resume" if args.resume else "--verbose",
            ]
            run(command, dry_run=args.dry_run)
    else:
        if args.artifact_granularity != "image":
            raise ValueError(
                "Concurrent preprocessing requires --artifact-granularity image"
            )
        # Each worker gets a disjoint set of unique images and writes to the
        # shared artifact directories.  The workers load independent model
        # copies, so this is intentionally opt-in for VRAM safety.
        chunks: list[list[tuple[str, dict[str, Any]]]] = [
            [] for _ in range(args.preprocess_workers)
        ]
        for dataset in datasets:
            rows = json.loads(
                (args.data_root / "prepared" / dataset / "preproc_input.json")
                .read_text(encoding="utf-8")
            )
            for index, row in enumerate(rows):
                chunks[index % args.preprocess_workers].append((dataset, row))

        commands: list[list[str]] = []
        for worker_index, worker_rows in enumerate(chunks):
            by_dataset: dict[str, list[dict[str, Any]]] = {
                dataset: [] for dataset in datasets
            }
            for dataset, row in worker_rows:
                by_dataset[dataset].append(row)
            script_parts = ["set -e"]
            for dataset in datasets:
                chunk_path = (
                    args.data_root / "prepared" / dataset
                    / f"preproc_input.worker{worker_index}.json"
                )
                chunk_path.write_text(
                    json.dumps(by_dataset[dataset], ensure_ascii=False),
                    encoding="utf-8",
                )
                output = args.data_root / "artifacts" / dataset / "preprocessing"
                script_parts.append(
                    " ".join([
                        "python3", "src/image_preprocessor.py",
                        "--profile", args.render_profile,
                        "--json_file", shlex.quote(str(chunk_path.resolve())),
                        "--output_folder", shlex.quote(str(output.resolve())),
                        "--paper_fasttext_path",
                        shlex.quote(str(absolute_without_resolving(args.fasttext))),
                        "--output_format", "jpg",
                        "--resume" if args.resume else "--verbose",
                    ])
                )
            command = docker_base("gom-paper-preprocess:1", args) + [
                "bash", "-lc", "; ".join(script_parts)
            ]
            commands.append(command)
            print(
                f"+ preprocessing worker {worker_index}: "
                f"{len(worker_rows)} images across {len(datasets)} datasets",
                flush=True,
            )
        if args.dry_run:
            for command in commands:
                run(command, dry_run=True)
        else:
            processes = [
                subprocess.Popen(command, cwd=ROOT)
                for command in commands
            ]
            statuses = [process.wait() for process in processes]
            if any(status != 0 for status in statuses):
                raise RuntimeError(
                    f"Concurrent preprocessing failed with statuses {statuses}"
                )
    verification = docker_base("gom-paper-preprocess:1", args) + [
        "python3",
        "reproduction/verify_weights.py",
        "--cache",
        "/model-cache",
        "--output",
        str((args.data_root / "artifacts" / "preprocessing_weights.json").resolve()),
    ]
    run(verification, dry_run=args.dry_run)


def audit(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    for dataset in datasets:
        command = [
            sys.executable,
            str(REPRODUCTION_ROOT / "audit_relations.py"),
            "--eval", str(args.data_root / "prepared" / dataset / "eval.jsonl"),
            "--preprocessing", str(args.data_root / "artifacts" / dataset / "preprocessing"),
            "--output", str(args.data_root / "artifacts" / dataset / "relation_audit.json"),
            "--artifact-granularity", args.artifact_granularity,
        ]
        run(command, dry_run=args.dry_run)


def inference(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    if args.inference_batch_size < 1:
        raise ValueError("--inference-batch-size must be >= 1")
    build_images(args, inference=True, preprocess=False)
    spec = load_yaml(PAPER_SPEC)
    settings = list(
        itertools.product(
            spec["published_decoding"]["seeds"],
            spec["published_decoding"]["temperatures"],
            spec["published_decoding"]["top_p"],
        )
    )
    assert len(settings) == 27
    for model_key in [value.strip() for value in args.models.split(",") if value.strip()]:
        if model_key not in spec["models"]:
            raise ValueError(f"Unknown model key: {model_key}")
        command = docker_base("gom-paper-inference:1", args) + [
            "python3", "reproduction/run_table2.py",
            "--data-root", str(args.data_root.resolve()),
            "--datasets", ",".join(datasets),
            "--model-key", model_key,
            "--prompt-profile", args.prompt_profile,
            "--artifact-granularity", args.artifact_granularity,
            "--batch-size", str(args.inference_batch_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        if args.max_num_seqs:
            command.extend(["--max-num-seqs", str(args.max_num_seqs)])
        if args.fold_system_into_user:
            command.append("--fold-system-into-user")
        if args.one_per_image:
            command.append("--one-per-image")
        if args.single_setting:
            command.append("--single-setting")
        if args.setting:
            command.extend(["--setting", args.setting])
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        if args.extra_conditions:
            command.extend(["--extra-conditions", args.extra_conditions])
        if args.resume:
            command.append("--resume")
        run(command, dry_run=args.dry_run)


def score(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    command = [
        sys.executable,
        str(REPRODUCTION_ROOT / "score_table2.py"),
        "--data-root", str(args.data_root.resolve()),
        "--datasets", ",".join(datasets),
        "--models", args.models,
        "--output", str(
            args.data_root / f"table2_report.{args.prompt_profile}.json"
        ),
        "--artifact-granularity", args.artifact_granularity,
        "--prompt-profile", args.prompt_profile,
    ]
    if args.one_per_image:
        command.append("--one-per-image")
    if args.single_setting:
        command.append("--single-setting")
    run(command, dry_run=args.dry_run)


def print_plan() -> None:
    spec = load_yaml(PAPER_SPEC)
    combinations = list(
        itertools.product(
            spec["published_decoding"]["seeds"],
            spec["published_decoding"]["temperatures"],
            spec["published_decoding"]["top_p"],
        )
    )
    registry = load_yaml(MANIFEST_REGISTRY)["datasets"]
    rows_per_model = sum(
        int(registry[dataset]["canonical_rows"])
        * (5 if dataset == "refcocog" else len(spec["conditions"]))
        for dataset in ALL_DATASETS
    )
    print(json.dumps({
        "datasets": {
            dataset: {
                "images": registry[dataset]["unique_images"],
                "queries": registry[dataset]["canonical_rows"],
                "image_split_status": registry[dataset]["status"],
                "query_status": registry[dataset]["query_status"],
            }
            for dataset in ALL_DATASETS
        },
        "models": spec["models"],
        "conditions": list(spec["conditions"]),
        "decoding_runs": len(combinations),
        "condition_runs": 3 * (3 * 7 + 5) * len(combinations),
        "generations_per_model": rows_per_model * len(combinations),
        "total_generations": 3 * rows_per_model * len(combinations),
    }, indent=2))


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.model_cache = args.model_cache.expanduser().resolve()
    datasets = selected_datasets(args)
    if args.command == "plan":
        print_plan()
        return 0
    if args.command == "preflight":
        preflight(args, datasets, require_images=False, require_fasttext=False)
    elif args.command == "datasets":
        install_datasets(args, datasets)
    elif args.command == "prepare":
        prepare(args, datasets)
    elif args.command == "preprocess":
        preprocess(args, datasets)
    elif args.command == "audit":
        audit(args, datasets)
    elif args.command == "inference":
        inference(args, datasets)
    elif args.command == "score":
        score(args, datasets)
    elif args.command in {"vqav2", "table2"}:
        install_datasets(args, datasets)
        preflight(args, datasets, require_images=True, require_fasttext=True)
        preprocess(args, datasets)
        audit(args, datasets)
        inference(args, datasets)
        score(args, datasets)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
