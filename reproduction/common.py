"""Shared validation and provenance helpers for the Table 2 reproduction."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_ROOT = Path(__file__).resolve().parent
PAPER_SPEC = REPRODUCTION_ROOT / "paper_spec.yaml"
MANIFEST_REGISTRY = REPRODUCTION_ROOT / "manifests.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return value


def stable_question_id(dataset: str, image_name: str, question: str) -> str:
    payload = f"{dataset}\0{image_name}\0{question}".encode("utf-8")
    return f"{dataset}_{hashlib.sha1(payload).hexdigest()[:20]}"


def majority_answer(answers: Iterable[str]) -> str:
    values = [str(answer).strip() for answer in answers]
    counts = Counter(values)
    first = {answer: values.index(answer) for answer in counts}
    return max(counts, key=lambda answer: (counts[answer], -first[answer]))


def canonical_rows(
    dataset: str,
    manifest: Path,
    image_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"{manifest}: expected a JSON list")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise ValueError(f"{manifest}: row {index} is not an object")
        image_name = Path(str(item.get("image_path", ""))).name
        targets = item.get("targets") if dataset == "refcocog" else None
        if targets is not None:
            if not isinstance(targets, list) or not targets:
                raise ValueError(f"{manifest}: row {index} has no REC targets")
            descriptions = []
            normalized_targets = []
            for target_index, target in enumerate(targets):
                if not isinstance(target, dict):
                    raise ValueError(
                        f"{manifest}: row {index} target {target_index} is not an object"
                    )
                description = str(target.get("description", "")).strip()
                bbox = target.get("bbox_xywh") or target.get("bbox")
                if not description or not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(
                        f"{manifest}: row {index} target {target_index} is incomplete"
                    )
                descriptions.append(description)
                normalized_targets.append(
                    {**target, "description": description, "bbox_xywh": bbox}
                )
            question = "Target object descriptions: " + "; ".join(descriptions)
        else:
            descriptions = None
            normalized_targets = None
            question = str(
                item.get("question")
                or item.get("description")
                or item.get("expression")
                or ""
            ).strip()
        if not image_name or not question:
            raise ValueError(f"{manifest}: row {index} has no image or query")
        key = (image_name, question)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing benchmark image: {image_path}")
        answers = item.get("answers")
        if dataset in {"vqav1", "vqav2"}:
            if not isinstance(answers, list) or len(answers) != 10:
                raise ValueError(
                    f"{manifest}: row {index} requires exactly 10 annotator answers"
                )
        answer = item.get("answer")
        if answer is None and isinstance(answers, list) and answers:
            answer = majority_answer(answers)
        row = {
            "dataset": dataset,
            "paper_row_index": index,
            "image_id": image_path.stem,
            "image_path": image_path.resolve().as_posix(),
            "question": question,
            "question_id": stable_question_id(dataset, image_name, question),
        }
        if isinstance(answers, list):
            row["answers"] = [str(value).strip() for value in answers]
        if answer is not None:
            row["answer"] = str(answer).strip()
        bbox = item.get("bbox_xywh") or item.get("bbox")
        if bbox is not None:
            row["bbox_xywh"] = bbox
        if normalized_targets is not None:
            row["descriptions"] = descriptions
            row["targets"] = normalized_targets
        rows.append(row)
    return rows, duplicates


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_row_per_image(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the first canonical question for each image, preserving order."""
    seen: set[str] = set()
    selected = []
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        selected.append(row)
    return selected
