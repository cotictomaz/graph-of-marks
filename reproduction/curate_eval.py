#!/usr/bin/env python3
"""Curate the eval set: one deliberately chosen question per image, locked.

The historical runs evaluated each image's *first* canonical question, which
mixes in appearance questions the mark overlay destroys, subjective questions
the image cannot answer, and ambiguous referents. This script defines the eval
set once and for all, deterministically and a-priori (no model outputs):

  1. drop questions where question_filter.curation_reason fires;
  2. rank survivors: relational/spatial > yes_no/count > identity > open;
  3. keep the first candidate (canonical order) in the best rank;
  4. drop the image if nothing survives.

It rewrites prepared/<ds>/eval.jsonl and preproc_input.json in place (the
selected question conditions the render), appends a "curation" block to
provenance.json, writes <data_root>/curation_report.md, and locks the selected
question_ids in reproduction/manifests/<ds>_curated_v1.txt (committed; .txt
because .gitignore blocks *.json). Re-runs must reproduce the committed lists:
--check verifies without writing, --update is required to change a lock.

Run AFTER `reproduce.py prepare` and BEFORE preprocess (which must then be
invoked with --skip-prepare). RefCOCOg is never curated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from question_filter import curation_reason, load_question_intent_module  # noqa: E402

QI = load_question_intent_module()

DATASETS = ("gqa", "vqav1", "vqav2")
RANK_NAMES = ("relational_spatial", "yes_no_count", "identity", "open")
LOCK_DIR = Path(__file__).resolve().parent / "manifests"


def rank_of(question: str) -> int:
    intent = QI.parse_question_intent(question)
    if intent.relation_terms or intent.question_type == "spatial":
        return 0
    if intent.question_type in {"yes_no", "count"}:
        return 1
    if intent.question_type == "identity":
        return 2
    return 3


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def curate(rows: list[dict]) -> tuple[list[dict], dict]:
    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(row["image_id"], []).append(row)
    selected: list[dict] = []
    stats = {
        "images_total": len(by_image),
        "questions_total": len(rows),
        "drop_reasons": {},
        "rank_counts": dict.fromkeys(RANK_NAMES, 0),
        "images_dropped": [],
    }
    for image_id, candidates in by_image.items():
        best: tuple[int, dict] | None = None
        for row in candidates:
            answers = row.get("answers") or (
                [row["answer"]] if row.get("answer") else []
            )
            reason = curation_reason(row["question"], answers)
            if reason is not None:
                stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
                continue
            rank = rank_of(row["question"])
            if best is None or rank < best[0]:
                best = (rank, row)
        if best is None:
            stats["images_dropped"].append(image_id)
            continue
        stats["rank_counts"][RANK_NAMES[best[0]]] += 1
        selected.append(best[1])
    return selected, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the selection matches the committed lock lists; write nothing.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Allow overwriting a committed lock list that differs.",
    )
    args = parser.parse_args()
    datasets = [v.strip() for v in args.datasets.split(",") if v.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Cannot curate {unknown}; only {DATASETS} are curatable")

    report_lines = ["# Curated eval set (v1)\n"]
    failures = []
    for dataset in datasets:
        prepared = args.data_root / "prepared" / dataset
        rows = read_jsonl(prepared / "eval.jsonl")
        selected, stats = curate(rows)
        ids = [row["question_id"] for row in selected]
        lock_path = LOCK_DIR / f"{dataset}_curated_v1.txt"

        if lock_path.is_file():
            committed = lock_path.read_text().split()
            if committed != ids:
                message = (
                    f"{dataset}: selection ({len(ids)} ids) differs from committed "
                    f"{lock_path.name} ({len(committed)} ids)"
                )
                if args.check or not args.update:
                    failures.append(message + ("" if args.check else "; pass --update to overwrite"))
                    continue
        elif args.check:
            failures.append(f"{dataset}: no committed lock list {lock_path.name}")
            continue

        if args.check:
            print(f"{dataset}: OK ({len(ids)} ids match {lock_path.name})")
            continue

        lock_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        with (prepared / "eval.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        preproc = [
            {
                "image_path": row["image_path"],
                "question": row["question"],
                "question_id": row["question_id"],
            }
            for row in selected
        ]
        (prepared / "preproc_input.json").write_text(
            json.dumps(preproc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        provenance_path = prepared / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["curation"] = {
            "version": "curated_v1",
            "lock_list": lock_path.name,
            "images_kept": len(selected),
            "images_total": stats["images_total"],
            "question_drop_reasons": stats["drop_reasons"],
            "rank_counts": stats["rank_counts"],
        }
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        report_lines.append(f"## {dataset}\n")
        report_lines.append(
            f"- images kept: {len(selected)}/{stats['images_total']} "
            f"(from {stats['questions_total']} candidate questions)"
        )
        report_lines.append(f"- selection by rank: {stats['rank_counts']}")
        report_lines.append(f"- question drop reasons: {stats['drop_reasons']}")
        dropped = stats["images_dropped"]
        report_lines.append(
            f"- images dropped (no acceptable question): {len(dropped)}"
            + (f" — e.g. {dropped[:10]}" if dropped else "")
        )
        report_lines.append("")
        print(
            f"{dataset}: kept {len(selected)}/{stats['images_total']} images, "
            f"ranks {stats['rank_counts']}, dropped questions {sum(stats['drop_reasons'].values())}"
        )

    if failures:
        for message in failures:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    if not args.check:
        (args.data_root / "curation_report.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
