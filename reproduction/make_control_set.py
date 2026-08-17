#!/usr/bin/env python3
"""Build the ~50-row control set for the pre-run visual QA gate.

Contents: the 12 flip-case (image, question) pairs from
FLIP_EXAMPLES_PAPER_GOM.md (minus any the curation filter rejects) plus a
deterministic stratified sample of curated rows per dataset x question-rank
bucket. Writes prepared/{gqa,vqav1,vqav2}/{eval.jsonl,preproc_input.json} into
a fresh data root; preprocess it with `reproduce.py preprocess --skip-prepare`.

Source rows come from an existing prepared root's FULL eval.jsonl (any root
where `reproduce.py prepare` ran; the rows are canonical and root-independent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from curate_eval import curate, rank_of, read_jsonl, RANK_NAMES
from question_filter import curation_reason

# (image stem, verbatim question) from FLIP_EXAMPLES_PAPER_GOM.md — all GQA.
FLIP_CASES = (
    ("2354833", "On which side of the picture are the shelves?"),
    ("2335852", "Is the man to the right or to the left of the elephant?"),
    ("2333988", "Who is wearing a jacket?"),
    ("2392912", "Who is wearing the shirt?"),
    ("2359506", "That snow is where?"),
    ("2321902", "Is the child to the right or to the left of the woman that is to the right of the man?"),
    ("2382290", "What is the man wearing?"),
    ("2401706", "What kind of animal is to the left of the zebra that is eating grass?"),
    ("2316593", "What is the small item of furniture called?"),
    ("2412283", "Is the bookcase to the left of the chair that is not antique?"),
    ("2387622", "Are there any benches near the sidewalk?"),
    ("2396350", "Where is the dog?"),
)

# rows per rank bucket, per dataset (deterministic: curated-selection order)
STRATA = {
    "gqa": {"relational_spatial": 6, "yes_no_count": 4, "identity": 2, "open": 2},
    "vqav1": {"relational_spatial": 4, "yes_no_count": 4, "identity": 2, "open": 2},
    "vqav2": {"relational_spatial": 4, "yes_no_count": 4, "identity": 2, "open": 2},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Existing data root whose prepared/<ds>/eval.jsonl is the FULL canonical set.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for dataset in ("gqa", "vqav1", "vqav2"):
        rows = read_jsonl(args.source_root / "prepared" / dataset / "eval.jsonl")
        selected_rows: list[dict] = []
        used_images: set[str] = set()

        if dataset == "gqa":
            by_key = {(row["image_id"], row["question"]): row for row in rows}
            for stem, question in FLIP_CASES:
                row = by_key.get((stem, question))
                if row is None:
                    raise SystemExit(f"flip case not found in eval rows: {stem} / {question}")
                answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
                if curation_reason(row["question"], answers) is not None:
                    print(f"skip flip case rejected by curation: {stem} ({question[:50]})")
                    continue
                selected_rows.append(row)
                used_images.add(row["image_id"])

        curated, _ = curate(rows)
        buckets = dict.fromkeys(RANK_NAMES)
        for name in buckets:
            buckets[name] = []
        for row in curated:
            if row["image_id"] in used_images:
                continue
            buckets[RANK_NAMES[rank_of(row["question"])]].append(row)
        for name, count in STRATA[dataset].items():
            take = buckets[name][:count]
            selected_rows.extend(take)
            used_images.update(row["image_id"] for row in take)

        prepared = args.output_root / "prepared" / dataset
        prepared.mkdir(parents=True, exist_ok=True)
        with (prepared / "eval.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        preproc = [
            {
                "image_path": row["image_path"],
                "question": row["question"],
                "question_id": row["question_id"],
            }
            for row in selected_rows
        ]
        (prepared / "preproc_input.json").write_text(
            json.dumps(preproc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (prepared / "provenance.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "purpose": "control set for visual QA gate",
                    "source_root": str(args.source_root),
                    "rows": len(selected_rows),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{dataset}: {len(selected_rows)} control rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
