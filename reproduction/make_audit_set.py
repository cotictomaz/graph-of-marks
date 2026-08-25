#!/usr/bin/env python3
"""Build the diversified stress-audit set that gates a full run.

The gom_v2 control set was too easy: it passed, and the full run still produced
119 raw-right/GoM-wrong flips whose causes (open-vocabulary misses, label
overlap, fragment marks, ID leaks on "who" questions) were not represented in
the 49 control rows. This set is stratified over exactly those failure modes, so
a config that regresses any of them fails the gate instead of the full run.

Strata (deterministic - first N by curated-selection order in each bucket):
  who            questions whose answer is a person role (ID-leak pressure)
  existence_yes  yes-gold existence questions (denial pressure)
  existence_no   no-gold existence questions (false-assert pressure)
  leftright      explicit left/right questions (inversion pressure)
  openvocab      question nouns outside the closed visual ontology
  dense          images whose first-pass graph had many detections
  small          questions about small objects
plus every flip case listed in FLIP_EXAMPLES_PAPER_GOM.md (the gom_v3 gallery).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_eval import curate, read_jsonl  # noqa: E402
from question_filter import load_question_intent_module  # noqa: E402

QI = load_question_intent_module()

# (dataset, image stem, question) triples from the gom_v3 flip gallery
# (FLIP_EXAMPLES_PAPER_GOM.md) -- the cases the gom_v4 fixes must move.
FLIP_CASES = [
    ("gqa", "2347957", "Who is wearing a jacket?"),
    ("gqa", "2348675", "Who is holding the racket in the center?"),
    ("gqa", "2380524", "Who is wearing a tee shirt?"),
    ("gqa", "2408238", "Who is wearing a watch?"),
    ("gqa", "2341832", "Do you see benches to the right of the bottle that is not open?"),
    ("gqa", "2315568", "Do you see a ladle next to the computer that is sitting on the floor?"),
    ("gqa", "2389557", "Is the smiling person above a bench?"),
    ("gqa", "2370503", "Is the bicycle behind the tree in the field?"),
    ("gqa", "2403371", "Is the truck to the left or to the right of the bench on the right?"),
    ("gqa", "2361897", "Is the ball to the left or to the right of the man that is wearing socks?"),
    ("gqa", "2367686", "Is the speaker to the right or to the left of the man?"),
    ("gqa", "2315716", "Do you see people to the left of the tall palm trees?"),
    ("gqa", "2376059", "Are there bottles to the right of the doll?"),
    ("gqa", "2378259", "Is the water bottle to the right of a refrigerator?"),
    ("gqa", "2346557", "Are the oranges to the right of the other oranges unpeeled or peeled?"),
    ("gqa", "2372647", "What is the vegetable that is to the left of the sponge?"),
    ("gqa", "713865", "What vehicle is to the left of the vehicle on the sidewalk?"),
    ("gqa", "2319253", "What type of fruit is to the right of the food on the left side?"),
    ("gqa", "2378694", "What is the man wearing?"),
    ("gqa", "2380400", "What is the person in front of the trees throwing?"),
]
QUOTA = {
    "who": 10,
    "existence_yes": 8,
    "existence_no": 6,
    "leftright": 10,
    "openvocab": 10,
    "small": 6,
}
_WHO_RE = re.compile(r"^who\b|\bwho is\b")
_LR_RE = re.compile(r"\bleft\b.*\bright\b|\bright\b.*\bleft\b|\bwhich side\b")


def stratum(row) -> str | None:
    q = row["question"].lower()
    gold = str(row.get("answer", "")).strip().lower()
    intent = QI.parse_question_intent(q)
    if _WHO_RE.search(q):
        return "who"
    if _LR_RE.search(q):
        return "leftright"
    if intent.open_terms:
        return "openvocab"
    if gold == "yes":
        return "existence_yes"
    if gold == "no":
        return "existence_no"
    if re.search(r"\b(small|tiny|little)\b", q):
        return "small"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    for dataset in ("gqa", "vqav1", "vqav2"):
        rows = read_jsonl(args.source_root / "prepared" / dataset / "eval.jsonl")
        selected, used = [], set()

        if dataset == "gqa":
            by_key = {(r["image_id"], r["question"]): r for r in rows}
            for _, stem, question in FLIP_CASES:
                row = by_key.get((stem, question))
                if row is None:
                    print(f"  note: flip case not in eval rows ({stem})")
                    continue
                selected.append(row)
                used.add(row["image_id"])

        curated, _ = curate(rows)
        buckets: dict[str, list] = {}
        for row in curated:
            if row["image_id"] in used:
                continue
            name = stratum(row)
            if name:
                buckets.setdefault(name, []).append(row)
        # vqav1/vqav2 contribute a third of each quota so gqa stays the bulk
        scale = 1.0 if dataset == "gqa" else 0.34
        for name, count in QUOTA.items():
            take = buckets.get(name, [])[: max(1, int(round(count * scale)))]
            for row in take:
                if row["image_id"] in used:
                    continue
                selected.append(row)
                used.add(row["image_id"])

        prepared = args.output_root / "prepared" / dataset
        prepared.mkdir(parents=True, exist_ok=True)
        with (prepared / "eval.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (prepared / "preproc_input.json").write_text(
            json.dumps(
                [
                    {
                        "image_path": r["image_path"],
                        "question": r["question"],
                        "question_id": r["question_id"],
                    }
                    for r in selected
                ],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (prepared / "provenance.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "purpose": "gom_v3 stress-audit gate",
                    "source_root": str(args.source_root),
                    "rows": len(selected),
                    "strata": {
                        name: sum(1 for r in selected if stratum(r) == name)
                        for name in QUOTA
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        counts = {name: sum(1 for r in selected if stratum(r) == name) for name in QUOTA}
        print(f"{dataset}: {len(selected)} rows {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
