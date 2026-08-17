#!/usr/bin/env python3
"""Answer-quality gate: scan predictions for mark-vocabulary leakage.

The failure modes this catches are the ones the flip audit found the pipeline
cannot prevent on its own: the model answering with a mark's reference ID, with
an arrow's relation word, or denying an object that is outlined right in front
of it. Run on a small audit root before committing to a full run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_paper"))
from vqa_metrics import gqa_hit, vqa_soft_acc_phrase  # noqa: E402

ID_RE = re.compile(r"\b[a-z][a-z ]*_[0-9]+\b|\b(?:person|object|item)\s+[0-9]+\b", re.I)
REL_RE = re.compile(
    r"^\W*(above|below|left of|right of|in front of|behind|near|next to|"
    r"touching(?:\s+\w+){0,2})\W*$",
    re.I,
)
PLAN_RE = re.compile(r"^(i will|i'll|let me|to determine|first,? i)\b", re.I)
WHO_RE = re.compile(r"^who\b|\bwho is\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--prompt-profile", required=True)
    ap.add_argument("--models", default="gemma3_4b,qwen25_vl_7b,llamav_o1_11b")
    ap.add_argument("--datasets", default="gqa,vqav1,vqav2")
    ap.add_argument(
        "--conditions",
        default="raw,gom_text,gom_numeric,gom_text_labeled,gom_numeric_labeled",
    )
    ap.add_argument("--setting", default="seed0_temp0.2_top_p0.90.jsonl")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    rows = {}
    for dataset in datasets:
        path = args.data_root / "prepared" / dataset / "eval.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row

    def score(pred, row):
        if row["dataset"] == "gqa":
            return gqa_hit(pred, row["answer"])
        return vqa_soft_acc_phrase(pred, row.get("answers") or [row["answer"]])

    print(f"## Leakage gate: {args.data_root} / {args.prompt_profile}\n")
    print("| model | condition | ID leak | rel word | plan | yes->no | acc |")
    print("|---|---|---:|---:|---:|---:|---:|")
    examples, totals = [], {}
    for model in models:
        for condition in conditions:
            preds, hits = [], 0.0
            for dataset in datasets:
                path = (
                    args.data_root / "predictions" / args.prompt_profile / model
                    / dataset / condition / args.setting
                )
                if not path.is_file():
                    continue
                preds.extend(
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                )
            if not preds:
                continue
            id_leak = rel_leak = plan = denial = 0
            for item in preds:
                row = rows[item["question_id"]]
                value = (item["prediction"] or "").strip()
                hits += score(value, row)
                flags = []
                if ID_RE.search(value):
                    id_leak += 1
                    flags.append("ID")
                if condition != "raw" and REL_RE.match(value):
                    rel_leak += 1
                    flags.append("REL")
                if PLAN_RE.match(value):
                    plan += 1
                    flags.append("PLAN")
                if (
                    str(row.get("answer", "")).lower() == "yes"
                    and value.lower().rstrip(".") == "no"
                ):
                    denial += 1
                    flags.append("DENY")
                if flags and condition != "raw":
                    examples.append(
                        (model, condition, "/".join(flags), row["question"][:52], value[:40])
                    )
            acc = 100.0 * hits / len(preds)
            totals[(model, condition)] = (id_leak, rel_leak, plan, denial, acc)
            print(
                f"| {model} | {condition} | {id_leak} | {rel_leak} | {plan} | {denial} | {acc:.1f} |"
            )

    print(f"\nflagged generations shown ({min(len(examples), args.show)} of {len(examples)}):")
    for row in examples[: args.show]:
        print(f"  [{row[2]}] {row[0]} {row[1]}: {row[3]!r} -> {row[4]!r}")

    marked = {k: v for k, v in totals.items() if k[1] != "raw"}
    total_id = sum(v[0] for v in marked.values())
    total_rel = sum(v[1] for v in marked.values())
    total_plan = sum(v[2] for v in marked.values())
    print(
        f"\nTOTAL across marked conditions: ID={total_id} REL={total_rel} PLAN={total_plan}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
