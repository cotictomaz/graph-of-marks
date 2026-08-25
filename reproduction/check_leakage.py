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

GENERIC_TAG_RE = re.compile(
    r"\b(?:person|people|object|item|thing|entity)[ _][0-9]+\b", re.IGNORECASE
)
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
    # Calibrated BETWEEN the two measured runs, and stated as such: over the same
    # 2,975 curated rows the worst condition (qwen, text tags) scores 2.25% generic
    # leaks under gom_v3 and 0.81% under gom_v4. 1.5% therefore passes gom_v4 with
    # headroom and fails a regression back toward gom_v3. It is a regression gate,
    # not an absolute quality bar.
    ap.add_argument("--max-generic-rate", type=float, default=1.5)
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
    print(
        "| model | condition | ID leak | who leak | generic leak | rel word "
        "| plan | yes->no | acc |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
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
            id_leak = rel_leak = plan = denial = who_leak = generic_leak = 0
            for item in preds:
                row = rows[item["question_id"]]
                value = (item["prediction"] or "").strip()
                row_score = score(value, row)
                hits += row_score
                flags = []
                if ID_RE.search(value):
                    id_leak += 1
                    flags.append("ID")
                    # "who ...?" is where the leak is fatal: the gold is always a
                    # person subtype, so a person_N answer scores 0 while a man_N
                    # answer normalizes to "man 1" and the phrase scorer accepts
                    # it. Only the scoring-zero leaks are the defect -- a tag that
                    # happens to carry the right class name costs nothing.
                    if (
                        row["question"].strip().lower().startswith("who")
                        and row_score == 0.0
                    ):
                        who_leak += 1
                        # The defect worth gating is a tag that carries no answer
                        # at all. `man_2` names what the model sees -- it is wrong
                        # only when the gold wants a role noun ("chef", "umpire")
                        # that no detector emits, which is a ceiling, not a render
                        # bug. `person_1` is the real leak: generic, answer-less,
                        # and fixable by naming the mark properly.
                        if GENERIC_TAG_RE.search(value):
                            generic_leak += 1
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
            totals[(model, condition)] = (
                id_leak, rel_leak, plan, denial, acc, generic_leak, who_leak
            )
            print(
                f"| {model} | {condition} | {id_leak} | {who_leak} | {generic_leak} "
                f"| {rel_leak} | {plan} | {denial} | {acc:.1f} |"
            )

    print(f"\nflagged generations shown ({min(len(examples), args.show)} of {len(examples)}):")
    for row in examples[: args.show]:
        print(f"  [{row[2]}] {row[0]} {row[1]}: {row[3]!r} -> {row[4]!r}")

    marked = {k: v for k, v in totals.items() if k[1] != "raw"}
    total_id = sum(v[0] for v in marked.values())
    total_rel = sum(v[1] for v in marked.values())
    total_plan = sum(v[2] for v in marked.values())
    worst_generic = max((v[5] for v in marked.values()), default=0)
    worst_who = max((v[6] for v in marked.values()), default=0)
    # Rate, not raw count: the threshold was first calibrated on a 95-row smoke set
    # and comparing that count against a 2,975-row run is meaningless. Measured
    # generic-leak rates for the worst condition (qwen, text tags): gom_v3 2.25%,
    # gom_v4 0.81% -- the specific-label fix cut it by ~65%. Numeric-ID conditions
    # are 0 in both. The gate catches a regression toward the gom_v3 rate.
    rows_per_condition = max(1, len(rows))
    worst_generic_rate = 100.0 * worst_generic / rows_per_condition
    print(
        f"\nTOTAL across marked conditions: ID={total_id} REL={total_rel} PLAN={total_plan}"
        f"\n  worst who-leak per condition={worst_who} (informational: includes tags "
        f"that DO name a person, e.g. man_2 against gold 'chef')"
        f"\n  worst generic-tag leak={worst_generic} of {rows_per_condition} rows "
        f"= {worst_generic_rate:.2f}% (gate: <= {args.max_generic_rate:.2f}%)"
    )
    failed = worst_generic_rate > args.max_generic_rate
    print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
