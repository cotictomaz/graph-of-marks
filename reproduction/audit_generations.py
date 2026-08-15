#!/usr/bin/env python3
"""Generation-level audit of a Table 2 run.

Answers, from the actual generations rather than aggregate scores:
  1. Are generations truncated / non-answers? (stop-behavior audit)
  2. How much of LlamaV's marked deficit is answer format? (lenient rescore)
  3. What do raw-right -> marked-wrong flips look like? (taxonomy vs graph marks)
  4. Which appearance questions does the filter miss? (false negatives)

Stdlib only; reads a data root produced by the reproduction pipeline.
Usage: python3 reproduction/audit_generations.py --data-root reproduction/data_v2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_paper"))

from common import first_row_per_image
from question_filter import COLOR_WORDS, appearance_reason
from score_table2 import normalize, official_vqa_score
from vqa_metrics import gqa_hit, vqa_soft_acc_phrase

MODELS = ("gemma3_4b", "qwen25_vl_7b", "llamav_o1_11b")
CONDS = (
    "raw", "segmented", "som_numeric", "gom_text",
    "gom_numeric", "gom_text_labeled", "gom_numeric_labeled",
)
DATASETS = ("gqa", "vqav1", "vqav2")
PLAN_RE = re.compile(r"^(i will|i'll|let me|to determine|first,? i)\b", re.I)
YES = {"yes"}; NO = {"no"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def preds(root, profile, model, ds, cond, setting="seed0_temp0.2_top_p0.90"):
    path = root / "predictions" / profile / model / ds / cond / f"{setting}.jsonl"
    return {r["question_id"]: r["prediction"] for r in read_jsonl(path)}


def is_right(pred, row, ds):
    if ds == "gqa":
        return normalize(pred) == normalize(row["answer"])
    return official_vqa_score(pred, row["answers"]) >= 0.9


def is_wrong(pred, row, ds):
    if ds == "gqa":
        return normalize(pred) != normalize(row["answer"])
    return official_vqa_score(pred, row["answers"]) <= 0.1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("reproduction/data_v2"))
    ap.add_argument("--prompt-profile", default="supplementary_concise")
    args = ap.parse_args()
    root = args.data_root

    rows_by = {}
    graphs_by = {}
    for ds in DATASETS:
        rows = first_row_per_image(read_jsonl(root / "prepared" / ds / "eval.jsonl"))
        rows_by[ds] = {r["question_id"]: r for r in rows}
        gdir = root / "artifacts" / ds / "preprocessing"
        for r in rows:
            stem = Path(r["image_path"]).stem
            gpath = gdir / f"{stem}_q1_graph.json"
            graphs_by[r["question_id"]] = json.loads(gpath.read_text()) if gpath.is_file() else None

    print("## 1. Stop-behavior audit (plan-only / empty / >=50 words, per 1000)\n")
    print("| model | dataset | condition | plan-only | empty | >=50w |")
    print("|---|---|---|---:|---:|---:|")
    for m in MODELS:
        for ds in DATASETS:
            for c in CONDS:
                p = preds(root, args.prompt_profile, m, ds, c)
                plan = sum(1 for v in p.values() if PLAN_RE.match(v.strip()))
                empty = sum(1 for v in p.values() if not v.strip())
                long = sum(1 for v in p.values() if len(v.split()) >= 50)
                if plan or empty or long:
                    print(f"| {m} | {ds} | {c} | {plan} | {empty} | {long} |")

    print("\n## 2. LlamaV lenient rescore (exact/official vs phrase-containment)\n")
    print("| dataset | condition | strict | lenient | recovered |")
    print("|---|---|---:|---:|---:|")
    for ds in DATASETS:
        rows = rows_by[ds]
        for c in CONDS:
            p = preds(root, args.prompt_profile, "llamav_o1_11b", ds, c)
            if ds == "gqa":
                strict = 100 * sum(normalize(v) == normalize(rows[q]["answer"]) for q, v in p.items()) / len(p)
                lenient = 100 * sum(gqa_hit(v, rows[q]["answer"]) for q, v in p.items()) / len(p)
            else:
                strict = 100 * sum(official_vqa_score(v, rows[q]["answers"]) for q, v in p.items()) / len(p)
                lenient = 100 * sum(vqa_soft_acc_phrase(v, rows[q]["answers"]) for q, v in p.items()) / len(p)
            print(f"| {ds} | {c} | {strict:.2f} | {lenient:.2f} | {lenient - strict:+.2f} |")

    print("\n## 3. Flip taxonomy: raw-right -> segmented-wrong (kept = survives appearance filter)\n")
    print("| model | dataset | flips | kept-flips | color_shift | label_leak | count=marks | yes->no | other |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    examples = {}
    for m in MODELS:
        for ds in DATASETS:
            rows = rows_by[ds]
            raw = preds(root, args.prompt_profile, m, ds, "raw")
            seg = preds(root, args.prompt_profile, m, ds, "segmented")
            cats = {"color_shift": 0, "label_leak": 0, "count=marks": 0, "yes->no": 0, "other": 0}
            flips = kept_flips = 0
            for q, row in rows.items():
                if not (is_right(raw[q], row, ds) and is_wrong(seg[q], row, ds)):
                    continue
                flips += 1
                answers = row.get("answers") or [row.get("answer", "")]
                kept = appearance_reason(row["question"], answers) is None
                if not kept:
                    continue
                kept_flips += 1
                pn = normalize(seg[q])
                tokens = set(pn.split())
                g = graphs_by.get(q)
                labels = {re.sub(r"_\d+$", "", n["label"]) for n in (g or {}).get("nodes", [])}
                gold = normalize(answers[0])
                cat = "other"
                if tokens & COLOR_WORDS and not (set(gold.split()) & tokens & COLOR_WORDS):
                    cat = "color_shift"
                elif pn in labels and pn != gold:
                    cat = "label_leak"
                elif re.search(r"\bhow many\b", row["question"].lower()) and g is not None and pn.isdigit() and int(pn) == len(g.get("nodes", [])) and pn != gold:
                    cat = "count=marks"
                elif gold in YES and pn in NO:
                    cat = "yes->no"
                cats[cat] += 1
                examples.setdefault((m, ds, cat), []).append(
                    (row["question"][:60], gold[:20], normalize(raw[q])[:20], pn[:40], Path(row["image_path"]).stem)
                )
            print(f"| {m} | {ds} | {flips} | {kept_flips} | {cats['color_shift']} | {cats['label_leak']} | {cats['count=marks']} | {cats['yes->no']} | {cats['other']} |")

    print("\n## 4. Filter false negatives (kept questions with color-word gold)\n")
    print("| dataset | kept | color-gold kept |")
    print("|---|---:|---:|")
    for ds in DATASETS:
        kept = fn = 0
        for q, row in rows_by[ds].items():
            answers = row.get("answers") or [row.get("answer", "")]
            if appearance_reason(row["question"], answers) is not None:
                continue
            kept += 1
            gold_tokens = set(normalize(answers[0]).split())
            if gold_tokens & COLOR_WORDS:
                fn += 1
        print(f"| {ds} | {kept} | {fn} |")

    print("\n## Examples per category (question | gold | raw | segmented | stem)\n")
    for (m, ds, cat), ex in sorted(examples.items()):
        if m != "qwen25_vl_7b" or cat == "other":
            continue
        print(f"### {m} {ds} {cat}")
        for e in ex[:5]:
            print(f"- {e[0]} | {e[1]} | {e[2]} | {e[3]} | `{e[4]}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
