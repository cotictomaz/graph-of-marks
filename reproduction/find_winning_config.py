#!/usr/bin/env python3
"""Instance-level search for subsets/configurations where marks beat raw.

Analyses on an existing prediction tree (default: data_v3, direct_concise):
  1. per-question-category deltas (incl. spatial/relational, the paper's core claim)
  2. graph-coverage split (marks can only help when the graph contains the queried objects)
  3. consistent win/loss instance profiling
  4. oracle ceilings — DIAGNOSTIC ONLY: outcome-based selection, not replicable results

Stdlib only. Usage:
  python3 reproduction/find_winning_config.py --data-root reproduction/data_v3 \
      --prompt-profile direct_concise
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]

from common import first_row_per_image
from score_table2 import normalize, official_vqa_score


def _load_question_intent():
    path = ROOT / "src" / "gom" / "question_intent.py"
    spec = importlib.util.spec_from_file_location("_gom_qi", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QI = _load_question_intent()
MODELS = ("gemma3_4b", "qwen25_vl_7b", "llamav_o1_11b")
MARKED = (
    "segmented", "som_numeric", "gom_text",
    "gom_numeric", "gom_text_labeled", "gom_numeric_labeled",
)
DATASETS = ("gqa", "vqav1", "vqav2")
_SUFFIX_RE = re.compile(r"_\d+$")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(pred, row, ds):
    if ds == "gqa":
        return float(normalize(pred) == normalize(row["answer"]))
    return official_vqa_score(pred, row["answers"])


def load_instances(root, profile, model, ds):
    rows = first_row_per_image(read_jsonl(root / "prepared" / ds / "eval.jsonl"))
    gdir = root / "artifacts" / ds / "preprocessing"
    preds = {}
    for cond in ("raw",) + MARKED:
        path = root / "predictions" / profile / model / ds / cond / "seed0_temp0.2_top_p0.90.jsonl"
        preds[cond] = {r["question_id"]: r["prediction"] for r in read_jsonl(path)}
    out = []
    for row in rows:
        q = row["question_id"]
        intent = QI.parse_question_intent(row["question"])
        gpath = gdir / f"{Path(row['image_path']).stem}_q1_graph.json"
        graph = json.loads(gpath.read_text()) if gpath.is_file() else {"nodes": []}
        labels = {
            QI.canonical_object_label(_SUFFIX_RE.sub("", n["label"]))
            for n in graph.get("nodes", [])
        }
        covered = bool(intent.object_terms) and all(
            term in labels for term in intent.object_terms
        )
        rec = {
            "qid": q,
            "category": intent.question_type,
            "relational": bool(intent.relation_terms)
            or intent.question_type == "spatial",
            "covered": covered,
            "has_terms": bool(intent.object_terms),
            "n_nodes": len(graph.get("nodes", [])),
            "scores": {c: score(preds[c][q], row, ds) for c in ("raw",) + MARKED},
        }
        out.append(rec)
    return out


def acc(instances, cond):
    if not instances:
        return float("nan")
    return 100.0 * sum(i["scores"][cond] for i in instances) / len(instances)


def best_marked(instances):
    if not instances:
        return "-", float("nan")
    return max(((c, acc(instances, c)) for c in MARKED), key=lambda x: x[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("reproduction/data_v3"))
    ap.add_argument("--prompt-profile", default="direct_concise")
    args = ap.parse_args()

    all_inst = {}
    for m in MODELS:
        for ds in DATASETS:
            all_inst[(m, ds)] = load_instances(args.data_root, args.prompt_profile, m, ds)

    print("## 1. Per-category delta (best marked - raw), strict\n")
    print("| model | dataset | category | n | raw | best cond | best | delta |")
    print("|---|---|---|---:|---:|---|---:|---:|")
    for (m, ds), inst in all_inst.items():
        cats = sorted({i["category"] for i in inst}) + ["RELATIONAL", "ALL"]
        for cat in cats:
            if cat == "ALL":
                sub = inst
            elif cat == "RELATIONAL":
                sub = [i for i in inst if i["relational"]]
            else:
                sub = [i for i in inst if i["category"] == cat]
            if len(sub) < 30:
                continue
            raw = acc(sub, "raw")
            bc, bv = best_marked(sub)
            flag = " **WIN**" if bv > raw else ""
            print(f"| {m} | {ds} | {cat} | {len(sub)} | {raw:.2f} | {bc} | {bv:.2f} | {bv - raw:+.2f}{flag} |")

    print("\n## 2. Graph-coverage split (questions with object terms only)\n")
    print("| model | dataset | slice | n | raw | best | delta |")
    print("|---|---|---|---:|---:|---:|---:|")
    for (m, ds), inst in all_inst.items():
        withterms = [i for i in inst if i["has_terms"]]
        for name, sub in (
            ("covered", [i for i in withterms if i["covered"]]),
            ("uncovered", [i for i in withterms if not i["covered"]]),
        ):
            if len(sub) < 30:
                continue
            raw = acc(sub, "raw")
            _, bv = best_marked(sub)
            print(f"| {m} | {ds} | {name} | {len(sub)} | {raw:.2f} | {bv:.2f} | {bv - raw:+.2f} |")

    print("\n## 3. Consistent win/loss profiling\n")
    for (m, ds), inst in all_inst.items():
        winners = [
            i for i in inst
            if i["scores"]["raw"] <= 0.1
            and all(i["scores"][c] >= 0.9 for c in MARKED)
        ]
        losers = [
            i for i in inst
            if i["scores"]["raw"] >= 0.9
            and all(i["scores"][c] <= 0.1 for c in MARKED)
        ]
        wc = Counter(i["category"] for i in winners)
        lc = Counter(i["category"] for i in losers)
        print(f"- {m}/{ds}: consistent winners {len(winners)} {dict(wc)} | consistent losers {len(losers)} {dict(lc)}")

    print("\n## 4. ORACLE CEILINGS — diagnostic only, outcome-based, NOT replicable results\n")
    print("| model | dataset | kept frac | raw(kept) | best(kept) | delta | per-inst-oracle delta (full set) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (m, ds), inst in all_inst.items():
        bc, _ = best_marked(inst)
        kept = [i for i in inst if not (i["scores"]["raw"] >= 0.9 and i["scores"][bc] <= 0.1)]
        raw_k, best_k = acc(kept, "raw"), acc(kept, bc)
        oracle = 100.0 * sum(max(i["scores"][c] for c in MARKED) for i in inst) / len(inst)
        raw_full = acc(inst, "raw")
        print(f"| {m} | {ds} | {len(kept)}/{len(inst)} | {raw_k:.2f} | {best_k:.2f} | {best_k - raw_k:+.2f} | {oracle - raw_full:+.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
