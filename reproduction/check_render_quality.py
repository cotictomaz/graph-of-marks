#!/usr/bin/env python3
"""Programmatic render-quality gate for a preprocessed data root.

Checks the three things the flip audit showed we cannot verify by eyeballing a
handful of renders:
  1. label_overlap_count == 0 on every variant of every image;
  2. the object the question is about is actually marked (query coverage);
  3. no image exceeds the mark/arrow budget.

Exits non-zero if a hard check fails, so it can gate a run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from question_filter import load_question_intent_module  # noqa: E402

QI = load_question_intent_module()
_SUFFIX_RE = re.compile(r"_\d+$")


def label_terms(graph: dict) -> set[str]:
    terms = set()
    for node in graph.get("nodes", []):
        label = str(node.get("label", ""))
        if label == "scene":
            continue
        base = _SUFFIX_RE.sub("", label).lower()
        terms.add(base)
        terms.update(base.split())
        terms.add(QI.canonical_object_label(base))
    return {t for t in terms if t}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--datasets", default="gqa,vqav1,vqav2")
    ap.add_argument("--max-marks", type=int, default=8)
    ap.add_argument("--max-arrows", type=int, default=6)
    args = ap.parse_args()

    overlaps, missing_meta, budget, coverage_rows = [], [], [], []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        prepared = args.data_root / "prepared" / dataset / "eval.jsonl"
        if not prepared.is_file():
            continue
        rows = [json.loads(l) for l in prepared.read_text().splitlines() if l.strip()]
        art = args.data_root / "artifacts" / dataset / "preprocessing"
        for row in rows:
            stem = Path(row["image_path"]).stem
            graph_path = art / f"{stem}_q1_graph.json"
            variants_path = art / f"{stem}_q1_render_variants.json"
            if not graph_path.is_file() or not variants_path.is_file():
                missing_meta.append(f"{dataset}/{stem}: artifacts missing")
                continue
            graph = json.loads(graph_path.read_text())
            variants = json.loads(variants_path.read_text())

            for name, info in variants.items():
                count = info.get("label_overlap_count")
                if count is None:
                    missing_meta.append(f"{dataset}/{stem}/{name}: no label_overlap_count")
                elif count > 0:
                    overlaps.append(f"{dataset}/{stem}/{name}: {count} overlapping pairs")

            marks = len([n for n in graph.get("nodes", []) if n.get("label") != "scene"])
            arrows = max(
                (v.get("rendered_edge_count") or 0) for v in variants.values()
            )
            if marks > args.max_marks or arrows > args.max_arrows:
                budget.append(f"{dataset}/{stem}: {marks} marks, {arrows} arrows")

            intent = QI.parse_question_intent(row["question"])
            wanted = {
                QI.canonical_object_label(t)
                for t in (*intent.object_terms, *intent.open_terms)
            }
            wanted = {w for w in wanted if w}
            if wanted:
                have = label_terms(graph)
                hit = any(
                    w in have or any(w in h or h in w for h in have) for w in wanted
                )
                coverage_rows.append((hit, dataset, stem, row["question"], sorted(wanted)))

    print(f"## Render quality gate: {args.data_root}\n")
    print(f"label overlaps      : {len(overlaps)} (must be 0)")
    for line in overlaps[:15]:
        print(f"   {line}")
    print(f"missing metadata    : {len(missing_meta)}")
    for line in missing_meta[:10]:
        print(f"   {line}")
    print(f"over budget         : {len(budget)} (marks>{args.max_marks} or arrows>{args.max_arrows})")
    for line in budget[:10]:
        print(f"   {line}")

    if coverage_rows:
        hits = sum(1 for row in coverage_rows if row[0])
        print(
            f"query coverage      : {hits}/{len(coverage_rows)} "
            f"({100.0 * hits / len(coverage_rows):.1f}%) of questions have their object marked"
        )
        misses = [r for r in coverage_rows if not r[0]]
        for _, dataset, stem, question, wanted in misses[:15]:
            print(f"   MISS {dataset}/{stem}: {question[:58]!r} wanted={wanted}")

    hard_failures = len(overlaps) + len(missing_meta)
    print(f"\nRESULT: {'FAIL' if hard_failures else 'PASS'} ({hard_failures} hard failures)")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
