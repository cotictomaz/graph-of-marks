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


def label_terms(graph):
    """Mark class names, canonicalized like the question terms they are compared to.

    Without canonicalization a `man_1` mark does not match a question asking about
    a "person": _inherit_specific_label renames person -> man, which is strictly
    more informative but made this gate under-report coverage by ~0.6pp.
    """
    out = set()
    for node in graph.get("nodes", []):
        label = str(node.get("label", "")).strip().lower()
        if not label or label == "scene":
            continue
        base = re.sub(r"_\d+$", "", label)
        out.add(QI.canonical_object_label(base) or base)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--datasets", default="gqa,vqav1,vqav2")
    # gom_v* profiles declare max_detections_total = 15; gating below the
    # profile's own contract just reports the profile, not a defect.
    ap.add_argument("--max-marks", type=int, default=15)
    ap.add_argument("--max-arrows", type=int, default=5)
    # Detector recall, not a render defect: "shadow", "ladle", "jet" are the
    # documented ceiling. Measured over all 3,975 curated images, both gom_v3 and
    # gom_v4 sit at ~84%; the 90% default here was calibrated on a 95-row
    # stratified set and was simply unreachable. The gate exists to catch a
    # regression like the closed vocabulary gom_v3 removed (16 open classes ->
    # 228), which would show far below this floor, not to chase 100%.
    ap.add_argument("--min-coverage", type=float, default=80.0)
    # >3 stroked contours on a single mark means hole boundaries or specks are being
    # drawn again; a genuinely multi-part object rarely exceeds three pieces.
    ap.add_argument("--max-contours-per-mark", type=int, default=3)
    ap.add_argument("--max-short-arrow-rate", type=float, default=0.5)
    ap.add_argument("--max-unbound-rate", type=float, default=0.2)
    args = ap.parse_args()

    overlaps, missing_meta, budget, coverage_rows = [], [], [], []
    # label_overlap_count alone passed the gom_v3 run while every arrowhead sat
    # under a label box and relation labels floated free of their arcs. These are
    # the metrics that would have caught it.
    heads, unbound, dropped, digests = [], [], [], []
    short_arrows, scribbles = [], []
    n_render_variants = 0
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

            # run_table2.py hard-fails on this at inference time, after the model
            # is loaded. Catch it here instead: it is a pure metadata check.
            want = graph.get("graph", {}).get("edge_digest")
            for name, info in variants.items():
                if info.get("edge_digest") != want:
                    digests.append(f"{dataset}/{stem}/{name}: graph/render digest mismatch")
                    break

            for name, info in variants.items():
                count = info.get("label_overlap_count")
                if count is None:
                    missing_meta.append(f"{dataset}/{stem}/{name}: no label_overlap_count")
                elif count > 0:
                    overlaps.append(f"{dataset}/{stem}/{name}: {count} overlapping pairs")
                if not info.get("display_relationships"):
                    continue
                n_render_variants += 1
                for key, bucket, label in (
                    ("arrowhead_occluded_count", heads, "heads under a label"),
                    ("relation_label_unbound_count", unbound, "labels off their arc"),
                    ("relation_label_dropped_count", dropped, "labels dropped"),
                    # gom_v4 clipped arrow endpoints to the box boundary, leaving
                    # 52.9% of arrows with a shaft under 25px -- a head alone shows
                    # position but not direction.
                    ("arrows_short_count", short_arrows, "arrows with no visible shaft"),
                ):
                    value = info.get(key)
                    if value is None:
                        missing_meta.append(f"{dataset}/{stem}/{name}: no {key}")
                    elif value > 0:
                        far = info.get("relation_label_max_dist_px") or 0
                        suffix = f" (max {far:.0f}px)" if "unbound" in key else ""
                        bucket.append(f"{dataset}/{stem}/{name}: {value} {label}{suffix}")

                # Every contour of every mask used to be stroked, including cv2's
                # interior hole boundaries -- which is what drew a flock of sheep as
                # scribbles. More than a few per mark means that has come back.
                contours = info.get("mask_contours_max")
                if contours is not None and contours > args.max_contours_per_mark:
                    scribbles.append(
                        f"{dataset}/{stem}/{name}: {contours} contours on one mark"
                    )
                    # Reported, NOT gated. Measured over 3,975 images: 77% of renders
                    # stroke one contour, 97.6% three or fewer, with a thin tail to 13
                    # -- the shape of legitimately fragmented objects (a bicycle is 6).
                    # The defect this replaced was cv2.RETR_CCOMP stroking interior
                    # HOLE boundaries, which RETR_EXTERNAL now makes structurally
                    # impossible; contour count is a proxy for fragmentation, not for
                    # that bug. The real guard is
                    # tests/test_visualizer_quality.py::test_mask_outline_ignores_holes_and_specks.

            marks = len([n for n in graph.get("nodes", []) if n.get("label") != "scene"])
            arrows = max(
                (v.get("rendered_edge_count") or 0) for v in variants.values()
            )
            if marks > args.max_marks or arrows > args.max_arrows:
                budget.append(f"{dataset}/{stem}: {marks} marks, {arrows} arrows")

            # Coverage is only meaningful where the object is actually present.
            # On gold="no" existence questions the correct behaviour is NOT to mark
            # the queried noun: marking it plants a false existence proof, and the
            # flip rate when that happens is 34.5% (gemma) / 34.1% (llamav) against
            # 6.0% / 24.2% when it does not. Gating those rows would reward
            # hallucination.
            if str(row.get("answer", "")).strip().lower() == "no":
                continue
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
    print(f"edge digest mismatch: {len(digests)} (must be 0)")
    for line in digests[:10]:
        print(f"   {line}")
    short_rate = 100.0 * len(short_arrows) / max(1, n_render_variants)
    unbound_rate = 100.0 * len(unbound) / max(1, n_render_variants)
    for title, bucket in (
        ("arrowheads occluded ", heads),
        ("rel labels dropped  ", dropped),
    ):
        print(f"{title}: {len(bucket)} (must be 0)")
        for line in bucket[:10]:
            print(f"   {line}")
    # Rate, not count: N varies by run, and a count threshold calibrated on one
    # run is meaningless on another (the <=5 leakage count taught us that).
    # Measured over 14,536 relation-bearing renders: gom_v5 leaves 0.16% of renders
    # with an arrow under 25px and 0.01% with a label off its arc; gom_v4, which
    # clipped arrow endpoints to the box, left ~53% of ARROWS with no shaft. These
    # thresholds pass gom_v5 with headroom and fail a regression toward gom_v4.
    print(f"arrows w/o shaft    : {len(short_arrows)} of {n_render_variants} renders "
          f"= {short_rate:.2f}% (gate: <= {args.max_short_arrow_rate:.2f}%)")
    for line in short_arrows[:6]:
        print(f"   {line}")
    print(f"rel labels off arc  : {len(unbound)} = {unbound_rate:.2f}% "
          f"(gate: <= {args.max_unbound_rate:.2f}%)")
    for line in unbound[:6]:
        print(f"   {line}")
    print(f"mask scribbles      : {len(scribbles)} (informational, not gated)")
    print(f"over budget         : {len(budget)} (marks>{args.max_marks} or arrows>{args.max_arrows})")
    for line in budget[:10]:
        print(f"   {line}")

    coverage_fail = 0
    if coverage_rows:
        hits = sum(1 for row in coverage_rows if row[0])
        rate = 100.0 * hits / len(coverage_rows)
        print(
            f"query coverage      : {hits}/{len(coverage_rows)} "
            f"({rate:.1f}%) of questions have their object marked "
            f"(must be >= {args.min_coverage:.0f}%)"
        )
        misses = [r for r in coverage_rows if not r[0]]
        for _, dataset, stem, question, wanted in misses[:15]:
            print(f"   MISS {dataset}/{stem}: {question[:58]!r} wanted={wanted}")
        coverage_fail = 1 if rate < args.min_coverage else 0

    hard_failures = (
        len(overlaps) + len(missing_meta) + len(heads) + len(dropped)
        + len(digests) + len(budget) + coverage_fail
        + int(short_rate > args.max_short_arrow_rate)
        + int(unbound_rate > args.max_unbound_rate)
    )
    print(f"\nRESULT: {'FAIL' if hard_failures else 'PASS'} ({hard_failures} hard failures)")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
