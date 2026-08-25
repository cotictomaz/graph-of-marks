#!/usr/bin/env python3
"""Dump a raw-right/GoM-wrong flip gallery: markdown + side-by-side images.

Every earlier gallery (FLIP_EXAMPLES*.md) was assembled by hand, which is why
they drifted from the runs they described. This does the selection, the image
copy and the markdown in one pass, so a gallery is always reproducible from a
data root.

    python3 reproduction/make_flip_gallery.py \
        --data-root reproduction/data_smoke --prompt-profile gom_v4_concise \
        --model qwen25_vl_7b --condition gom_text_labeled \
        --output reproduction/FLIP_EXAMPLES_GOM_V4.md

Stdlib only; runs on the host.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_paper"))
from common import first_row_per_image  # noqa: E402
from vqa_metrics import gqa_hit, normalize, vqa_soft_acc_phrase  # noqa: E402
from check_leakage import GENERIC_TAG_RE, PLAN_RE, REL_RE  # noqa: E402

PERSON_SUBTYPES = {"man", "woman", "boy", "girl", "child", "guy", "lady", "person"}


def classify(row: dict, raw: str, marked: str, marks: list) -> str:
    """Best-guess mechanism, in the vocabulary of the measured failure taxonomy.

    A label, not a verdict: the point of the gallery is for a human to confirm or
    refute it against the render.
    """
    value = marked.strip()
    norm = normalize(value)
    first = norm.split()[0] if norm.split() else ""
    bases = {m.rsplit("_", 1)[0].lower() for m in marks}
    gold = normalize(str(row.get("answer", "")))

    if GENERIC_TAG_RE.search(value):
        return "generic_tag_answer"
    if PLAN_RE.match(value):
        return "plan_mode"
    if REL_RE.match(value):
        return "relation_word_answer"
    if first in (bases & PERSON_SUBTYPES) and first != gold:
        return "wrong_subtype_copied"
    if first in bases and first != gold:
        return "mark_label_copied"
    if gold == "no" and norm.startswith("yes"):
        return "false_premise_asserted"
    if gold == "yes" and norm.startswith("no"):
        return "absence_denied"
    if len(norm.split()) > 6:
        return "verbose"
    if not bases:
        return "no_marks"
    return "other"

VARIANT = {
    "gom_text": "gom_text_unlabeled",
    "gom_numeric": "gom_numeric_unlabeled",
    "gom_text_labeled": "gom_text_labeled",
    "gom_numeric_labeled": "gom_numeric_labeled",
    "segmented": "segmented",
    "som_numeric": "som_numeric",
}


def score(prediction: str, row: dict) -> float:
    if row["dataset"] == "gqa":
        return gqa_hit(prediction, row["answer"])
    return 1.0 if vqa_soft_acc_phrase(prediction, row["answers"]) >= 0.6 else 0.0


def load(path: Path) -> dict:
    return {
        json.loads(line)["question_id"]: json.loads(line)["prediction"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--prompt-profile", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="gom_text_labeled")
    ap.add_argument("--datasets", default="gqa,vqav1,vqav2")
    ap.add_argument("--setting", default="seed0_temp0.2_top_p0.90.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--stratify", action="store_true",
                    help="spread the selection across mechanisms instead of file order")
    ap.add_argument("--mechanism", default=None,
                    help="keep only flips classified as this mechanism")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    assets = args.output.parent / (args.output.stem.lower() + "_images")
    assets.mkdir(parents=True, exist_ok=True)
    variant = VARIANT.get(args.condition, args.condition)

    flips, totals = [], {"rows": 0, "flips": 0, "rescues": 0}
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        prepared = args.data_root / "prepared" / dataset / "eval.jsonl"
        if not prepared.is_file():
            continue
        rows = first_row_per_image(
            [json.loads(line) for line in prepared.read_text().splitlines() if line.strip()]
        )
        base = args.data_root / "predictions" / args.prompt_profile / args.model / dataset
        raw_path, marked_path = base / "raw" / args.setting, base / args.condition / args.setting
        if not (raw_path.is_file() and marked_path.is_file()):
            continue
        raw, marked = load(raw_path), load(marked_path)
        for row in rows:
            qid = row["question_id"]
            if qid not in raw or qid not in marked:
                continue
            totals["rows"] += 1
            before, after = score(raw[qid], row), score(marked[qid], row)
            if before == 1.0 and after == 0.0:
                totals["flips"] += 1
                stem = Path(row["image_path"]).stem
                graph_path = (
                    args.data_root / "artifacts" / dataset / "preprocessing"
                    / f"{stem}_q1_graph.json"
                )
                marks = [
                    n["label"]
                    for n in json.loads(graph_path.read_text(encoding="utf-8")).get("nodes", [])
                    if n.get("label") != "scene"
                ] if graph_path.is_file() else []
                flips.append((dataset, row, raw[qid], marked[qid],
                              classify(row, raw[qid], marked[qid], marks)))
            elif before == 0.0 and after == 1.0:
                totals["rescues"] += 1

    from collections import Counter, defaultdict

    census = Counter(f[4] for f in flips)
    if args.mechanism:
        flips = [f for f in flips if f[4] == args.mechanism]
    if args.stratify:
        by_mech = defaultdict(list)
        for flip in flips:
            by_mech[flip[4]].append(flip)
        ordered = []
        while len(ordered) < args.limit and any(by_mech.values()):
            for mech in sorted(by_mech, key=lambda m: -census[m]):
                if by_mech[mech]:
                    ordered.append(by_mech[mech].pop(0))
                    if len(ordered) >= args.limit:
                        break
        flips = ordered

    lines = [
        f"# GoM flips: `{args.model}` / `{args.condition}` / `{args.prompt_profile}`",
        "",
        f"Generated by `reproduction/make_flip_gallery.py` from `{args.data_root}`. "
        f"Every case is correct on the clean image and wrong on the GoM render "
        f"(lenient scoring). {totals['flips']} flips and {totals['rescues']} rescues "
        f"over {totals['rows']} rows; the {min(len(flips), args.limit)} below are "
        + ("spread across mechanisms (most common first), not cherry-picked."
           if args.stratify else "the first by dataset order, not a curated selection."),
        "",
    ]
    lines += [
        "**Mechanism census over all "
        + str(totals["flips"])
        + " flips** (heuristic labels, to be confirmed against each render):",
        "",
        "| mechanism | count |",
        "|---|---:|",
        *[f"| `{mech}` | {count} |" for mech, count in census.most_common()],
        "",
    ]

    for index, (dataset, row, raw_value, marked_value, mech) in enumerate(flips[: args.limit], 1):
        stem = Path(row["image_path"]).stem
        original = assets / f"{index:02d}_{stem}_original.jpg"
        rendered = assets / f"{index:02d}_{stem}_gom.jpg"
        shutil.copyfile(row["image_path"], original)
        source = (
            args.data_root / "artifacts" / dataset / "preprocessing" / "renders"
            / variant / f"{stem}_q1_output.jpg"
        )
        shutil.copyfile(source, rendered)
        graph_path = (
            args.data_root / "artifacts" / dataset / "preprocessing" / f"{stem}_q1_graph.json"
        )
        graph = json.loads(graph_path.read_text())
        marks = [n["label"] for n in graph.get("nodes", []) if n.get("label") != "scene"]
        triples = (
            args.data_root / "artifacts" / dataset / "preprocessing"
            / f"{stem}_q1_graph_triples.txt"
        )
        lines += [
            f"## {index}. `{stem}` ({dataset}) — `{mech}`",
            "",
            f"**Question:** {row['question']}",
            f"**Gold:** {row['answer']}",
            "",
            "| condition | model output |",
            "|---|---|",
            f"| raw (clean image) | **{raw_value.strip()[:80]}** correct |",
            f"| GoM (`{args.condition}`) | **{marked_value.strip()[:80]}** wrong |",
            "",
            f"*Marks ({len(marks)}):* {', '.join(marks) or 'none'}",
            "",
            "```",
            triples.read_text().strip() if triples.is_file() else "(no triples)",
            "```",
            "",
            "| original | GoM render (what the model saw) |",
            "|---|---|",
            f"| ![o]({assets.name}/{original.name}) | ![g]({assets.name}/{rendered.name}) |",
            "",
        ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{totals['flips']} flips / {totals['rescues']} rescues over {totals['rows']} rows "
        f"-> {args.output} ({min(len(flips), args.limit)} cases, images in {assets})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
