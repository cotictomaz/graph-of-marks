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


def load_features(root, profile, ds):
    """Model-independent a-priori features + per-model scores for each kept instance."""
    from question_filter import appearance_reason

    rows = first_row_per_image(read_jsonl(root / "prepared" / ds / "eval.jsonl"))
    gdir = root / "artifacts" / ds / "preprocessing"
    preds = {
        m: {
            c: {
                r["question_id"]: r["prediction"]
                for r in read_jsonl(
                    root / "predictions" / profile / m / ds / c / "seed0_temp0.2_top_p0.90.jsonl"
                )
            }
            for c in ("raw", "segmented")
        }
        for m in MODELS
    }
    out = []
    for row in rows:
        answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
        if appearance_reason(row["question"], answers) is not None:
            continue
        q = row["question_id"]
        intent = QI.parse_question_intent(row["question"])
        gpath = gdir / f"{Path(row['image_path']).stem}_q1_graph.json"
        graph = json.loads(gpath.read_text()) if gpath.is_file() else {"nodes": [], "links": []}
        labels = [QI.canonical_object_label(_SUFFIX_RE.sub("", n["label"])) for n in graph.get("nodes", [])]
        areas = [n.get("area_norm", 0.0) for n in graph.get("nodes", [])]
        multiplicity = max(
            (labels.count(t) for t in intent.object_terms), default=0
        )
        feats = {
            "qtype": intent.question_type,
            "relational": bool(intent.relation_terms) or intent.question_type == "spatial",
            "qlen": len(row["question"].split()),
            "n_nodes": len(labels),
            "n_edges": len(graph.get("links", [])),
            "multiplicity": multiplicity,
            "covered": bool(intent.object_terms) and all(t in labels for t in intent.object_terms),
            "tiny_objects": bool(areas) and min(areas) < 0.02,
        }
        scores = {
            m: {c: score(preds[m][c][q], row, ds) for c in ("raw", "segmented")}
            for m in MODELS
        }
        out.append({"qid": q, "image": Path(row["image_path"]).stem,
                    "question": row["question"], "feats": feats, "scores": scores})
    return out


BUCKETS = {
    "qtype": [("qtype=" + v, lambda f, v=v: f["qtype"] == v)
              for v in ("count", "yes_no", "spatial", "identity", "open")],
    "relational": [("relational", lambda f: f["relational"]),
                   ("non-relational", lambda f: not f["relational"])],
    "qlen": [("qlen<=5", lambda f: f["qlen"] <= 5),
             ("qlen6-9", lambda f: 6 <= f["qlen"] <= 9),
             ("qlen>=10", lambda f: f["qlen"] >= 10)],
    "n_nodes": [("nodes<=3", lambda f: f["n_nodes"] <= 3),
                ("nodes4-6", lambda f: 4 <= f["n_nodes"] <= 6),
                ("nodes>=7", lambda f: f["n_nodes"] >= 7)],
    "n_edges": [("edges<=6", lambda f: f["n_edges"] <= 6),
                ("edges>=7", lambda f: f["n_edges"] >= 7)],
    "multiplicity": [("mult=0", lambda f: f["multiplicity"] == 0),
                     ("mult=1", lambda f: f["multiplicity"] == 1),
                     ("mult>=2", lambda f: f["multiplicity"] >= 2)],
    "covered": [("covered", lambda f: f["covered"]),
                ("uncovered", lambda f: not f["covered"])],
    "tiny": [("tiny-objects", lambda f: f["tiny_objects"]),
             ("no-tiny-objects", lambda f: not f["tiny_objects"])],
}


def rule_deltas(instances, pred):
    sub = [i for i in instances if pred(i["feats"])]
    if len(sub) < 50:
        return None
    out = {"n": len(sub)}
    for m in MODELS:
        raw = sum(i["scores"][m]["raw"] for i in sub) / len(sub)
        seg = sum(i["scores"][m]["segmented"] for i in sub) / len(sub)
        out[m] = 100.0 * (seg - raw)
    return out


def rule_search(args) -> None:
    data = {ds: load_features(args.data_root, args.prompt_profile, ds) for ds in DATASETS}

    rules = []
    singles = [(name, fn, group) for group, items in BUCKETS.items() for name, fn in items]
    for name, fn, _ in singles:
        rules.append((name, fn))
    for i, (n1, f1, g1) in enumerate(singles):
        for n2, f2, g2 in singles[i + 1:]:
            if g1 == g2:
                continue
            rules.append((f"{n1} & {n2}", lambda f, a=f1, b=f2: a(f) and b(f)))

    print(f"## Rule search: derive on VQAv1 (n_kept={len(data['vqav1'])}), "
          f"verify on GQA+VQAv2 (kept {len(data['gqa'])}/{len(data['vqav2'])})")
    print(f"rules enumerated: {len(rules)}; criterion: Delta(segmented-raw)>0 for ALL "
          f"three models, n>=50, on train AND both held-out datasets\n")
    candidates = []
    for name, fn in rules:
        train = rule_deltas(data["vqav1"], fn)
        if train is None or any(train[m] <= 0 for m in MODELS):
            continue
        candidates.append((name, fn, train))
    print(f"| rule | n(v1) | " + " | ".join(m.split('_')[0] for m in MODELS) +
          " | GQA transfer | VQAv2 transfer | VERDICT |")
    print("|---|---:|" + "---:|" * 3 + "---|---|---|")
    verified = []
    for name, fn, train in sorted(candidates, key=lambda x: -min(x[2][m] for m in MODELS)):
        cols = "".join(f" {train[m]:+.2f} |" for m in MODELS)
        transfers = {}
        for ds in ("gqa", "vqav2"):
            d = rule_deltas(data[ds], fn)
            if d is None:
                transfers[ds] = "n<50"
            else:
                ok = all(d[m] > 0 for m in MODELS)
                transfers[ds] = ("PASS " if ok else "fail ") + \
                    "/".join(f"{d[m]:+.1f}" for m in MODELS) + f" (n={d['n']})"
        ok_all = all(t.startswith("PASS") for t in transfers.values())
        if ok_all:
            verified.append(name)
        print(f"| {name} | {train['n']} |{cols} {transfers['gqa']} | {transfers['vqav2']} | "
              f"{'**VERIFIED**' if ok_all else 'not verified'} |")
    if not candidates:
        print("| (no rule met the train criterion for all three models) | | | | | | |")
    print(f"\ntrain-passing candidates: {len(candidates)}; VERIFIED on both held-out "
          f"datasets: {len(verified)} {verified}")

    print("\n## Outcome-selected showcase (DIAGNOSTIC ONLY — selected on results, "
          "verifies nothing)")
    for m in MODELS:
        rescues = [i for ds in DATASETS for i in data[ds]
                   if i["scores"][m]["raw"] <= 0.1 and i["scores"][m]["segmented"] >= 0.9]
        path = args.data_root / f"showcase_rescues.{m}.json"
        path.write_text(json.dumps(
            [{"image": i["image"], "question": i["question"]} for i in rescues],
            indent=2, ensure_ascii=False))
        total = sum(len(data[ds]) for ds in DATASETS)
        print(f"- {m}: {len(rescues)}/{total} kept instances are rescues -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("reproduction/data_v3"))
    ap.add_argument("--prompt-profile", default="direct_concise")
    ap.add_argument("--rule-search", action="store_true")
    args = ap.parse_args()
    if args.rule_search:
        rule_search(args)
        return 0

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
