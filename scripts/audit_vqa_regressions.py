#!/usr/bin/env python3
"""Find examples where a raw-image baseline beats a preprocessed VQA run."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = list(";/[]\"{}()=+\\_-><@`,?!")
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def extract_answer(value: object) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    for lead in ("answer:", "the answer is", "answer is"):
        position = lowered.find(lead)
        if position != -1:
            text = text[position + len(lead):]
            break
    return text.split("\n", 1)[0].strip().strip(" '\"`")


def normalize_answer(value: object) -> str:
    text = extract_answer(value).lower().replace("\n", " ").replace("\t", " ").strip()
    text = _COMMA_STRIP.sub(r"\1\3", text)
    for punctuation in _PUNCT:
        text = text.replace(punctuation, "" if f"{punctuation} " in text else " ")
    text = _PERIOD_STRIP.sub("", text)
    words = [
        _NUMBER_MAP.get(word, word)
        for word in text.split()
        if word not in _ARTICLES
    ]
    return " ".join(words)


def vqa_score(prediction: object, answers: Iterable[object]) -> float:
    pred = normalize_answer(prediction)
    gold = [normalize_answer(answer) for answer in answers]
    matches = sum(answer == pred for answer in gold)
    if matches == 0:
        pred_tokens = pred.split()
        matches = sum(
            bool(tokens) and any(
                pred_tokens[i:i + len(tokens)] == tokens
                for i in range(len(pred_tokens) - len(tokens) + 1)
            )
            for tokens in (answer.split() for answer in gold)
        )
    return min(1.0, matches / 3.0)


def question_type(question: str) -> str:
    q = question.lower().strip()
    if "how many" in q:
        return "count"
    if re.search(r"\bwhat colou?r\b", q):
        return "color"
    if re.match(r"^(is|are|do|does|did|can|could|has|have)\b", q):
        return "yes_no"
    if re.search(r"\b(where|left|right|above|below|next to|behind|front)\b", q):
        return "spatial"
    return "identity"


def key(row: dict) -> Tuple[str, str]:
    return str(row.get("image_path", "")), str(row.get("question", ""))


def load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def index_predictions(rows: Iterable[dict]) -> Dict[Tuple[str, str], List[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def audit(annotations: List[dict], baseline: List[dict], variants: Dict[str, List[dict]]) -> dict:
    baseline_index = index_predictions(baseline)
    variant_indices = {name: index_predictions(rows) for name, rows in variants.items()}
    regressions = []
    totals = Counter()
    by_type = Counter()

    for annotation in annotations:
        sample_key = key(annotation)
        occurrence = totals[("__occurrence__", sample_key)]
        totals[("__occurrence__", sample_key)] += 1
        baseline_rows = baseline_index.get(sample_key, [])
        if occurrence >= len(baseline_rows):
            continue
        answers = annotation.get("answers", [])
        baseline_pred = baseline_rows[occurrence].get("generated_answer")
        baseline_score = vqa_score(baseline_pred, answers)
        for variant_name, variant_index in variant_indices.items():
            variant_rows = variant_index.get(sample_key, [])
            if occurrence >= len(variant_rows):
                continue
            variant_pred = variant_rows[occurrence].get("generated_answer")
            variant_score = vqa_score(variant_pred, answers)
            totals[(variant_name, "examples")] += 1
            totals[(variant_name, "baseline_score")] += baseline_score
            totals[(variant_name, "variant_score")] += variant_score
            if baseline_score > variant_score:
                qtype = question_type(sample_key[1])
                by_type[(variant_name, qtype)] += 1
                regressions.append({
                    "variant": variant_name,
                    "image_path": sample_key[0],
                    "question": sample_key[1],
                    "question_type": qtype,
                    "answers": answers,
                    "baseline_answer": baseline_pred,
                    "baseline_score": baseline_score,
                    "variant_answer": variant_pred,
                    "variant_score": variant_score,
                })

    summary = {}
    for name in variants:
        count = totals[(name, "examples")]
        summary[name] = {
            "examples": count,
            "baseline_accuracy": totals[(name, "baseline_score")] / count if count else 0.0,
            "variant_accuracy": totals[(name, "variant_score")] / count if count else 0.0,
            "strict_regressions": sum(1 for row in regressions if row["variant"] == name),
            "regressions_by_question_type": {
                qtype: by_type[(name, qtype)]
                for qtype in ("color", "count", "identity", "spatial", "yes_no")
                if by_type[(name, qtype)]
            },
        }
    return {"summary": summary, "regressions": regressions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="May be repeated",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    variants = {}
    for spec in args.variant:
        name, separator, path = spec.partition("=")
        if not separator or not name:
            parser.error("--variant must be NAME=PATH")
        variants[name] = load_rows(Path(path))

    report = audit(load_rows(args.annotations), load_rows(args.baseline), variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
