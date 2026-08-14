#!/usr/bin/env python3
"""Score configured decoding runs and emit a Table 2-style report."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PAPER_SPEC, first_row_per_image, load_yaml, sha256_file
from question_filter import keep_ids


CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "i'dve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PUNCTUATION = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize(value: str) -> str:
    text = (value or "").replace("\n", " ").replace("\t", " ").strip()
    punctuated = text
    for punctuation in PUNCTUATION:
        if (
            punctuation + " " in text
            or " " + punctuation in text
            or COMMA_STRIP.search(text) is not None
        ):
            punctuated = punctuated.replace(punctuation, "")
        else:
            punctuated = punctuated.replace(punctuation, " ")
    punctuated = PERIOD_STRIP.sub("", punctuated)
    words = []
    for word in punctuated.lower().split():
        word = NUMBER_WORDS.get(word, word)
        if word in ARTICLES:
            continue
        words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def official_vqa_score(prediction: str, answers: list[str]) -> float:
    # Normalize every response, including unanimous-answer rows. Model chat
    # outputs commonly capitalize a short answer ("Yes") while VQA annotations
    # are lowercase ("yes"); skipping normalization for unanimous annotations
    # incorrectly turns those otherwise exact predictions into misses.
    prediction = normalize(prediction)
    golds = [normalize(str(answer)) for answer in answers]
    values = []
    for held_out in range(len(golds)):
        matches = sum(
            prediction == answer
            for index, answer in enumerate(golds)
            if index != held_out
        )
        values.append(min(1.0, matches / 3.0))
    return sum(values) / len(values) if values else 0.0


def released_code_vqa_score(prediction: str, answer: str) -> float:
    """Mirror the paper-era qa_generation.py VQA/GQA comparison exactly."""
    return float((prediction or "").strip().lower() == str(answer).strip().lower())


def iou_xyxy(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def predicted_regions(prediction: str, graph_path: Path, condition: str) -> list[list[float]]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in graph.get("nodes", [])
        if node.get("label") != "scene" and node.get("bbox_norm") is not None
    ]
    normalized_prediction = prediction.lower()
    selected = []
    if "numeric" in condition:
        numbers = {int(value) for value in re.findall(r"\b\d+\b", prediction)}
        selected = [node for index, node in enumerate(nodes, start=1) if index in numbers]
    else:
        selected = [
            node
            for node in nodes
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(str(node['label']).lower())}(?![a-z0-9_])",
                normalized_prediction,
            )
        ]
    return [node["bbox_norm"] for node in selected]


def maximum_iou_matches(
    predictions: list[list[float]], truths: list[list[float]], threshold: float
) -> int:
    adjacency = [
        [index for index, truth in enumerate(truths) if iou_xyxy(prediction, truth) >= threshold]
        for prediction in predictions
    ]
    truth_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or augment(previous, visited):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(predictions)))


def score_row(row: dict, prediction: str, dataset: str, condition: str, graph_path: Path) -> float:
    if dataset in {"vqav1", "vqav2"}:
        return official_vqa_score(prediction, row["answers"])
    if dataset == "gqa":
        return float(normalize(prediction) == normalize(row["answer"]))
    width, height = Image.open(row["image_path"]).size
    targets = row.get("targets")
    boxes = (
        [target["bbox_xywh"] for target in targets]
        if targets is not None
        else [row["bbox_xywh"]]
    )
    truths = []
    for box in boxes:
        x, y, w, h = [float(value) for value in box]
        truths.append([x / width, y / height, (x + w) / width, (y + h) / height])
    regions = predicted_regions(prediction, graph_path, condition)
    return maximum_iou_matches(regions, truths, 0.9) / len(truths)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def graph_paths(
    rows: list[dict], preprocessing: Path, granularity: str = "image"
) -> dict[str, Path]:
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    result = {}
    for row in rows:
        stem = Path(row["image_path"]).stem
        if granularity == "image":
            index = 1
        else:
            counts[stem] += 1
            index = counts[stem]
        result[row["question_id"]] = preprocessing / f"{stem}_q{index}_graph.json"
    return result


def prediction_path(
    data_root: Path,
    model: str,
    dataset: str,
    condition: str,
    seed: int,
    temperature: float,
    top_p: float,
    prompt_profile: str,
) -> Path:
    return (
        data_root / "predictions" / prompt_profile / model / dataset / condition
        / f"seed{seed}_temp{temperature:.1f}_top_p{top_p:.2f}.jsonl"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-granularity", choices=("image", "question"), default="image"
    )
    parser.add_argument("--one-per-image", action="store_true")
    parser.add_argument("--single-setting", action="store_true")
    parser.add_argument("--prompt-profile", default="paper_declared")
    parser.add_argument(
        "--question-filter", choices=("none", "appearance"), default="none"
    )
    args = parser.parse_args()
    spec = load_yaml(PAPER_SPEC)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    decode = spec["published_decoding"]
    settings = (
        [(0, 0.2, 0.9)]
        if args.single_setting
        else list(itertools.product(decode["seeds"], decode["temperatures"], decode["top_p"]))
    )

    report_rows = []
    for model in models:
        for dataset in datasets:
            eval_path = args.data_root / "prepared" / dataset / "eval.jsonl"
            rows = read_jsonl(eval_path)
            if args.one_per_image:
                rows = first_row_per_image(rows)
            by_id = {row["question_id"]: row for row in rows}
            preprocessing = args.data_root / "artifacts" / dataset / "preprocessing"
            graphs = graph_paths(rows, preprocessing, args.artifact_granularity)
            keep = None
            if args.question_filter == "appearance" and dataset != "refcocog":
                keep = keep_ids(rows)
            for condition in spec["conditions"]:
                if dataset == "refcocog" and condition in {"raw", "segmented"}:
                    continue
                run_scores = []
                released_code_run_scores = []
                for seed, temperature, top_p in settings:
                    path = prediction_path(
                        args.data_root, model, dataset, condition,
                        seed, temperature, top_p, args.prompt_profile,
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    predictions = read_jsonl(path)
                    if len(predictions) != len(rows) or {
                        p["question_id"] for p in predictions
                    } != set(by_id):
                        raise RuntimeError(f"{path}: incomplete prediction file")
                    if keep is not None:
                        predictions = [
                            p for p in predictions if p["question_id"] in keep
                        ]
                    scores = []
                    released_code_scores = []
                    for prediction in predictions:
                        row = by_id[prediction["question_id"]]
                        scores.append(
                            score_row(
                                row,
                                prediction["prediction"],
                                dataset,
                                condition,
                                graphs[row["question_id"]],
                            )
                        )
                        if dataset in {"vqav1", "vqav2"}:
                            released_code_scores.append(
                                released_code_vqa_score(
                                    prediction["prediction"], row["answer"]
                                )
                            )
                    run_scores.append(100.0 * sum(scores) / len(scores))
                    if released_code_scores:
                        released_code_run_scores.append(
                            100.0
                            * sum(released_code_scores)
                            / len(released_code_scores)
                        )
                result = {
                    "model": model,
                    "dataset": dataset,
                    "condition": condition,
                    "prompt_profile": args.prompt_profile,
                    "question_filter": args.question_filter,
                    "n_scored": len(keep) if keep is not None else len(rows),
                    "n_total": len(rows),
                    "runs": len(settings),
                    "mean_accuracy_points": statistics.mean(run_scores),
                    "std_accuracy_points": statistics.pstdev(run_scores),
                    "min_accuracy_points": min(run_scores),
                    "max_accuracy_points": max(run_scores),
                }
                if released_code_run_scores:
                    result["released_code_mean_accuracy_points"] = statistics.mean(
                        released_code_run_scores
                    )
                    result["released_code_std_accuracy_points"] = statistics.pstdev(
                        released_code_run_scores
                    )
                report_rows.append(result)
    dataset_provenance = {}
    for dataset in datasets:
        path = args.data_root / "prepared" / dataset / "provenance.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset_provenance[dataset] = json.loads(path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "paper_spec_sha256": sha256_file(PAPER_SPEC),
        "scoring": {
            "vqav1_vqav2": "official leave-one-annotator-out consensus",
            "vqav1_vqav2_released_code_compatibility": (
                "lowercase exact match against the single stored answer"
            ),
            "gqa": "normalized exact match",
            "refcocog": (
                "target-instance accuracy using maximum one-to-one matching of "
                "predicted rendered regions at IoU >= 0.9"
            ),
            "standard_deviation": "population standard deviation over 27 runs",
        },
        "dataset_provenance": dataset_provenance,
        "rows": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    lines = [
        "| Model | Dataset | Condition | N | Primary accuracy | Released-code VQA compatibility |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report_rows:
        compatibility = (
            f"{row['released_code_mean_accuracy_points']:.2f} +/- "
            f"{row['released_code_std_accuracy_points']:.2f}"
            if "released_code_mean_accuracy_points" in row
            else "n/a"
        )
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['condition']} | "
            f"{row['n_scored']}/{row['n_total']} | "
            f"{row['mean_accuracy_points']:.2f} +/- {row['std_accuracy_points']:.2f} | "
            f"{compatibility} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
