import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_vqa_regressions.py"
SPEC = importlib.util.spec_from_file_location("audit_vqa_regressions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_finds_raw_correct_variant_wrong_case():
    annotations = [{
        "image_path": "image.jpg",
        "question": "What animal is next to the dog?",
        "answers": ["giraffe"] * 10,
    }]
    baseline = [{
        "image_path": "image.jpg",
        "question": annotations[0]["question"],
        "generated_answer": "giraffe",
    }]
    variant = [{
        "image_path": "image.jpg",
        "question": annotations[0]["question"],
        "generated_answer": "dog",
    }]

    report = MODULE.audit(annotations, baseline, {"gom": variant})
    assert report["summary"]["gom"]["strict_regressions"] == 1
    assert report["regressions"][0]["baseline_score"] == 1.0
    assert report["regressions"][0]["variant_score"] == 0.0
