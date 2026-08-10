import json

import pytest
from PIL import Image

from data_paper.prepare_paper_vqav2 import (
    load_paper_rows,
    select_screen_rows,
)


def test_paper_rows_are_validated_deduplicated_and_stable(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (2, 2)).save(image_dir / name)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"image_path": "a.jpg", "question": "Q1?", "answers": ["yes"] * 10},
                {"image_path": "a.jpg", "question": "Q1?", "answers": ["yes"] * 10},
                {"image_path": "b.jpg", "question": "Q2?", "answers": ["no"] * 10},
            ]
        ),
        encoding="utf-8",
    )

    rows, duplicates = load_paper_rows(manifest, image_dir)

    assert duplicates == 1
    assert len(rows) == 2
    assert rows[0]["question_id"].startswith("paper_vqav2_")
    assert rows[0]["answer"] == "yes"
    assert load_paper_rows(manifest, image_dir)[0] == rows


def test_paper_rows_require_ten_answers(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "a.jpg")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"image_path": "a.jpg", "question": "Q?", "answers": ["yes"]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 10"):
        load_paper_rows(manifest, image_dir)


def test_screen_selection_is_image_level_and_deterministic():
    rows = [
        {"image_id": f"image_{image}", "question_id": f"{image}_{question}"}
        for image in range(5)
        for question in range(image + 1)
    ]

    first = select_screen_rows(rows, image_count=2, seed=42)
    second = select_screen_rows(rows, image_count=2, seed=42)

    assert first == second
    selected_ids = {row["image_id"] for row in first}
    assert len(selected_ids) == 2
    assert all(row in first for row in rows if row["image_id"] in selected_ids)
