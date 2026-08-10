import json
from pathlib import Path

from reproduction.common import canonical_rows


def test_grouped_refcocog_targets_are_preserved(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"not decoded during canonicalization")
    manifest = tmp_path / "ref.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "image_path": "sample.jpg",
                    "targets": [
                        {"description": "left chair", "bbox_xywh": [1, 2, 3, 4]},
                        {"description": "right chair", "bbox_xywh": [5, 6, 7, 8]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    rows, duplicates = canonical_rows("refcocog", manifest, image_dir)

    assert duplicates == 0
    assert rows[0]["descriptions"] == ["left chair", "right chair"]
    assert [target["bbox_xywh"] for target in rows[0]["targets"]] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
