import json
from pathlib import Path

import pytest

from data_paper.run_vqa_full_finalists import (
    seed_predictions,
    validate_screen_metadata,
)
from data_paper.run_vqa_decode_grid import decoding_settings


def test_seed_predictions_matches_stable_question_ids(tmp_path: Path):
    source = tmp_path / "screen.json"
    source.write_text(
        json.dumps(
            [
                {"question_id": "q3", "pred_visual": "three"},
                {"question_id": "q1", "pred_visual": "one"},
            ]
        ),
        encoding="utf-8",
    )
    rows = [{"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"}]

    assert seed_predictions(rows, source, "pred_visual") == 2
    assert rows == [
        {"question_id": "q1", "pred_visual": "one"},
        {"question_id": "q2"},
        {"question_id": "q3", "pred_visual": "three"},
    ]


def test_validate_screen_metadata_rejects_prompt_mismatch(tmp_path: Path):
    source = tmp_path / "screen.json"
    source.write_text("[]", encoding="utf-8")
    source.with_suffix(".json.meta.json").write_text(
        json.dumps(
            {
                "model": "model",
                "mode": "visual",
                "prompt_profile": "neutral_concise",
                "temperature": 0.0,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prompt_profile"):
        validate_screen_metadata(
            source,
            model="model",
            mode="visual",
            profile="visual_aid_concise",
        )


def test_decode_grid_starts_with_released_paper_setting():
    settings = decoding_settings()

    assert settings[0] == ("paper_original", 0, 0.2, 0.9)
    assert len(settings) == 28
    assert all(row[0] == "robustness" for row in settings[1:])
