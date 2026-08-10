from reproduction.score_table2 import (
    maximum_iou_matches,
    normalize,
    official_vqa_score,
    released_code_vqa_score,
)


def test_vqa_normalization_matches_official_punctuation_digit_article_rules():
    assert normalize("The two, cats!") == "2 cats"
    assert normalize("dont") == "don't"
    assert normalize("3.14") == "3.14"


def test_vqa_consensus_uses_leave_one_annotator_out_credit():
    answers = ["yes", "yes", "yes"] + ["no"] * 7
    assert official_vqa_score("yes", answers) == 0.9


def test_vqa_unanimous_answer_scores_exactly():
    assert official_vqa_score("blue", ["blue"] * 10) == 1.0


def test_vqa_unanimous_answer_is_still_normalized():
    assert official_vqa_score("Yes.", ["yes"] * 10) == 1.0


def test_released_code_vqa_compatibility_is_only_lowercase_exact():
    assert released_code_vqa_score("Yes", "yes") == 1.0
    assert released_code_vqa_score("Yes.", "yes") == 0.0


def test_rec_matching_is_one_to_one():
    predictions = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    truths = [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.75, 0.75]]
    assert maximum_iou_matches(predictions, truths, 0.9) == 1


def test_rec_matching_can_reassign_an_earlier_prediction():
    predictions = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.9, 1.0]]
    truths = [[0.0, 0.0, 0.9, 1.0], [0.0, 0.0, 1.0, 1.0]]
    assert maximum_iou_matches(predictions, truths, 0.89) == 2
