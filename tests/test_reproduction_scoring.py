from reproduction.question_filter import appearance_reason, keep_ids
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


def test_appearance_filter_drops_each_appearance_category():
    assert appearance_reason("What color is the car?", []) == "color_question"
    assert appearance_reason("What is the table made of?", []) == "material_texture_pattern"
    assert appearance_reason("What does the sign say?", []) == "text_in_image"
    assert appearance_reason("Is the red car parked?", []) == "color_word_in_question"


def test_appearance_filter_drops_color_majority_gold_answers():
    answers = ["blue", "blue cup", "cup", "mug"]
    assert appearance_reason("What is on the table?", answers) == "color_word_in_answers"
    assert appearance_reason("What is on the table?", ["cup", "cup", "mug", "blue"]) is None


def test_appearance_filter_keeps_neutral_questions():
    assert appearance_reason("How many people are there?", ["2"] * 10) is None
    assert appearance_reason("Is the man wearing a hat?", ["yes"] * 10) is None
    assert appearance_reason("Where is the dog?", ["on the left"] * 10) is None


def test_keep_ids_returns_surviving_question_ids():
    rows = [
        {"question_id": "a", "question": "What color is it?", "answers": []},
        {"question_id": "b", "question": "How many dogs?", "answers": ["2"]},
        {"question_id": "c", "question": "Is there a horse?", "answer": "yes"},
    ]
    assert keep_ids(rows) == {"b", "c"}


def test_subjective_questions_are_filtered():
    assert appearance_reason("Have you visited this zoo?", ["yes"]) == "subjective"
    assert appearance_reason("Does the dog like what he's eating?", ["yes"]) == "subjective"
    assert appearance_reason("Were these taken on the same day?", ["yes"]) == "subjective"
    assert appearance_reason("How many potted plants do you see?", ["3"]) is None
