"""The surviving mark must keep the question's specific name.

"Who is wearing a tee shirt?" detects man@0.47 on the very mask the ensemble
already called person@0.54. Dedup keeps the higher-scoring box and drops `man`
at mask IoU 1.000, so the mark stays `person_1` -- which is what every model then
answers, scoring 0 against gold "man". That is the single largest ID-leak
mechanism in the run (61 of 79 GQA who-questions for Qwen).
"""
from gom.pipeline.preprocessor import ImageGraphPreprocessor

inherit = ImageGraphPreprocessor._inherit_specific_label


def test_survivor_inherits_the_specific_name():
    labels = ["man", "person", "tie"]
    assert inherit(labels, dropped=0, survivor=1) is True
    assert labels[1] == "man"
    assert labels[2] == "tie"


def test_never_renames_a_genuinely_different_class():
    """`van` does not canonicalize onto `bus`, so a dropped van must not rename it."""
    labels = ["van", "bus"]
    assert inherit(labels, dropped=0, survivor=1) is False
    assert labels[1] == "bus"


def test_generic_never_overwrites_a_specific_name():
    labels = ["person", "man"]
    assert inherit(labels, dropped=0, survivor=1) is False
    assert labels[1] == "man"


def test_identical_labels_are_a_no_op():
    labels = ["person", "person"]
    assert inherit(labels, dropped=0, survivor=1) is False
