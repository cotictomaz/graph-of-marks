from gom.question_intent import canonical_object_label, parse_question_intent


def test_animal_relation_question_is_controlled_and_targeted():
    intent = parse_question_intent("What animal is laying next to the dog?")

    assert intent.question_type == "identity"
    assert intent.anchor_terms == {"dog"}
    assert "stuffed giraffe" in intent.relation_source_terms
    assert intent.relation_anchor_terms == {"dog"}
    assert intent.target_categories == {"animal"}
    assert intent.relation_terms == {"next_to"}
    assert "stuffed giraffe" in intent.detector_queries
    assert "stuffed turtle" in intent.detector_queries
    assert "pig" in intent.detector_queries
    assert "inside" not in intent.relation_terms
    assert "hotdog" not in intent.object_terms
    assert len(intent.detector_queries) < 30


def test_visual_plural_aliases_are_canonical():
    assert canonical_object_label("people") == "person"
    assert canonical_object_label("monitors") == "tv"
    intent = parse_question_intent("How many people are in the bus?")
    assert "person" in intent.anchor_terms
    assert "bus" in intent.anchor_terms


def test_relation_anchor_is_the_object_after_the_relation_phrase():
    intent = parse_question_intent(
        "How many people are standing in front of the doorway of the bus?"
    )
    assert intent.relation_anchor_terms == {"doorway"}
    assert intent.relation_source_terms == {"person"}


def test_relation_matching_uses_phrase_boundaries():
    assert not parse_question_intent("What is this photo taken looking through?").relation_terms
    assert parse_question_intent("What is beside the chair?").relation_terms == {"next_to"}


def test_non_object_question_words_do_not_become_detector_queries():
    assert parse_question_intent("Where is he looking?").detector_queries == ("person",)
    assert parse_question_intent("Is this a creamy soup?").detector_queries == ("soup",)
    assert parse_question_intent("Is this rice noodle soup?").detector_queries == ("soup",)
    assert "between" not in parse_question_intent(
        "Why is there a gap between the roof and wall?"
    ).detector_queries


def test_person_pronoun_is_a_relation_source():
    intent = parse_question_intent("What is he on top of?")
    assert intent.object_terms == {"person"}
    assert intent.relation_source_terms == {"person"}
    assert intent.relation_terms == {"on_top_of"}
