import pytest

from gom.config import default_config, validate_paper_config


def test_quality_profile_is_default():
    cfg = default_config()
    assert cfg.profile == "quality_vqa"
    assert cfg.targeted_open_vocabulary
    assert not cfg.singleton_filtering_enabled
    assert cfg.quality_question_pruning
    assert not cfg.fill_segmentation
    assert not cfg.show_bboxes
    assert cfg.min_relations_per_object == 0
    assert cfg.relation_selection_policy == "question_only"
    assert (cfg.obj_fontsize_inside, cfg.obj_fontsize_outside, cfg.rel_fontsize) == (
        14,
        14,
        12,
    )


def test_legacy_profile_is_explicit():
    cfg = default_config(profile="paper_legacy")
    assert cfg.profile == "paper_legacy"
    assert not cfg.targeted_open_vocabulary
    assert cfg.singleton_filtering_enabled
    assert cfg.fill_segmentation


def test_aaai26_profile_is_locked_to_published_pipeline():
    cfg = default_config(profile="paper_aaai26")
    validate_paper_config(cfg)
    assert cfg.relation_selection_policy == "paper_algorithm"
    assert cfg.owl_model_revision == "2a1560802f8cf3c408fec9b809d705f56a2f7146"
    assert cfg.wbf_iou_threshold == 0.9
    assert cfg.sam_version == "hq"
    assert cfg.paper_require_fasttext
    assert cfg.max_detections_total == 0
    assert cfg.max_detections_per_label == 0
    assert cfg.apply_label_nms
    assert cfg.post_segmentation_dedup
    assert cfg.same_class_iou_threshold == 0.5
    assert cfg.same_class_mask_iou_threshold == 0.8
    assert not cfg.cross_class_suppression
    assert set(cfg.render_variants) == {
        "segmented",
        "som_numeric",
        "gom_text_unlabeled",
        "gom_numeric_unlabeled",
        "gom_text_labeled",
        "gom_numeric_labeled",
    }


def test_aaai26_profile_rejects_drift():
    cfg = default_config(profile="paper_aaai26")
    cfg.wbf_iou_threshold = 0.55
    with pytest.raises(ValueError, match="configuration drift"):
        validate_paper_config(cfg)


def test_outline_and_lowfill_variants_differ_from_paper_only_in_fill():
    outline = default_config("paper_aaai26_outline")
    lowfill = default_config("paper_aaai26_lowfill")
    paper = default_config("paper_aaai26")
    assert not outline.fill_segmentation
    assert lowfill.fill_segmentation and lowfill.seg_fill_alpha == 0.10
    for cfg in (outline, lowfill):
        assert cfg.sam_version == paper.sam_version
        assert cfg.paper_require_fasttext
        assert cfg.render_variants == paper.render_variants
    # variants must not trip the paper lock
    validate_paper_config(outline)
    validate_paper_config(lowfill)


def test_outline_clean_and_thin_variants():
    clean = default_config("paper_aaai26_outline_clean")
    thin = default_config("paper_aaai26_outline_thin")
    assert not clean.fill_segmentation and clean.enable_mask_quality_filter
    assert not thin.fill_segmentation and thin.bbox_linewidth == 0.8
    validate_paper_config(clean)
    validate_paper_config(thin)


def test_declutter_variant_minimizes_annotations():
    cfg = default_config("paper_aaai26_declutter")
    assert not cfg.fill_segmentation
    assert cfg.enable_mask_quality_filter
    assert cfg.render_question_relations_only
    assert cfg.max_relations_per_object == 1
    assert cfg.filter_redundant_relations
    validate_paper_config(cfg)
