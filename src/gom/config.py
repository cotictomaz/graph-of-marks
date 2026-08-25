# igp/config.py
"""
IGP Configuration Module

Centralized configuration system for the Image Graph Preprocessing pipeline.
Re-exports configuration dataclasses from individual modules and provides
a fallback PreprocessorConfig to prevent circular import issues during startup.

Configuration Architecture:
    
    PreprocessorConfig (pipeline/preprocessor.py):
        Master configuration containing all pipeline parameters
        - I/O paths and dataset settings
        - Detector selection and thresholds
        - Fusion and NMS parameters
        - Relationship extraction config
        - Visualization settings
        - Performance tuning knobs
    
    SegmenterConfig (segmentation/base.py):
        SAM-based segmentation configuration
        - Model selection (SAM1/SAM2/HQ/Fast)
        - Post-processing options
        - Device and precision settings
    
    RelationsConfig (relations/inference.py):
        Relationship extraction configuration
        - Geometric predicates
        - CLIP-based semantic scoring
        - LLM-guided reasoning (optional)
        - Filtering and pruning thresholds
    
    VisualizerConfig (viz/visualizer.py):
        Visualization rendering configuration
        - Box/label/mask rendering
        - Color schemes
        - Font and layout
        - Output format options

Usage Patterns:

    1. Import specific configs:
        >>> from gom.config import RelationsConfig, VisualizerConfig
        >>> rel_cfg = RelationsConfig(clip_threshold=0.6)
        >>> viz_cfg = VisualizerConfig(show_masks=True)
    
    2. Import PreprocessorConfig (full pipeline):
        >>> from gom.config import PreprocessorConfig
        >>> config = PreprocessorConfig(
        ...     detectors_to_use=("yolov8", "owlvit"),
        ...     question="What objects are in the scene?"
        ... )
    
    3. Use with pipeline:
        >>> from gom.pipeline.preprocessor import Preprocessor
        >>> from gom.config import PreprocessorConfig
        >>> config = PreprocessorConfig(output_folder="results/")
        >>> preprocessor = Preprocessor(config)

Fallback Mechanism:
    During initial import, PreprocessorConfig may not be available due to
    circular dependencies. This module provides a lightweight fallback that
    defines only essential fields. Once pipeline.preprocessor fully loads,
    the real PreprocessorConfig replaces the fallback.

Configuration Hierarchy:
    PreprocessorConfig
    ├── I/O: input_path, output_folder, json_file
    ├── Dataset: dataset, split, num_instances
    ├── Filtering: question, apply_question_filter, aggressive_pruning
    ├── Detectors: detectors_to_use, threshold_*, grounding_dino_*
    ├── Fusion: wbf_iou_threshold, label_nms_threshold, cross_class_*
    ├── Relations: max_relations_per_object, relations_max_clip_pairs
    ├── Segmentation: segmenter, sam_checkpoint_path
    ├── Depth: use_depth, depth_model
    ├── Visualization: visualizer_config, show_*, render_*
    └── Performance: use_cache, batch_size, num_workers

See Also:
    - gom.pipeline.preprocessor: Main pipeline implementation
    - gom.segmentation.base: Segmentation config details
    - gom.relations.inference: Relationship config details
    - gom.viz.visualizer: Visualization config details
"""
from __future__ import annotations

from typing import Any

from gom.relations.inference import RelationsConfig
from gom.segmentation.base import SegmenterConfig
from gom.viz.visualizer import VisualizerConfig

# Re-export configuration dataclasses from their respective modules

# PreprocessorConfig is defined in the pipeline module
# Use try/except to provide a lightweight fallback and avoid circular import errors
try:
    from gom.pipeline.preprocessor import PreprocessorConfig
except Exception as _exc:
    # Fallback configuration class used only during early import phases
    # This will be replaced by the actual PreprocessorConfig once the pipeline module loads
    from dataclasses import dataclass, field
    from typing import Any, Dict, Optional, Tuple

    @dataclass
    class PreprocessorConfig:  # type: ignore[no-redef]
        """
        Fallback preprocessing configuration (lightweight import-time version).
        
        This minimal version prevents circular import errors during module initialization.
        The full PreprocessorConfig from gom.pipeline.preprocessor will replace this
        once all dependencies are loaded.
        
        This fallback includes only the most essential fields. For complete documentation,
        see gom.pipeline.preprocessor.PreprocessorConfig.
        
        Essential Fields:
            input_path: Input image or directory path
            output_folder: Output directory for results
            dataset: Dataset name for batch processing
            question: Question for VQA-aware filtering
            detectors_to_use: Tuple of detector names
        
        Notes:
            - This is a bootstrap class, not the production config
            - Missing many fields from full PreprocessorConfig
            - Should not be used directly in application code
            - Automatically replaced after gom.pipeline loads
        """
        # Segmentation
        segmenter: str = "sam2"
        segmenter_kwargs: Dict[str, Any] = field(default_factory=dict)

        # Output configuration
        output_folder: str = "output_images"

        # Dataset configuration and batching
        dataset: Optional[str] = None
        split: str = "train"
        image_column: str = "image"
        num_instances: int = -1  # -1 means process all instances

        # Question-based filtering parameters
        question: str = ""
        apply_question_filter: bool = True
        aggressive_pruning: bool = False  # Keep only objects mentioned in question
        filter_relations_by_question: bool = True
        relation_selection_policy: str = "question_only"
        paper_ranked_max_relations: int = 16
        threshold_object_similarity: float = 0.50  # CLIP similarity threshold for objects
        threshold_relation_similarity: float = 0.50  # CLIP similarity threshold for relations
        clip_pruning_threshold: float = 0.25  # Minimum CLIP score to keep detection
        semantic_boost_weight: float = 0.4  # Weight for CLIP scores vs detection scores
        context_expansion_radius: float = 2.0  # Radius multiplier for context expansion
        context_min_iou: float = 0.1  # Minimum IoU for context inclusion

        # Object detector configuration and confidence thresholds
        detectors_to_use: Tuple[str, ...] = ("owlvit", "yolov8", "detectron2")
        threshold_owl: float = 0.50  # OWL-ViT confidence threshold
        threshold_yolo: float = 0.50  # YOLOv8 confidence threshold
        threshold_detectron: float = 0.50  # Detectron2 confidence threshold
        threshold_grounding_dino: float = 0.30  # GroundingDINO confidence threshold
        grounding_dino_text_threshold: float = 0.25  # GroundingDINO text similarity threshold
        auto_detector_thresholds: bool = False
        auto_threshold_min_default: float = 0.25
        auto_threshold_min_owl: float = 0.25
        auto_threshold_min_yolo: float = 0.25
        auto_threshold_min_detectron: float = 0.25
        auto_threshold_min_grounding_dino: float = 0.15
        auto_threshold_max_per_detector: Optional[int] = None

        # Relationship inference constraints
        max_relations_per_object: int = 3  # Maximum relationships per object
        # Arrows per rendered image, 0 = unbounded. Bounds the whole render, not
        # just each source object; a crowded render pushes relation labels far
        # off their own arcs.
        max_relations_total: int = 0
        min_relations_per_object: int = 0  # Minimum relationships per object
        # CLIP-based relationship scoring limits (performance tuning)
        relations_max_clip_pairs: int = 1000  # Global limit on CLIP-scored pairs
        relations_per_src_clip_pairs: int = 50  # Per-source limit on CLIP-scored candidates

        # Non-Maximum Suppression and fusion parameters
        label_nms_threshold: float = 0.50  # IoU threshold for per-label NMS
        seg_iou_threshold: float = 0.50  # IoU threshold for segmentation mask merging
        wbf_iou_threshold: float = 0.90  # IoU threshold for Weighted Boxes Fusion
        skip_box_threshold: float = 0.10  # Skip boxes below this confidence in fusion
        # Paper-faithful fusion (c438ebc): raw concat of detectors + per-class greedy NMS,
        # no WBF/cross-class/group-merge/semantic-dedup. Opt-in; default keeps gom behavior.
        paper_faithful_fusion: bool = False
        early_nms_threshold: float = 0.50  # per-class NMS IoU after fusion (paper value)

        # Advanced deduplication and merging options
        cross_class_suppression: bool = True  # Remove overlaps between different classes
        cross_class_iou_threshold: float = 0.65  # IoU threshold for cross-class suppression
        same_class_iou_threshold: float = 0.30  # IoU threshold for same-class deduplication
        same_class_mask_iou_threshold: float = 0.80
        cross_class_score_diff_threshold: float = 0.80  # Score difference ratio for cross-class dedup
        enable_group_merge: bool = False  # Enable semantic grouping and merging
        merge_mask_iou_threshold: float = 0.80  # IoU threshold for mask-based merging
        merge_box_iou_threshold: float = 0.90  # IoU threshold for box-based merging
        mask_union_max_expand_ratio: float = 1.25
        enable_semantic_dedup: bool = False  # Enable CLIP-based semantic deduplication
        semantic_dedup_iou_threshold: float = 0.70  # IoU threshold for semantic dedup
        enable_containment_removal: bool = False  # Remove fully contained detections
        containment_threshold: float = 0.95  # Area overlap threshold for containment
        
        # Cascade and cache management
        cascade_conf_threshold: float = 0.40  # Confidence threshold for cascade fusion
        detection_mask_merge_iou_thr: float = 0.60  # IoU for detection-mask merging
        clip_cache_max_age_days: float = 30.0  # CLIP cache entry expiration (days)
        
        # Non-competing detection recovery (reduces false negatives)
        keep_non_competing_low_scores: bool = True  # Enable low-score recovery
        non_competing_iou_threshold: float = 0.30  # IoU threshold for competition check
        non_competing_min_score: float = 0.05  # Minimum score for recovery

        # geometry
        margin: int = 20
        min_distance: float = 10  # Reduced from 50 to allow closer object relationships
        max_distance: float = 20000

                # SAM settings
        sam_version: str = "1"  # "1" | "2" | "hq"
        sam_hq_model_type: str = "vit_h"
        points_per_side: int = 32
        pred_iou_thresh: float = 0.88
        stability_score_thresh: float = 0.95
        min_mask_region_area: int = 100

        # detection cache
        enable_detection_cache: bool = True
        max_cache_size: int = 100

        # visualization
        label_mode: str = "original"
        display_labels: bool = True
        display_relationships: bool = True
        display_relation_labels: bool = True
        show_segmentation: bool = True
        fill_segmentation: bool = True
        display_legend: bool = False
        seg_fill_alpha: float = 0.25
        bbox_linewidth: float = 2.0
        obj_fontsize_inside: int = 14
        obj_fontsize_outside: int = 14
        rel_fontsize: int = 12
        legend_fontsize: int = 8
        rel_arrow_linewidth: float = 2.0
        rel_arrow_mutation_scale: float = 26.0
        auto_scale_styles: bool = True
        style_ref_px: int = 1000
        style_scale_min: float = 0.5
        style_scale_max: float = 2.0
        obj_fontsize_inside_min: int = 10
        obj_fontsize_outside_min: int = 10
        rel_fontsize_min: int = 9
        render_variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
        resolve_overlaps: bool = True
        show_bboxes: bool = True
        show_confidence: bool = False

        # mask post-processing
        close_holes: bool = True
        hole_kernel: int = 7
        min_hole_area: int = 100
        remove_small_components: bool = True
        min_component_area: int = 150

        # exports
        save_image_only: bool = False
        skip_graph: bool = False
        skip_prompt: bool = False
        skip_visualization: bool = False
        export_preproc_only: bool = False
        output_format: str = "jpg"  # jpg, png, svg
        save_without_background: bool = False
        verbose: bool = False

        # device
        preproc_device: Optional[str] = None
        # If True, always run full preprocessing per question (ignore detection cache)
        force_preprocess_per_question: bool = False
        resume_existing_outputs: bool = False

        # color tweaks
        color_sat_boost: float = 1.1
        color_val_boost: float = 1.1


_LEGACY_PROFILE = {
    "profile": "paper_legacy",
    "targeted_open_vocabulary": False,
    "singleton_filtering_enabled": True,
    "quality_question_pruning": False,
    "render_question_relations_only": False,
    "min_relations_per_object": 1,
    "fill_segmentation": True,
    "seg_fill_alpha": 0.25,
    "show_bboxes": True,
    "obj_fontsize_inside": 9,
    "obj_fontsize_outside": 10,
    "rel_fontsize": 8,
}


PAPER_TABLE2_RENDER_VARIANTS = {
    "segmented": {
        "display_labels": False,
        "display_relationships": False,
        "display_relation_labels": False,
    },
    "som_numeric": {
        "label_mode": "numeric",
        "display_labels": True,
        "display_relationships": False,
        "display_relation_labels": False,
    },
    "gom_text_unlabeled": {
        "label_mode": "original",
        "display_labels": True,
        "display_relationships": True,
        "display_relation_labels": False,
    },
    "gom_numeric_unlabeled": {
        "label_mode": "numeric",
        "display_labels": True,
        "display_relationships": True,
        "display_relation_labels": False,
    },
    "gom_text_labeled": {
        "label_mode": "original",
        "display_labels": True,
        "display_relationships": True,
        "display_relation_labels": True,
    },
    "gom_numeric_labeled": {
        "label_mode": "numeric",
        "display_labels": True,
        "display_relationships": True,
        "display_relation_labels": True,
    },
}


_PAPER_AAAI26_PROFILE = {
    "profile": "paper_aaai26",
    "relation_selection_policy": "paper_algorithm",
    "targeted_open_vocabulary": False,
    "singleton_filtering_enabled": False,
    "quality_question_pruning": False,
    "render_question_relations_only": False,
    "aggressive_pruning": False,
    "use_clip_semantic_pruning": False,
    "context_expansion_enabled": False,
    "false_negative_reduction": False,
    "threshold_owl": 0.50,
    "owl_model_revision": "2a1560802f8cf3c408fec9b809d705f56a2f7146",
    "threshold_yolo": 0.50,
    "threshold_detectron": 0.50,
    "auto_detector_thresholds": False,
    "wbf_iou_threshold": 0.90,
    "ensemble_detector_weights": {
        "owlvit": 2.0,
        "yolov8": 1.5,
        "detectron2": 1.0,
    },
    "paper_faithful_fusion": False,
    "apply_label_nms": True,
    "cross_class_suppression": False,
    "enable_group_merge": False,
    "enable_semantic_dedup": False,
    "same_class_iou_threshold": 0.50,
    "same_class_mask_iou_threshold": 0.80,
    "enable_containment_removal": True,
    "enable_mask_quality_filter": False,
    "post_segmentation_dedup": True,
    "detection_cross_class_suppression_enabled": False,
    "detection_mask_merge_enabled": False,
    "use_spatial_fusion": False,
    "use_hierarchical_fusion": False,
    "detection_resize": False,
    "max_detections_total": 0,
    "max_detections_per_label": 0,
    "max_objects_per_question": 0,
    "max_picture_area_ratio": 1.1,
    "sam_version": "hq",
    "sam_hq_model_type": "vit_h",
    "skip_depth_when_unused": False,
    "max_relations_per_object": 3,
    "min_relations_per_object": 0,
    "margin": 20,
    "paper_direction_margin": 20.0,
    "paper_depth_threshold": 0.10,
    "paper_near_threshold": 5000.0,
    "paper_touching_iou_threshold": 0.10,
    "paper_touching_gap_threshold": 3.0,
    "paper_very_close_threshold": 0.05,
    "paper_close_threshold": 0.12,
    "paper_require_fasttext": True,
    "threshold_object_similarity": 0.50,
    "threshold_relation_similarity": 0.50,
    "fill_segmentation": True,
    "seg_fill_alpha": 0.25,
    "show_bboxes": False,
    "obj_fontsize_inside": 9,
    "obj_fontsize_outside": 10,
    "rel_fontsize": 8,
    # Dense relation graphs need room for the collision resolver to separate
    # labels; the previous 20 px cap forced long labels back onto each other.
    "relation_label_offset_px": 14.0,
    "relation_label_max_dist_px": 50.0,
    "bbox_linewidth": 1.8,
    "rel_arrow_linewidth": 2.0,
    "rel_arrow_mutation_scale": 12.0,
    "label_bbox_linewidth": 1.0,
    "relation_label_bbox_linewidth": 1.0,
    "color_sat_boost": 1.0,
    "color_val_boost": 1.0,
    "auto_scale_styles": False,
    "filter_redundant_relations": False,
    "cap_relations_per_object": False,
    "render_variants": PAPER_TABLE2_RENDER_VARIANTS,
}


PAPER_AAAI26_LOCKED_FIELDS = {
    key: value
    for key, value in _PAPER_AAAI26_PROFILE.items()
    if key not in {"render_variants", "profile"}
}

# Best-config variants: identical to the paper profile except for the mask fill.
# Not locked — they are deliberate deviations for the post-audit runs.
_PAPER_AAAI26_OUTLINE_PROFILE = {
    **_PAPER_AAAI26_PROFILE,
    "profile": "paper_aaai26_outline",
    "fill_segmentation": False,
}
_PAPER_AAAI26_LOWFILL_PROFILE = {
    **_PAPER_AAAI26_PROFILE,
    "profile": "paper_aaai26_lowfill",
    "seg_fill_alpha": 0.10,
}
_PAPER_AAAI26_OUTLINE_CLEAN_PROFILE = {
    **_PAPER_AAAI26_OUTLINE_PROFILE,
    "profile": "paper_aaai26_outline_clean",
    "enable_mask_quality_filter": True,  # drops background/stuff masks (sky, grass, ...)
}
_PAPER_AAAI26_OUTLINE_THIN_PROFILE = {
    **_PAPER_AAAI26_OUTLINE_PROFILE,
    "profile": "paper_aaai26_outline_thin",
    "bbox_linewidth": 0.8,
}
_PAPER_AAAI26_DECLUTTER_PROFILE = {
    **_PAPER_AAAI26_OUTLINE_PROFILE,
    "profile": "paper_aaai26_declutter",
    "enable_mask_quality_filter": True,
    "render_question_relations_only": True,
    "max_relations_per_object": 1,
    "filter_redundant_relations": True,
}

# gom_v2: the post-audit best-effort profile. Outline renders plus everything the
# flip-case audit showed the paper profile was missing: question-derived open-vocab
# detection, detection caps, cross-class dedup, stuff-mask filtering, and a bounded
# fallback when Algorithm 3 matches nothing (instead of marking every detection).
_GOM_V2_PROFILE = {
    **_PAPER_AAAI26_OUTLINE_PROFILE,
    "profile": "gom_v2",
    "targeted_open_vocabulary": True,
    "max_detections_total": 15,
    "max_detections_per_label": 4,
    "cross_class_suppression": True,
    "enable_mask_quality_filter": True,
    "max_relations_per_object": 1,
    "filter_redundant_relations": True,
    "max_picture_area_ratio": 0.95,
    "paper_zero_match_top_k": 6,
}

# gom_v3: gom_v2 plus the fixes from the 20-case flip audit
# (reproduction/FLIP_AUDIT_GOM_V2.md) - deterministic zero-overlap label
# placement, part-of-object fragment dedup, and open-vocabulary detector queries
# (the query change lives in question_intent.py and applies wherever targeted
# open-vocabulary detection is enabled).
_GOM_V3_PROFILE = {
    **_GOM_V2_PROFILE,
    "profile": "gom_v3",
    "deterministic_label_placement": True,
    "measure_text_with_renderer": True,
    "same_class_fragment_containment": 0.70,
}

# gom_v4: gom_v3 plus the fixes from the 20-case gom_v3 flip audit
# (reproduction/FLIP_EXAMPLES_PAPER_GOM.md). The render half - arrows drawn before
# labels, heads reserved in the label registry, endpoints clipped to the target box,
# relation labels seated on the drawn arc with a leader - lives in visualizer.py and
# applies to every profile; the detector-query and relation-selection halves live in
# question_intent.py and relations/paper.py. Only the head size is a profile field:
# 12.0 renders a 6.7 px head that the 6.25 px white halo swallows.
_GOM_V4_PROFILE = {
    **_GOM_V3_PROFILE,
    "profile": "gom_v4",
    "rel_arrow_mutation_scale": 18.0,
    "max_relations_total": 5,
}

# gom_v5: gom_v4 plus the fixes from the gom_v4 flip-gallery review. The render
# half lives in visualizer.py and applies to every profile (arrows are no longer
# clipped to the box, so the shaft is visible; mask outlines use RETR_EXTERNAL plus
# an area floor, so hole boundaries and specks are no longer stroked). The two
# profile fields here are the per-image arrow cap's companion: drop relations whose
# endpoints nearly coincide, since no arc can show their direction.
_GOM_V5_PROFILE = {
    **_GOM_V4_PROFILE,
    "profile": "gom_v5",
    "min_centroid_separation_px": 20.0,
}

_PROFILE_TABLES = {
    "paper_legacy": _LEGACY_PROFILE,
    "paper_aaai26": _PAPER_AAAI26_PROFILE,
    "paper_aaai26_outline": _PAPER_AAAI26_OUTLINE_PROFILE,
    "paper_aaai26_lowfill": _PAPER_AAAI26_LOWFILL_PROFILE,
    "paper_aaai26_outline_clean": _PAPER_AAAI26_OUTLINE_CLEAN_PROFILE,
    "paper_aaai26_outline_thin": _PAPER_AAAI26_OUTLINE_THIN_PROFILE,
    "paper_aaai26_declutter": _PAPER_AAAI26_DECLUTTER_PROFILE,
    "gom_v2": _GOM_V2_PROFILE,
    "gom_v3": _GOM_V3_PROFILE,
    "gom_v4": _GOM_V4_PROFILE,
    "gom_v5": _GOM_V5_PROFILE,
}


def validate_paper_config(cfg: PreprocessorConfig) -> None:
    """Reject silent drift in the immutable paper-declared profile."""
    if getattr(cfg, "profile", None) != "paper_aaai26":
        return
    mismatches = {
        key: (getattr(cfg, key, None), expected)
        for key, expected in PAPER_AAAI26_LOCKED_FIELDS.items()
        if getattr(cfg, key, None) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in sorted(mismatches.items())
        )
        raise ValueError(f"paper_aaai26 configuration drift: {details}")


def default_config(profile: str = "quality_vqa", **overrides: Any) -> PreprocessorConfig:
    """
    Create a PreprocessorConfig with sensible defaults and optional overrides.
    """
    valid = {"quality_vqa", *_PROFILE_TABLES}
    if profile not in valid:
        raise ValueError(f"profile must be one of {sorted(valid)}")
    cfg = PreprocessorConfig()
    cfg.profile = profile
    table = _PROFILE_TABLES.get(profile)
    if table:
        import copy

        for key, value in table.items():
            setattr(cfg, key, copy.deepcopy(value))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
