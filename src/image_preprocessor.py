#!/usr/bin/env python3
"""Image Graph Preprocessor CLI."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

# Load .env (HF_HOME, HF_TOKEN, ...) before importing torch-heavy modules.
try:
    from gom.utils.env import load_dotenv

    load_dotenv()
except Exception:
    pass

from gom.config import default_config, PreprocessorConfig
from gom.pipeline.preprocessor import ImageGraphPreprocessor as Preprocessor


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image Graph Preprocessor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--profile",
        choices=["quality_vqa", "paper_legacy", "paper_aaai26"],
        default="quality_vqa",
        help="Quality-first defaults or historical paper rendering/filtering",
    )

    # I/O
    p.add_argument("--input_path", type=str, default=None)
    p.add_argument("--json_file", type=str, default="")
    p.add_argument("--output_folder", type=str, default="output_images")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--image_column", type=str, default="image")
    p.add_argument("--num_instances", type=int, default=-1)

    # Question filtering
    p.add_argument("--question", type=str, default="")
    p.add_argument("--disable_question_filter", action="store_true")
    p.add_argument("--aggressive_pruning", action="store_true")
    p.add_argument("--no_filter_relations_by_question", action="store_true")
    p.add_argument("--threshold_object_similarity", type=float, default=0.50)
    p.add_argument("--threshold_relation_similarity", type=float, default=0.50)
    p.add_argument("--clip_pruning_threshold", type=float, default=0.25)
    p.add_argument("--semantic_boost_weight", type=float, default=0.4)
    p.add_argument("--context_expansion_radius", type=float, default=2.0)
    p.add_argument("--context_min_iou", type=float, default=0.1)

    # Detectors
    p.add_argument("--detectors", type=str, default="owlvit,yolov8,detectron2")
    p.add_argument("--owl_threshold", type=float, default=0.50)
    p.add_argument("--yolo_threshold", type=float, default=0.50)
    p.add_argument("--detectron_threshold", type=float, default=0.50)
    p.add_argument("--grounding_dino_threshold", type=float, default=0.30)
    p.add_argument("--grounding_dino_text_threshold", type=float, default=0.25)

    # Relations
    p.add_argument("--max_relations_per_object", type=int, default=3)
    p.add_argument("--min_relations_per_object", type=int, default=None)
    p.add_argument(
        "--relation_selection_policy",
        choices=["question_only", "paper_ranked", "paper_algorithm"],
        default=None,
    )
    p.add_argument("--paper_ranked_max_relations", type=int, default=None)
    p.add_argument("--paper_fasttext_path", type=str, default=None)
    p.add_argument("--targeted_owl_threshold", type=float, default=0.20)
    p.add_argument("--margin", type=int, default=20)
    p.add_argument("--min_distance", type=float, default=10)
    p.add_argument("--max_distance", type=float, default=20000)

    # NMS / fusion
    p.add_argument("--label_nms_threshold", type=float, default=0.50)
    p.add_argument("--seg_iou_threshold", type=float, default=0.50)
    p.add_argument("--wbf_iou_threshold", type=float, default=0.90)
    p.add_argument("--skip_box_threshold", type=float, default=0.10)
    p.add_argument("--paper_faithful_fusion", action="store_true",
                   help="Replicate paper c438ebc detection merge: raw concat + per-class NMS (no WBF/cross-class cascades)")
    p.add_argument("--early_nms_threshold", type=float, default=0.50,
                   help="Per-class NMS IoU applied after fusion in paper-faithful mode")
    p.add_argument("--cross_class_iou_threshold", type=float, default=0.75)
    p.add_argument("--same_class_iou_threshold", type=float, default=None)
    p.add_argument("--cross_class_score_diff_threshold", type=float, default=0.80)
    p.add_argument("--cascade_conf_threshold", type=float, default=0.40)
    p.add_argument("--detection_mask_merge_iou_thr", type=float, default=0.60)
    p.add_argument("--clip_cache_max_age_days", type=float, default=30.0)
    p.add_argument(
        "--keep_non_competing_low_scores",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--non_competing_iou_threshold", type=float, default=0.30)
    p.add_argument("--non_competing_min_score", type=float, default=0.05)

    # SAM
    p.add_argument("--sam_version", type=str, choices=["1", "2", "hq"], default="hq")
    p.add_argument("--sam_hq_model_type", type=str, choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.90)
    p.add_argument("--stability_score_thresh", type=float, default=0.92)
    p.add_argument("--min_mask_region_area", type=int, default=100)
    p.add_argument("--preproc_device", type=str, default=None)

    # Visualization
    p.add_argument("--label_mode", type=str, choices=["original", "numeric", "alphabetic"], default="original")
    p.add_argument("--display_labels", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--display_relationships", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--display_relation_labels", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--show_segmentation", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fill_segmentation", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no_legend", action="store_true", default=None)
    p.add_argument("--seg_fill_alpha", type=float, default=None)
    p.add_argument("--bbox_linewidth", type=float, default=None)
    p.add_argument("--obj_fontsize_inside", type=int, default=None)
    p.add_argument("--obj_fontsize_outside", type=int, default=None)
    p.add_argument("--rel_fontsize", type=int, default=None)
    p.add_argument("--legend_fontsize", type=int, default=8)
    p.add_argument("--rel_arrow_linewidth", type=float, default=None)
    p.add_argument("--rel_arrow_mutation_scale", type=float, default=None)
    p.add_argument(
        "--auto_scale_styles",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--style_ref_px", type=int, default=None)
    p.add_argument("--style_scale_min", type=float, default=None)
    p.add_argument("--style_scale_max", type=float, default=None)
    p.add_argument("--obj_fontsize_inside_min", type=int, default=None)
    p.add_argument("--obj_fontsize_outside_min", type=int, default=None)
    p.add_argument("--rel_fontsize_min", type=int, default=None)
    p.add_argument(
        "--render_variants_json",
        type=str,
        default="",
        help="JSON object mapping variant names to VisualizerConfig overrides",
    )
    p.add_argument(
        "--resolve_overlaps",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--show_bboxes", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no_bboxes", action="store_true", default=None)
    p.add_argument("--no_masks", action="store_true")
    p.add_argument("--no_instances", action="store_true")
    p.add_argument("--show_confidence", action="store_true")

    # Output format
    p.add_argument("--output_format", type=str, choices=["jpg", "png", "svg"], default="jpg")
    p.add_argument("--save_without_background", action="store_true")

    # Colors
    p.add_argument("--color_sat_boost", type=float, default=None)
    p.add_argument("--color_val_boost", type=float, default=None)

    # Mask post-processing
    p.add_argument("--close_holes", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--hole_kernel", type=int, default=7)
    p.add_argument("--min_hole_area", type=int, default=100)

    # Output options
    p.add_argument("--save_image_only", action="store_true")
    p.add_argument("--skip_graph", action="store_true")
    p.add_argument("--skip_prompt", action="store_true")
    p.add_argument("--skip_visualization", action="store_true")
    p.add_argument("--export_preproc_only", action="store_true")

    # Cache
    p.add_argument(
        "--enable_detection_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--max_cache_size", type=int, default=100)
    p.add_argument("--clear_cache", action="store_true")

    # Misc
    p.add_argument("--config", type=str, default="")
    p.add_argument("--save_config", type=str, default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verbose", action="count", default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose complete graph/render outputs already exist",
    )
    p.add_argument("--version", action="store_true")

    return p.parse_args(argv)


def _merge_cfg_from_dict(cfg: PreprocessorConfig, data: Dict[str, Any]) -> None:
    for k, v in (data or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


# Config attribute -> CLI dest, for the few options whose names differ.
_CFG_ATTR_TO_CLI_DEST = {
    "threshold_owl": "owl_threshold",
    "threshold_yolo": "yolo_threshold",
    "threshold_detectron": "detectron_threshold",
    "threshold_grounding_dino": "grounding_dino_threshold",
    "detectors_to_use": "detectors",
    "apply_question_filter": "disable_question_filter",
    "filter_relations_by_question": "no_filter_relations_by_question",
    "display_legend": "no_legend",
    "show_bboxes": "no_bboxes",
    "preproc_device": "preproc_device",
}


def _explicit_cli_dests(argv: list[str]) -> set:
    """Dest names of options the user actually typed on the command line."""
    dests = set()
    for tok in argv:
        if tok.startswith("--"):
            dests.add(tok[2:].split("=", 1)[0].replace("-", "_"))
    return dests


def _load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_config_file(cfg: PreprocessorConfig, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _apply_optional_flags(cfg: PreprocessorConfig, args: argparse.Namespace) -> None:
    opt_map = {
        "seed": args.seed,
        "num_workers": args.workers,
        "no_progress": bool(args.no_progress),
        "verbose": int(args.verbose or 0),
    }
    for k, v in opt_map.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)


def _build_config(args: argparse.Namespace) -> PreprocessorConfig:
    cfg = default_config(profile=args.profile)

    # I/O
    cfg.input_path = args.input_path
    cfg.json_file = args.json_file
    cfg.output_folder = args.output_folder
    cfg.dataset = args.dataset
    cfg.split = args.split
    cfg.image_column = args.image_column
    cfg.num_instances = args.num_instances

    # Question filtering
    cfg.question = args.question
    cfg.apply_question_filter = not args.disable_question_filter
    cfg.aggressive_pruning = bool(args.aggressive_pruning)
    cfg.filter_relations_by_question = not args.no_filter_relations_by_question
    cfg.threshold_object_similarity = float(args.threshold_object_similarity)
    cfg.threshold_relation_similarity = float(args.threshold_relation_similarity)
    cfg.clip_pruning_threshold = float(args.clip_pruning_threshold)
    cfg.semantic_boost_weight = float(args.semantic_boost_weight)
    cfg.context_expansion_radius = float(args.context_expansion_radius)
    cfg.context_min_iou = float(args.context_min_iou)

    # Detectors
    cfg.detectors_to_use = tuple(d.strip() for d in args.detectors.split(",") if d.strip())
    cfg.threshold_owl = float(args.owl_threshold)
    cfg.threshold_yolo = float(args.yolo_threshold)
    cfg.threshold_detectron = float(args.detectron_threshold)
    cfg.threshold_grounding_dino = float(args.grounding_dino_threshold)
    cfg.grounding_dino_text_threshold = float(args.grounding_dino_text_threshold)

    # Relations
    cfg.max_relations_per_object = int(args.max_relations_per_object)
    if args.min_relations_per_object is not None:
        cfg.min_relations_per_object = int(args.min_relations_per_object)
    if args.relation_selection_policy is not None:
        cfg.relation_selection_policy = args.relation_selection_policy
    if args.paper_ranked_max_relations is not None:
        cfg.paper_ranked_max_relations = int(args.paper_ranked_max_relations)
    if args.paper_fasttext_path is not None:
        cfg.paper_fasttext_path = args.paper_fasttext_path
    cfg.targeted_owl_threshold = float(args.targeted_owl_threshold)
    cfg.margin = int(args.margin)
    cfg.min_distance = float(args.min_distance)
    cfg.max_distance = float(args.max_distance)

    # NMS / fusion
    cfg.label_nms_threshold = float(args.label_nms_threshold)
    cfg.seg_iou_threshold = float(args.seg_iou_threshold)
    cfg.wbf_iou_threshold = float(args.wbf_iou_threshold)
    cfg.skip_box_threshold = float(args.skip_box_threshold)
    cfg.paper_faithful_fusion = bool(args.paper_faithful_fusion)
    cfg.early_nms_threshold = float(args.early_nms_threshold)
    cfg.cross_class_iou_threshold = float(args.cross_class_iou_threshold)
    if args.same_class_iou_threshold is not None:
        cfg.same_class_iou_threshold = float(args.same_class_iou_threshold)
    cfg.cross_class_score_diff_threshold = float(args.cross_class_score_diff_threshold)
    cfg.cascade_conf_threshold = float(args.cascade_conf_threshold)
    cfg.detection_mask_merge_iou_thr = float(args.detection_mask_merge_iou_thr)
    cfg.clip_cache_max_age_days = float(args.clip_cache_max_age_days)
    if args.keep_non_competing_low_scores is not None:
        cfg.keep_non_competing_low_scores = bool(args.keep_non_competing_low_scores)
    cfg.non_competing_iou_threshold = float(args.non_competing_iou_threshold)
    cfg.non_competing_min_score = float(args.non_competing_min_score)

    # SAM
    cfg.sam_version = args.sam_version
    cfg.sam_hq_model_type = args.sam_hq_model_type
    cfg.points_per_side = int(args.points_per_side)
    cfg.pred_iou_thresh = float(args.pred_iou_thresh)
    cfg.stability_score_thresh = float(args.stability_score_thresh)
    cfg.min_mask_region_area = int(args.min_mask_region_area)
    cfg.preproc_device = args.preproc_device

    # Visualization
    cfg.label_mode = args.label_mode
    if args.display_labels is not None:
        cfg.display_labels = bool(args.display_labels)
    if args.display_relationships is not None:
        cfg.display_relationships = bool(args.display_relationships)
    if args.display_relation_labels is not None:
        cfg.display_relation_labels = bool(args.display_relation_labels)

    if args.no_instances:
        cfg.show_segmentation = False
        cfg.show_bboxes = False
    else:
        if args.show_segmentation is not None:
            cfg.show_segmentation = bool(args.show_segmentation)
        cfg.show_segmentation = cfg.show_segmentation and not args.no_masks
        if args.show_bboxes is not None:
            cfg.show_bboxes = bool(args.show_bboxes)
        if args.no_bboxes:
            cfg.show_bboxes = False

    if args.fill_segmentation is not None:
        cfg.fill_segmentation = bool(args.fill_segmentation)
    if args.no_legend:
        cfg.display_legend = False
    if args.seg_fill_alpha is not None:
        cfg.seg_fill_alpha = float(args.seg_fill_alpha)
    if args.bbox_linewidth is not None:
        cfg.bbox_linewidth = float(args.bbox_linewidth)
    if args.obj_fontsize_inside is not None:
        cfg.obj_fontsize_inside = int(args.obj_fontsize_inside)
    if args.obj_fontsize_outside is not None:
        cfg.obj_fontsize_outside = int(args.obj_fontsize_outside)
    if args.rel_fontsize is not None:
        cfg.rel_fontsize = int(args.rel_fontsize)
    cfg.legend_fontsize = int(args.legend_fontsize)
    if args.rel_arrow_linewidth is not None:
        cfg.rel_arrow_linewidth = float(args.rel_arrow_linewidth)
    if args.rel_arrow_mutation_scale is not None:
        cfg.rel_arrow_mutation_scale = float(args.rel_arrow_mutation_scale)
    if args.auto_scale_styles is not None:
        cfg.auto_scale_styles = bool(args.auto_scale_styles)
    if args.style_ref_px is not None:
        cfg.style_ref_px = int(args.style_ref_px)
    if args.style_scale_min is not None:
        cfg.style_scale_min = float(args.style_scale_min)
    if args.style_scale_max is not None:
        cfg.style_scale_max = float(args.style_scale_max)
    if args.obj_fontsize_inside_min is not None:
        cfg.obj_fontsize_inside_min = int(args.obj_fontsize_inside_min)
    if args.obj_fontsize_outside_min is not None:
        cfg.obj_fontsize_outside_min = int(args.obj_fontsize_outside_min)
    if args.rel_fontsize_min is not None:
        cfg.rel_fontsize_min = int(args.rel_fontsize_min)
    if args.render_variants_json:
        variants = _load_config_file(args.render_variants_json)
        if not isinstance(variants, dict):
            raise ValueError("--render_variants_json must contain a JSON object")
        cfg.render_variants = variants
    if args.resolve_overlaps is not None:
        cfg.resolve_overlaps = bool(args.resolve_overlaps)
    cfg.show_confidence = bool(args.show_confidence)

    # Colors
    if args.color_sat_boost is not None:
        cfg.color_sat_boost = float(args.color_sat_boost)
    if args.color_val_boost is not None:
        cfg.color_val_boost = float(args.color_val_boost)

    # Mask post-processing
    if args.close_holes is not None:
        cfg.close_holes = bool(args.close_holes)
    cfg.hole_kernel = int(args.hole_kernel)
    cfg.min_hole_area = int(args.min_hole_area)

    # Output
    cfg.save_image_only = bool(args.save_image_only)
    cfg.skip_graph = bool(args.skip_graph)
    cfg.skip_prompt = bool(args.skip_prompt)
    cfg.skip_visualization = bool(args.skip_visualization)
    cfg.export_preproc_only = bool(args.export_preproc_only)
    cfg.output_format = args.output_format
    cfg.save_without_background = bool(args.save_without_background)

    # Cache
    if args.enable_detection_cache is not None:
        cfg.enable_detection_cache = bool(args.enable_detection_cache)
    cfg.max_cache_size = int(args.max_cache_size)

    _apply_optional_flags(cfg, args)
    cfg.resume_existing_outputs = bool(args.resume)
    return cfg


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(int(args.verbose or 0))

    if args.version:
        print("Image Graph Preprocessor")
        return 0

    cfg = _build_config(args)

    if args.config:
        # Precedence: CLI defaults < config file < explicitly passed CLI flags.
        try:
            overrides = _load_config_file(args.config)
            explicit = _explicit_cli_dests(argv if argv is not None else sys.argv[1:])
            overrides = {
                k: v for k, v in overrides.items()
                if _CFG_ATTR_TO_CLI_DEST.get(k, k) not in explicit
            }
            _merge_cfg_from_dict(cfg, overrides)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return 2

    if cfg.profile == "paper_aaai26":
        try:
            from gom.config import validate_paper_config

            validate_paper_config(cfg)
        except Exception as e:
            logging.error(str(e))
            return 2

    if args.save_config:
        try:
            _dump_config_file(cfg, args.save_config)
            logging.info(f"Config saved to: {args.save_config}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return 2

    if args.dry_run:
        data = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    if not (args.input_path or args.json_file or args.dataset):
        logging.error("Specify at least one of --input_path, --json_file, or --dataset")
        return 2

    if args.clear_cache:
        pre = Preprocessor(cfg)
        pre.clear_caches()
        if not (args.input_path or args.json_file or args.dataset):
            return 0

    try:
        preproc = Preprocessor(cfg)
        preproc.run()
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception as e:
        logging.exception(f"Pipeline error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
