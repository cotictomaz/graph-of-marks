"""
Graph-of-Marks (GoM) Integrity Test Suite

This test suite validates the integrity of the codebase without running actual ML models.
It tests:
- All main imports work correctly
- Core classes can be instantiated
- Key functions exist and have correct signatures
- Configuration objects work properly
- Utility functions produce expected results

Run with: pytest tests/test_integrity.py -v
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from dataclasses import fields
import inspect


# =============================================================================
# SECTION 1: IMPORT TESTS
# =============================================================================

class TestMainImports:
    """Test that all main API imports work correctly."""

    def test_import_gom_main(self):
        """Test main gom package import."""
        import gom
        assert hasattr(gom, 'GoM')
        assert hasattr(gom, 'ProcessingConfig')

    def test_import_gom_api(self):
        """Test gom.api module imports."""
        from gom.api import GoM, ProcessingConfig, create_pipeline, run
        assert callable(GoM)
        assert callable(ProcessingConfig)
        assert callable(create_pipeline)
        assert callable(run)

    def test_import_backward_compat_aliases(self):
        """Test backward compatibility aliases exist."""
        from gom.api import Gom, GraphOfMarks
        from gom.api import GoM
        assert Gom is GoM
        assert GraphOfMarks is GoM

    def test_import_style_presets(self):
        """Test style presets are available."""
        from gom.api import GOM_STYLE_PRESETS
        assert isinstance(GOM_STYLE_PRESETS, dict)
        assert 'som_text' in GOM_STYLE_PRESETS
        assert 'gom_text' in GOM_STYLE_PRESETS
        assert 'gom_numeric_labeled' in GOM_STYLE_PRESETS


class TestTypeImports:
    """Test that core type imports work."""

    def test_import_detection(self):
        """Test Detection type import."""
        from gom.types import Detection
        assert callable(Detection)

    def test_import_relationship(self):
        """Test Relationship type import."""
        from gom.types import Relationship
        assert callable(Relationship)

    def test_import_box_type(self):
        """Test Box type alias import."""
        from gom.types import Box
        assert Box is not None


class TestConfigImports:
    """Test that configuration classes can be imported."""

    def test_import_preprocessor_config(self):
        """Test PreprocessorConfig import."""
        from gom.config import PreprocessorConfig
        assert callable(PreprocessorConfig)

    def test_import_default_config(self):
        """Test default_config helper import."""
        from gom.config import default_config
        assert callable(default_config)


class TestDetectorImports:
    """Test detector module imports."""

    def test_import_detector_base(self):
        """Test Detector base class import."""
        from gom.detectors.base import Detector
        assert Detector is not None

    def test_import_detector_manager(self):
        """Test DetectorManager import."""
        from gom.detectors.manager import DetectorManager
        assert callable(DetectorManager)

    def test_import_yolov8_detector(self):
        """Test YOLOv8Detector import."""
        from gom.detectors.yolov8 import YOLOv8Detector
        assert callable(YOLOv8Detector)

    def test_import_owlvit_detector(self):
        """Test OwlViTDetector import."""
        from gom.detectors.owlvit import OwlViTDetector
        assert callable(OwlViTDetector)


class TestSegmentationImports:
    """Test segmentation module imports."""

    def test_import_segmenter_base(self):
        """Test Segmenter base class import."""
        from gom.segmentation.base import Segmenter, SegmenterConfig
        assert Segmenter is not None
        assert callable(SegmenterConfig)

    def test_import_sam2_segmenter(self):
        """Test Sam2Segmenter import."""
        from gom.segmentation.sam2 import Sam2Segmenter
        assert callable(Sam2Segmenter)


class TestRelationsImports:
    """Test relations module imports."""

    def test_import_relation_inferencer(self):
        """Test RelationInferencer import."""
        from gom.relations.inference import RelationInferencer
        assert callable(RelationInferencer)

    def test_import_relations_config(self):
        """Test RelationsConfig import."""
        from gom.relations.inference import RelationsConfig
        assert callable(RelationsConfig)

    def test_import_geometry_functions(self):
        """Test geometry module imports."""
        from gom.relations.geometry.core import as_xyxy, iou, center
        assert callable(as_xyxy)
        assert callable(iou)
        assert callable(center)


class TestVisualizationImports:
    """Test visualization module imports."""

    def test_import_visualizer(self):
        """Test Visualizer import."""
        from gom.viz.visualizer import Visualizer, VisualizerConfig
        assert callable(Visualizer)
        assert callable(VisualizerConfig)


class TestGraphImports:
    """Test graph module imports."""

    def test_import_scene_graph_builder(self):
        """Test SceneGraphBuilder import."""
        from gom.graph.scene_graph import SceneGraphBuilder
        assert callable(SceneGraphBuilder)

    def test_import_graph_prompt_utils(self):
        """Test graph prompt utilities import."""
        from gom.graph.prompt import graph_to_prompt, graph_to_triples_text
        assert callable(graph_to_prompt)
        assert callable(graph_to_triples_text)


class TestUtilityImports:
    """Test utility module imports."""

    def test_import_box_utils(self):
        """Test box utility imports."""
        from gom.utils.boxes import area, iou, center, center_distance, nms, clamp_xyxy
        assert callable(area)
        assert callable(iou)
        assert callable(center)
        assert callable(center_distance)
        assert callable(nms)
        assert callable(clamp_xyxy)

    def test_import_color_utils(self):
        """Test color utility imports."""
        from gom.utils.colors import ColorCycler, text_color_for_bg, base_label
        assert callable(ColorCycler)
        assert callable(text_color_for_bg)
        assert callable(base_label)

    def test_import_depth_estimator(self):
        """Test depth estimator imports."""
        from gom.utils.depth_v2 import DepthEstimatorV2, DepthConfig, DepthModel
        assert callable(DepthEstimatorV2)
        assert callable(DepthConfig)
        assert DepthModel is not None

    def test_import_cache_utils(self):
        """Test cache utility imports."""
        from gom.utils.cache_advanced import LRUCache, ImageDetectionCache, CacheStats
        assert callable(LRUCache)
        assert callable(ImageDetectionCache)
        assert callable(CacheStats)


class TestPipelineImports:
    """Test pipeline module imports."""

    def test_import_preprocessor(self):
        """Test ImageGraphPreprocessor import."""
        from gom.pipeline.preprocessor import ImageGraphPreprocessor
        assert callable(ImageGraphPreprocessor)


class TestFusionImports:
    """Test fusion module imports."""

    def test_import_fusion_methods(self):
        """Test fusion method imports."""
        from gom.fusion.wbf import fuse_detections_wbf
        assert callable(fuse_detections_wbf)


# =============================================================================
# SECTION 2: CORE TYPE TESTS
# =============================================================================

class TestDetectionType:
    """Test Detection dataclass functionality."""

    def test_detection_creation(self):
        """Test creating a Detection object."""
        from gom.types import Detection
        det = Detection(
            box=(10.0, 20.0, 100.0, 150.0),
            label="person",
            score=0.95
        )
        assert det.box == (10.0, 20.0, 100.0, 150.0)
        assert det.label == "person"
        assert det.score == 0.95

    def test_detection_default_values(self):
        """Test Detection default values."""
        from gom.types import Detection
        det = Detection(box=(0, 0, 10, 10), label="test")
        assert det.score == 1.0
        assert det.source is None

    def test_detection_with_extra(self):
        """Test Detection with extra metadata."""
        from gom.types import Detection
        det = Detection(
            box=(0, 0, 10, 10),
            label="test",
            extra={"mask": np.zeros((10, 10))}
        )
        assert "mask" in det.extra


class TestRelationshipType:
    """Test Relationship dataclass functionality."""

    def test_relationship_creation(self):
        """Test creating a Relationship object."""
        from gom.types import Relationship
        rel = Relationship(
            src_idx=0,
            tgt_idx=1,
            relation="left_of"
        )
        assert rel.src_idx == 0
        assert rel.tgt_idx == 1
        assert rel.relation == "left_of"

    def test_relationship_with_distance(self):
        """Test Relationship with distance."""
        from gom.types import Relationship
        rel = Relationship(
            src_idx=0,
            tgt_idx=1,
            relation="near",
            distance=50.5
        )
        assert rel.distance == 50.5


# =============================================================================
# SECTION 3: CONFIGURATION TESTS
# =============================================================================

class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""

    def test_processing_config_creation(self):
        """Test creating ProcessingConfig with defaults."""
        from gom.api import ProcessingConfig
        config = ProcessingConfig()
        assert hasattr(config, 'threshold')
        assert hasattr(config, 'style')
        assert hasattr(config, 'display_labels')

    def test_processing_config_from_style(self):
        """Test creating ProcessingConfig from style preset."""
        from gom.api import ProcessingConfig
        config = ProcessingConfig.from_style("gom_text")
        assert config.style == "gom_text"
        assert config.display_relationships is True

    def test_processing_config_with_overrides(self):
        """Test ProcessingConfig with custom overrides."""
        from gom.api import ProcessingConfig
        config = ProcessingConfig.from_style("som_text", threshold=0.5)
        assert config.threshold == 0.5


class TestPreprocessorConfig:
    """Test PreprocessorConfig dataclass."""

    def test_preprocessor_config_creation(self):
        """Test creating PreprocessorConfig with defaults."""
        from gom.config import PreprocessorConfig
        config = PreprocessorConfig()
        assert hasattr(config, 'detectors_to_use')
        assert hasattr(config, 'sam_version')

    def test_preprocessor_config_fields(self):
        """Test PreprocessorConfig has expected fields."""
        from gom.config import PreprocessorConfig
        field_names = {f.name for f in fields(PreprocessorConfig)}
        assert 'threshold_yolo' in field_names
        assert 'sam_version' in field_names
        assert 'max_relations_per_object' in field_names

    def test_default_config_helper(self):
        """Test default_config helper function."""
        from gom.config import default_config
        config = default_config(verbose=True)
        assert config.verbose is True


class TestRelationsConfig:
    """Test RelationsConfig dataclass."""

    def test_relations_config_creation(self):
        """Test creating RelationsConfig."""
        from gom.relations.inference import RelationsConfig
        config = RelationsConfig()
        assert hasattr(config, 'max_relations_per_object')
        assert hasattr(config, 'use_geometric_relations')
        assert hasattr(config, 'use_clip_relations')


class TestVisualizerConfig:
    """Test VisualizerConfig dataclass."""

    def test_visualizer_config_creation(self):
        """Test creating VisualizerConfig."""
        from gom.viz.visualizer import VisualizerConfig
        config = VisualizerConfig()
        assert hasattr(config, 'display_labels')
        assert hasattr(config, 'show_segmentation')


class TestDepthConfig:
    """Test DepthConfig dataclass."""

    def test_depth_config_creation(self):
        """Test creating DepthConfig."""
        from gom.utils.depth_v2 import DepthConfig, DepthModel
        config = DepthConfig()
        assert config.model_name == DepthModel.DEPTH_ANYTHING_V2_LARGE
        assert config.lazy_load is True

    def test_depth_model_enum(self):
        """Test DepthModel enum values."""
        from gom.utils.depth_v2 import DepthModel
        assert DepthModel.DEPTH_ANYTHING_V2_SMALL.value == "depth_anything_v2_vits"
        assert DepthModel.MIDAS_DPT_LARGE.value == "DPT_Large"


# =============================================================================
# SECTION 4: UTILITY FUNCTION TESTS
# =============================================================================

class TestBoxUtils:
    """Test box utility functions."""

    def test_area_calculation(self):
        """Test box area calculation."""
        from gom.utils.boxes import area
        box = (0, 0, 10, 20)
        assert area(box) == 200.0

    def test_area_zero(self):
        """Test area of zero-size box."""
        from gom.utils.boxes import area
        box = (5, 5, 5, 5)
        assert area(box) == 0.0

    def test_iou_calculation(self):
        """Test IoU calculation."""
        from gom.utils.boxes import iou
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        result = iou(box1, box2)
        assert 0 <= result <= 1
        # Intersection: (5,5) to (10,10) = 25
        # Union: 100 + 100 - 25 = 175
        assert abs(result - 25/175) < 1e-6

    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        from gom.utils.boxes import iou
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert iou(box1, box2) == 0.0

    def test_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        from gom.utils.boxes import iou
        box = (0, 0, 10, 10)
        assert iou(box, box) == 1.0

    def test_center_calculation(self):
        """Test center calculation."""
        from gom.utils.boxes import center
        box = (0, 0, 10, 20)
        cx, cy = center(box)
        assert cx == 5.0
        assert cy == 10.0

    def test_center_distance(self):
        """Test center distance calculation."""
        from gom.utils.boxes import center_distance
        box1 = (0, 0, 10, 10)  # center (5, 5)
        box2 = (10, 0, 20, 10)  # center (15, 5)
        dist = center_distance(box1, box2)
        assert dist == 10.0

    def test_clamp_xyxy(self):
        """Test box clamping to image bounds."""
        from gom.utils.boxes import clamp_xyxy
        box = (-10, -10, 200, 150)
        clamped = clamp_xyxy(box, W=100, H=100)
        assert clamped[0] >= 0
        assert clamped[1] >= 0
        assert clamped[2] <= 100
        assert clamped[3] <= 100

    def test_nms(self):
        """Test non-maximum suppression."""
        from gom.utils.boxes import nms
        boxes = [
            (0, 0, 10, 10),
            (1, 1, 11, 11),  # overlaps with first
            (50, 50, 60, 60),  # no overlap
        ]
        scores = [0.9, 0.8, 0.7]
        kept = nms(boxes, scores, iou_thresh=0.5)
        assert 0 in kept  # highest score
        assert 2 in kept  # no overlap
        assert len(kept) == 2


class TestColorUtils:
    """Test color utility functions."""

    def test_color_cycler_creation(self):
        """Test ColorCycler can be created."""
        from gom.utils.colors import ColorCycler
        cycler = ColorCycler()
        assert cycler is not None

    def test_color_for_label(self):
        """Test color generation for label using ColorCycler."""
        from gom.utils.colors import ColorCycler
        cycler = ColorCycler()
        color = cycler.color_for_label("person")
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # #RRGGBB

    def test_color_consistency(self):
        """Test same label always gets same color in same cycler."""
        from gom.utils.colors import ColorCycler
        cycler = ColorCycler()
        color1 = cycler.color_for_label("person")
        color2 = cycler.color_for_label("person")
        assert color1 == color2

    def test_text_color_for_bg(self):
        """Test text color selection for background."""
        from gom.utils.colors import text_color_for_bg
        # Light background should get dark text
        light_text = text_color_for_bg("#FFFFFF")
        dark_text = text_color_for_bg("#000000")
        assert light_text != dark_text

    def test_base_label(self):
        """Test base label extraction."""
        from gom.utils.colors import base_label
        assert base_label("person_1") == "person"
        assert base_label("cat") == "cat"


class TestGeometryFunctions:
    """Test geometry relation functions."""

    def test_as_xyxy(self):
        """Test as_xyxy conversion."""
        from gom.relations.geometry.core import as_xyxy
        box = [10, 20, 30, 40]
        result = as_xyxy(box)
        assert result == (10, 20, 30, 40)

    def test_geometry_area(self):
        """Test geometry module area calculation."""
        from gom.relations.geometry.core import area
        box = (0, 0, 10, 10)
        assert area(box) == 100.0

    def test_geometry_center(self):
        """Test geometry module center calculation."""
        from gom.relations.geometry.core import center
        box = (0, 0, 10, 10)
        cx, cy = center(box)
        assert cx == 5.0
        assert cy == 5.0


# =============================================================================
# SECTION 5: CACHE TESTS
# =============================================================================

class TestLRUCache:
    """Test LRU cache functionality."""

    def test_cache_creation(self):
        """Test cache creation."""
        from gom.utils.cache_advanced import LRUCache
        cache = LRUCache(max_items=10, max_size_mb=100.0)
        assert len(cache) == 0

    def test_cache_put_get(self):
        """Test cache put and get operations."""
        from gom.utils.cache_advanced import LRUCache
        cache = LRUCache(max_items=10)
        cache.put("key1", {"data": [1, 2, 3]})
        result = cache.get("key1")
        assert result == {"data": [1, 2, 3]}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from gom.utils.cache_advanced import LRUCache
        cache = LRUCache(max_items=10)
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        """Test LRU eviction."""
        from gom.utils.cache_advanced import LRUCache
        cache = LRUCache(max_items=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_cache_clear(self):
        """Test cache clear."""
        from gom.utils.cache_advanced import LRUCache
        cache = LRUCache(max_items=10)
        cache.put("key", "value")
        cache.clear()
        assert len(cache) == 0


class TestImageDetectionCache:
    """Test ImageDetectionCache functionality."""

    def test_detection_cache_creation(self):
        """Test detection cache creation."""
        from gom.utils.cache_advanced import ImageDetectionCache
        cache = ImageDetectionCache(max_items=50)
        assert len(cache) == 0

    def test_detection_cache_key_generation(self):
        """Test cache key generation."""
        from gom.utils.cache_advanced import ImageDetectionCache
        img = Image.new("RGB", (100, 100), color="red")
        key = ImageDetectionCache.generate_key(
            image=img,
            detectors=("yolov8",),
            thresholds={"yolov8": 0.5}
        )
        assert isinstance(key, str)
        assert key.startswith("det_")

    def test_detection_cache_deterministic_keys(self):
        """Test that same inputs produce same key."""
        from gom.utils.cache_advanced import ImageDetectionCache
        img = Image.new("RGB", (100, 100), color="red")
        key1 = ImageDetectionCache.generate_key(
            image=img,
            detectors=("yolov8",),
            thresholds={"yolov8": 0.5}
        )
        key2 = ImageDetectionCache.generate_key(
            image=img,
            detectors=("yolov8",),
            thresholds={"yolov8": 0.5}
        )
        assert key1 == key2


# =============================================================================
# SECTION 6: GRAPH TESTS
# =============================================================================

class TestSceneGraph:
    """Test scene graph functionality."""

    def test_graph_to_prompt(self):
        """Test graph to prompt conversion."""
        import networkx as nx
        from gom.graph.prompt import graph_to_prompt

        G = nx.DiGraph()
        G.add_node(0, label="person", box=(0, 0, 50, 100))
        G.add_node(1, label="chair", box=(50, 50, 100, 150))
        G.add_edge(0, 1, relation="sitting_on")

        prompt = graph_to_prompt(G)
        assert isinstance(prompt, str)

    def test_graph_to_triples(self):
        """Test graph to triples text conversion."""
        import networkx as nx
        from gom.graph.prompt import graph_to_triples_text

        G = nx.DiGraph()
        G.add_node(0, label="person", box=(0, 0, 50, 100))
        G.add_node(1, label="chair", box=(50, 50, 100, 150))
        G.add_edge(0, 1, relation="sitting_on")

        triples = graph_to_triples_text(G)
        assert isinstance(triples, str)
        assert "person" in triples or "sitting_on" in triples


# =============================================================================
# SECTION 7: DETECTOR BASE CLASS TESTS
# =============================================================================

class TestDetectorBase:
    """Test Detector base class functionality."""

    def test_detector_is_abstract(self):
        """Test that Detector is abstract."""
        from gom.detectors.base import Detector
        with pytest.raises(TypeError):
            Detector(device="cpu", score_threshold=0.5)

    def test_detector_interface(self):
        """Test Detector interface methods exist."""
        from gom.detectors.base import Detector
        assert hasattr(Detector, 'detect')
        assert hasattr(Detector, 'detect_batch')
        assert hasattr(Detector, 'warmup')
        assert hasattr(Detector, 'close')


# =============================================================================
# SECTION 8: SEGMENTATION BASE CLASS TESTS
# =============================================================================

class TestSegmenterBase:
    """Test Segmenter base class functionality."""

    def test_segmenter_config(self):
        """Test SegmenterConfig creation."""
        from gom.segmentation.base import SegmenterConfig
        config = SegmenterConfig(device="cpu", close_holes=True)
        assert config.device == "cpu"
        assert config.close_holes is True

    def test_segmenter_utility_methods(self):
        """Test Segmenter utility methods exist."""
        from gom.segmentation.base import Segmenter
        assert hasattr(Segmenter, 'clamp_box_xyxy')
        assert hasattr(Segmenter, 'bbox_from_mask')
        assert hasattr(Segmenter, 'postprocess_mask')


# =============================================================================
# SECTION 9: API FUNCTIONALITY TESTS (MOCKED)
# =============================================================================

class TestGoMAPI:
    """Test GoM API functionality."""

    def test_gom_class_exists(self):
        """Test GoM class can be imported."""
        from gom.api import GoM
        assert callable(GoM)

    def test_gom_has_expected_methods(self):
        """Test GoM class has expected methods."""
        from gom.api import GoM
        assert hasattr(GoM, 'process')
        assert hasattr(GoM, '__init__')

    def test_create_pipeline_function_exists(self):
        """Test create_pipeline factory function exists."""
        from gom.api import create_pipeline
        assert callable(create_pipeline)

    def test_processing_config_style_presets(self):
        """Test all style presets are valid."""
        from gom.api import ProcessingConfig, GOM_STYLE_PRESETS

        for style_name in GOM_STYLE_PRESETS:
            config = ProcessingConfig.from_style(style_name)
            assert config.style == style_name


# =============================================================================
# SECTION 10: DEPTH ESTIMATOR TESTS (MOCKED)
# =============================================================================

class TestDepthEstimator:
    """Test DepthEstimatorV2 functionality."""

    def test_depth_estimator_creation_no_torch(self):
        """Test depth estimator gracefully handles missing torch."""
        from gom.utils.depth_v2 import DepthEstimatorV2, DepthConfig

        config = DepthConfig(lazy_load=True)
        estimator = DepthEstimatorV2(config=config)
        # Should not crash even without loading model
        assert estimator is not None

    def test_depth_cache_info(self):
        """Test depth cache info method."""
        from gom.utils.depth_v2 import DepthEstimatorV2, DepthConfig

        config = DepthConfig(lazy_load=True)
        estimator = DepthEstimatorV2(config=config)
        info = estimator.get_cache_info()
        assert "cached_maps" in info
        assert "max_size" in info

    def test_depth_clear_cache(self):
        """Test depth cache clearing."""
        from gom.utils.depth_v2 import DepthEstimatorV2, DepthConfig

        config = DepthConfig(lazy_load=True)
        estimator = DepthEstimatorV2(config=config)
        estimator.clear_cache()  # Should not raise


# =============================================================================
# SECTION 11: VISUALIZER TESTS
# =============================================================================

class TestVisualizer:
    """Test Visualizer functionality."""

    def test_visualizer_creation(self):
        """Test Visualizer can be created."""
        from gom.viz.visualizer import Visualizer, VisualizerConfig

        config = VisualizerConfig()
        viz = Visualizer(config=config)
        assert viz is not None

    def test_visualizer_config_fields(self):
        """Test VisualizerConfig has expected fields."""
        from gom.viz.visualizer import VisualizerConfig

        config = VisualizerConfig(
            display_labels=True,
            display_relationships=True,
            show_segmentation=True
        )
        assert config.display_labels is True
        assert config.display_relationships is True


# =============================================================================
# SECTION 12: FUSION TESTS
# =============================================================================

class TestFusion:
    """Test fusion functionality."""

    def test_wbf_fusion_function_exists(self):
        """Test WBF fusion function can be imported."""
        from gom.fusion.wbf import fuse_detections_wbf
        assert callable(fuse_detections_wbf)

    def test_wbf_fusion_empty(self):
        """Test WBF fusion with empty input."""
        from gom.fusion.wbf import fuse_detections_wbf

        result = fuse_detections_wbf(
            detections=[],
            image_size=(100, 100),
            iou_thr=0.5
        )
        assert isinstance(result, list)

    def test_wbf_fusion_with_detections(self):
        """Test WBF fusion with Detection objects."""
        from gom.fusion.wbf import fuse_detections_wbf
        from gom.types import Detection

        detections = [
            Detection(box=(10, 10, 50, 50), label="person", score=0.9, source="yolo"),
            Detection(box=(60, 60, 90, 90), label="car", score=0.8, source="yolo"),
        ]

        result = fuse_detections_wbf(
            detections=detections,
            image_size=(100, 100),
            iou_thr=0.5
        )
        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# SECTION 13: RELATIONS INFERENCE TESTS
# =============================================================================

class TestRelationsInference:
    """Test relation inference functionality."""

    def test_relations_config_defaults(self):
        """Test RelationsConfig default values."""
        from gom.relations.inference import RelationsConfig

        config = RelationsConfig()
        assert config.enabled is True
        assert config.use_geometric_relations is True

    def test_relation_inferencer_class_exists(self):
        """Test RelationInferencer class can be imported."""
        from gom.relations.inference import RelationInferencer
        assert callable(RelationInferencer)

    def test_relation_inferencer_has_infer_method(self):
        """Test RelationInferencer has infer method."""
        from gom.relations.inference import RelationInferencer
        assert hasattr(RelationInferencer, 'infer')


# =============================================================================
# SECTION 14: INTEGRATION SMOKE TESTS
# =============================================================================

class TestIntegrationSmoke:
    """Smoke tests for integration scenarios."""

    def test_full_import_chain(self):
        """Test that full import chain works."""
        # Main API
        from gom import GoM, ProcessingConfig

        # Types
        from gom.types import Detection, Relationship

        # Config
        from gom.config import PreprocessorConfig

        # Detectors
        from gom.detectors.base import Detector

        # Segmentation
        from gom.segmentation.base import Segmenter

        # Relations
        from gom.relations.inference import RelationsConfig

        # Graph
        from gom.graph.scene_graph import SceneGraphBuilder

        # Visualization
        from gom.viz.visualizer import Visualizer

        # Utilities
        from gom.utils.boxes import iou
        from gom.utils.colors import ColorCycler

        # All imports successful
        assert True

    def test_create_detection_list(self):
        """Test creating a list of detections."""
        from gom.types import Detection

        detections = [
            Detection(box=(10, 10, 50, 50), label="person", score=0.9),
            Detection(box=(60, 60, 100, 100), label="car", score=0.8),
            Detection(box=(120, 30, 180, 90), label="dog", score=0.75),
        ]
        assert len(detections) == 3
        assert all(isinstance(d, Detection) for d in detections)

    def test_create_relationship_list(self):
        """Test creating a list of relationships."""
        from gom.types import Relationship

        relationships = [
            Relationship(src_idx=0, tgt_idx=1, relation="left_of"),
            Relationship(src_idx=0, tgt_idx=2, relation="above"),
            Relationship(src_idx=1, tgt_idx=2, relation="near", distance=30.5),
        ]
        assert len(relationships) == 3
        assert all(isinstance(r, Relationship) for r in relationships)

    def test_box_operations_chain(self):
        """Test chaining box operations."""
        from gom.utils.boxes import area, iou, center, center_distance

        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)

        # Chain of operations
        a1 = area(box1)
        a2 = area(box2)
        overlap = iou(box1, box2)
        c1 = center(box1)
        c2 = center(box2)
        dist = center_distance(box1, box2)

        assert a1 == 10000
        assert a2 == 10000
        assert 0 < overlap < 1
        assert c1 == (50, 50)
        assert c2 == (100, 100)
        assert dist > 0


# =============================================================================
# SECTION 15: EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_detection_list(self):
        """Test handling empty detection list."""
        from gom.types import Detection
        detections = []
        assert len(detections) == 0

    def test_zero_area_box(self):
        """Test zero area box handling."""
        from gom.utils.boxes import area
        box = (50, 50, 50, 50)
        assert area(box) == 0

    def test_negative_coordinates_box(self):
        """Test box with negative coordinates."""
        from gom.utils.boxes import clamp_xyxy
        box = (-10, -20, 50, 60)
        clamped = clamp_xyxy(box, W=100, H=100)
        assert clamped[0] >= 0
        assert clamped[1] >= 0

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes."""
        from gom.utils.boxes import iou
        box = (10, 20, 30, 40)
        assert iou(box, box) == 1.0

    def test_cache_stats(self):
        """Test cache statistics."""
        from gom.utils.cache_advanced import CacheStats
        stats = CacheStats(hits=10, misses=5, evictions=2)
        assert stats.hit_rate == 10 / 15


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
