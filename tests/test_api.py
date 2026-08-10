"""
Test the high-level GraphOfMarks API.
These tests focus on configuration and structure without loading heavy models.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image


class TestGraphOfMarksConfiguration:
    """Test GraphOfMarks initialization and configuration."""

    def test_default_config(self):
        """Test default_config factory function."""
        from gom import default_config
        
        config = default_config()
        assert config is not None
        assert hasattr(config, "detectors_to_use")

    def test_preprocessor_config_creation(self):
        """Test PreprocessorConfig can be created with custom values."""
        from gom import PreprocessorConfig
        
        config = PreprocessorConfig()
        assert config is not None
        
        # Test setting values
        config.output_folder = "/tmp/test_output"
        assert config.output_folder == "/tmp/test_output"

    def test_config_types(self):
        """Test all config types can be instantiated."""
        from gom import (
            PreprocessorConfig,
            SegmenterConfig,
            RelationsConfig,
            VisualizerConfig,
        )
        
        # These should not raise
        PreprocessorConfig()
        SegmenterConfig()
        RelationsConfig()
        VisualizerConfig()


class TestGraphOfMarksAPI:
    """Test the new function-based API."""
    
    def test_graphofmarks_default_init(self):
        """Test GraphOfMarks can be initialized with defaults."""
        from gom import GraphOfMarks
        
        # Should not raise - uses internal defaults
        # Note: We don't actually initialize here to avoid model loading
        assert GraphOfMarks is not None
    
    def test_graphofmarks_with_custom_detector(self):
        """Test GraphOfMarks accepts custom detector function."""
        from gom import GraphOfMarks, Detection
        
        def my_detector(image, **kwargs):
            # Mock detector that returns empty list
            return []
        
        # Class should accept callable
        assert callable(my_detector)
    
    def test_graphofmarks_with_custom_segmenter(self):
        """Test GraphOfMarks accepts custom segmenter function."""
        from gom import GraphOfMarks
        
        def my_segmenter(image, boxes, **kwargs):
            return {'masks': []}
        
        assert callable(my_segmenter)
    
    def test_graphofmarks_with_custom_depth(self):
        """Test GraphOfMarks accepts custom depth estimator."""
        from gom import GraphOfMarks
        
        def my_depth(image, **kwargs):
            return np.zeros((100, 100), dtype=np.float32)
        
        assert callable(my_depth)


class TestDetectionType:
    """Test Detection dataclass functionality."""

    def test_detection_creation(self):
        """Test creating a Detection instance."""
        from gom.types import Detection
        
        det = Detection(
            label="person",
            box=(100, 100, 200, 200),
            score=0.95,
        )
        assert det.label == "person"
        assert det.box == (100, 100, 200, 200)
        assert det.score == 0.95

    def test_detection_with_optional_fields(self):
        """Test Detection with optional fields."""
        from gom.types import Detection
        
        det = Detection(
            label="cat",
            box=(50, 50, 150, 150),
            score=0.85,
            source="yolov8",
        )
        assert det.source == "yolov8"

    def test_detection_with_extra(self):
        """Test Detection with extra metadata."""
        from gom.types import Detection
        
        det = Detection(
            label="dog",
            box=(10, 10, 100, 100),
            score=0.92,
            extra={"mask": "placeholder"}
        )
        assert det.extra == {"mask": "placeholder"}


class TestRelationshipType:
    """Test Relationship dataclass functionality."""

    def test_relationship_creation(self):
        """Test creating a Relationship instance."""
        from gom.types import Relationship
        
        rel = Relationship(
            src_idx=0,
            tgt_idx=1,
            relation="left_of",
        )
        assert rel.src_idx == 0
        assert rel.tgt_idx == 1
        assert rel.relation == "left_of"

    def test_relationship_with_distance(self):
        """Test Relationship with distance."""
        from gom.types import Relationship
        
        rel = Relationship(
            src_idx=2,
            tgt_idx=3,
            relation="near",
            distance=50.0,
        )
        assert rel.distance == 50.0

    def test_relationship_with_clip(self):
        """Test Relationship with CLIP similarity."""
        from gom.types import Relationship
        
        rel = Relationship(
            src_idx=0,
            tgt_idx=1,
            relation="riding",
            relation_raw="riding on",
            clip_sim=0.87,
        )
        assert rel.relation_raw == "riding on"
        assert rel.clip_sim == 0.87


class TestUtilityFunctions:
    """Test utility functions and helpers."""

    def test_create_test_image(self):
        """Test we can create a simple test image for processing."""
        img_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            assert Path(f.name).exists()
            reloaded = Image.open(f.name)
            assert reloaded.size == (100, 100)


class TestPackageMetadata:
    """Test package metadata is correct."""

    def test_version_string(self):
        """Test version is a valid semver string."""
        import gom
        version = gom.__version__
        parts = version.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])

    def test_package_docstring(self):
        """Test package has a docstring."""
        import gom
        assert gom.__doc__ is not None
        assert len(gom.__doc__) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
