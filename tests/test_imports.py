"""
Test that all public exports from the gom package can be imported.
These tests verify the package structure without loading heavy models.
"""
import pytest


class TestPackageImports:
    """Test all top-level imports from the gom package."""

    def test_import_gom(self):
        """Test basic package import."""
        import gom
        assert hasattr(gom, "__version__")
        # Version is dynamic, just check it's a valid version string
        assert isinstance(gom.__version__, str)
        assert len(gom.__version__.split(".")) >= 2

    def test_import_graphofmarks(self):
        """Test GraphOfMarks high-level API import."""
        from gom import GraphOfMarks
        assert GraphOfMarks is not None

    def test_import_imagegraphpreprocessor(self):
        """Test ImageGraphPreprocessor import."""
        from gom import ImageGraphPreprocessor
        assert ImageGraphPreprocessor is not None

    def test_import_types(self):
        """Test type imports."""
        from gom import Detection, Relationship, Box, MaskDict
        assert Detection is not None
        assert Relationship is not None
        assert Box is not None
        assert MaskDict is not None

    def test_import_config(self):
        """Test configuration imports."""
        from gom import (
            PreprocessorConfig,
            SegmenterConfig,
            RelationsConfig,
            VisualizerConfig,
            default_config,
        )
        assert PreprocessorConfig is not None
        assert SegmenterConfig is not None
        assert RelationsConfig is not None
        assert VisualizerConfig is not None
        assert callable(default_config)

    def test_all_exports(self):
        """Test that __all__ matches actual exports."""
        import gom
        expected_exports = [
            "GraphOfMarks",
            "ImageGraphPreprocessor",
            "Detection",
            "Relationship",
            "Box",
            "MaskDict",
            "PreprocessorConfig",
            "SegmenterConfig",
            "RelationsConfig",
            "VisualizerConfig",
            "default_config",
        ]
        for export in expected_exports:
            assert export in gom.__all__, f"{export} not in __all__"
            assert hasattr(gom, export), f"{export} not accessible from gom"


class TestSubmoduleImports:
    """Test submodule imports."""

    def test_import_config_module(self):
        """Test config submodule."""
        from gom.config import PreprocessorConfig
        assert PreprocessorConfig is not None

    def test_import_types_module(self):
        """Test types submodule."""
        from gom.types import Detection, Relationship
        assert Detection is not None
        assert Relationship is not None

    def test_import_api_module(self):
        """Test API module."""
        from gom.api import GraphOfMarks
        assert GraphOfMarks is not None

    def test_import_pipeline(self):
        """Test pipeline submodule."""
        from gom.pipeline.preprocessor import ImageGraphPreprocessor
        assert ImageGraphPreprocessor is not None


class TestTypeStructures:
    """Test that types have expected structure."""

    def test_detection_fields(self):
        """Test Detection dataclass has expected fields."""
        from gom.types import Detection
        import dataclasses
        
        assert dataclasses.is_dataclass(Detection)
        field_names = [f.name for f in dataclasses.fields(Detection)]
        # Core fields that should exist
        assert "label" in field_names
        assert "box" in field_names
        assert "score" in field_names

    def test_relationship_fields(self):
        """Test Relationship dataclass has expected fields."""
        from gom.types import Relationship
        import dataclasses
        
        assert dataclasses.is_dataclass(Relationship)
        field_names = [f.name for f in dataclasses.fields(Relationship)]
        # Actual field names per types.py
        assert "src_idx" in field_names
        assert "tgt_idx" in field_names
        assert "relation" in field_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
