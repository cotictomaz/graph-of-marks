"""
Test the demo script functionality.
Verifies the demo can run and the simplified API works correctly.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image


class TestDemoFunctionality:
    """Tests for demo script functionality."""

    def test_import_graph_of_marks(self):
        """Test basic import works."""
        from gom import GraphOfMarks, __version__
        assert GraphOfMarks is not None
        assert __version__ is not None

    def test_create_test_image(self):
        """Test creating a test image similar to demo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_image.jpg"
            
            # Create test image like in demo
            img = np.ones((480, 640, 3), dtype=np.uint8) * 200
            img[100:200, 100:250] = [255, 0, 0]  # Red rectangle
            img[150:300, 300:450] = [0, 255, 0]  # Green rectangle
            img[280:400, 150:300] = [0, 0, 255]  # Blue rectangle
            
            Image.fromarray(img).save(test_path)
            assert test_path.exists()
            
            # Verify image
            reloaded = Image.open(test_path)
            assert reloaded.size == (640, 480)

    def test_custom_detector_with_yolo_example(self):
        """Test custom detector wrapping YOLO."""
        from gom import Detection
        
        # Example: Wrap YOLOv8 into a custom detector function
        def yolo_detector(image, conf_threshold=0.5, **kwargs):
            """
            Custom detector using YOLOv8.
            
            Usage:
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')
                
                def yolo_detector(image, **kwargs):
                    results = model(image, verbose=False)
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            detections.append(Detection(
                                box=tuple(box.xyxy[0].tolist()),
                                label=model.names[int(box.cls)],
                                score=float(box.conf),
                                source='yolov8'
                            ))
                    return detections
            """
            # For testing, return empty list (no actual model loaded)
            return []
        
        # Verify signature is correct
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = yolo_detector(test_image)
        assert isinstance(detections, list)

    def test_custom_segmenter_with_sam_example(self):
        """Test custom segmenter wrapping SAM."""
        
        # Example: Wrap SAM into a custom segmenter function
        def sam_segmenter(image, boxes, **kwargs):
            """
            Custom segmenter using Segment Anything Model.
            
            Usage:
                from segment_anything import sam_model_registry, SamPredictor
                
                sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
                predictor = SamPredictor(sam)
                
                def sam_segmenter(image, boxes, **kwargs):
                    predictor.set_image(image)
                    masks = []
                    for box in boxes:
                        mask, _, _ = predictor.predict(box=np.array(box))
                        masks.append(mask[0])  # Take first mask
                    return {'masks': masks}
            """
            # For testing, return empty masks matching box count
            return {'masks': [np.zeros((100, 100), dtype=np.uint8) for _ in boxes]}
        
        # Verify signature
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]
        result = sam_segmenter(test_image, boxes)
        
        assert 'masks' in result
        assert len(result['masks']) == len(boxes)

    def test_custom_depth_with_depth_anything_example(self):
        """Test custom depth estimator wrapping Depth Anything."""
        
        # Example: Wrap Depth Anything into a custom depth function
        def depth_anything_estimator(image, **kwargs):
            """
            Custom depth estimator using Depth Anything V2.
            
            Usage:
                from depth_anything_v2.dpt import DepthAnythingV2
                
                model = DepthAnythingV2(encoder='vitl')
                model.load_state_dict(torch.load('depth_anything_v2_vitl.pth'))
                
                def depth_anything_estimator(image, **kwargs):
                    depth = model.infer_image(image)
                    # Normalize to [0, 1]
                    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    return depth.astype(np.float32)
            """
            # For testing, return normalized depth map
            h, w = image.shape[:2]
            # Simulate depth: closer objects (smaller y) are closer
            y_coords = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
            depth_map = y_coords.astype(np.float32) / h
            return depth_map
        
        # Verify signature
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = depth_anything_estimator(test_image)
        
        assert depth.shape == (100, 100)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_graphofmarks_api_signature(self):
        """Test GraphOfMarks accepts the new API parameters."""
        from gom import GraphOfMarks
        import inspect

        # Check __init__ signature accepts the new parameters
        sig = inspect.signature(GraphOfMarks.__init__)
        params = list(sig.parameters.keys())

        # New API parameters (refactored to use custom functions)
        assert 'detect_fn' in params
        assert 'segment_fn' in params
        assert 'depth_fn' in params

        # Old parameters should NOT be present
        assert 'detectors' not in params
        assert 'sam_version' not in params

    def test_vqa_models_no_wrappers(self):
        """Test that VLM wrappers have been removed."""
        from gom.vqa import models
        
        # These should NOT exist anymore
        assert not hasattr(models, 'VLLMWrapper')
        assert not hasattr(models, 'HFVLModel')
        assert not hasattr(models, 'OllamaWrapper')
        
        # Utility function should still exist
        assert hasattr(models, 'download_repo_with_bar')


class TestAPIDocumentation:
    """Test that API documentation is correct."""

    def test_graphofmarks_has_docstring(self):
        """Test GraphOfMarks has proper documentation."""
        from gom import GraphOfMarks
        
        assert GraphOfMarks.__doc__ is not None
        assert 'detector' in GraphOfMarks.__doc__
        assert 'segmenter' in GraphOfMarks.__doc__
        assert 'custom' in GraphOfMarks.__doc__.lower()

    def test_create_pipeline_factory(self):
        """Test create_pipeline factory function exists."""
        from gom.api import create_pipeline
        
        assert callable(create_pipeline)
        assert create_pipeline.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
