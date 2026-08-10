import numpy as np

from gom.pipeline.preprocessor import ImageGraphPreprocessor


def _mask(array):
    return {"segmentation": np.asarray(array, dtype=bool)}


def test_same_class_nested_boxes_with_same_mask_are_deduplicated():
    pipeline = object.__new__(ImageGraphPreprocessor)
    mask = np.ones((8, 8), dtype=bool)
    result = pipeline._remove_overlapping_objects(
        boxes=[[0, 0, 10, 10], [2, 2, 7, 7]],
        labels=["dog", "dog"],
        scores=[0.9, 0.8],
        masks=[_mask(mask), _mask(mask)],
        iou_threshold=0.5,
        same_class_mask_iou_threshold=0.8,
        containment_threshold=0.9,
        cross_class=False,
    )
    assert result[-1] == [0]


def test_contained_same_class_boxes_are_deduplicated_even_with_partial_masks():
    pipeline = object.__new__(ImageGraphPreprocessor)
    outer = np.zeros((8, 8), dtype=bool)
    inner = np.zeros((8, 8), dtype=bool)
    outer[:4, :] = True
    inner[4:, :] = True
    result = pipeline._remove_overlapping_objects(
        boxes=[[0, 0, 10, 10], [2, 2, 7, 7]],
        labels=["person", "person"],
        scores=[0.9, 0.8],
        masks=[_mask(outer), _mask(inner)],
        iou_threshold=0.5,
        same_class_mask_iou_threshold=0.8,
        containment_threshold=0.9,
        cross_class=False,
    )
    assert result[-1] == [0]


def test_distinct_same_class_instances_are_preserved():
    pipeline = object.__new__(ImageGraphPreprocessor)
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[:, :3] = True
    right[:, 5:] = True
    result = pipeline._remove_overlapping_objects(
        boxes=[[0, 0, 5, 8], [3, 0, 8, 8]],
        labels=["person", "person"],
        scores=[0.9, 0.8],
        masks=[_mask(left), _mask(right)],
        iou_threshold=0.5,
        same_class_mask_iou_threshold=0.8,
        containment_threshold=0.9,
        cross_class=False,
    )
    assert result[-1] == [0, 1]
