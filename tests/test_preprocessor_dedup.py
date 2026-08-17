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


def test_part_of_object_fragment_is_removed_but_distinct_instances_survive():
    """An elephant's leg detected as a second 'elephant' must go; two adjacent
    elephants must both stay (their masks are disjoint)."""
    import numpy as np
    from gom.pipeline.preprocessor import ImageGraphPreprocessor

    pre = ImageGraphPreprocessor.__new__(ImageGraphPreprocessor)

    body = np.zeros((100, 100), dtype=bool)
    body[10:90, 10:60] = True
    leg = np.zeros((100, 100), dtype=bool)
    leg[60:88, 20:32] = True  # inside the body mask
    boxes = [[10, 10, 60, 90], [20, 60, 32, 88]]
    kept = pre._remove_overlapping_objects(
        list(boxes), ["elephant_1", "elephant_2"], [0.9, 0.95],
        masks=[{"segmentation": body}, {"segmentation": leg}],
        same_class_fragment_containment=0.85,
    )
    assert kept[1] == ["elephant_1"], "the leg fragment should be dropped"

    other = np.zeros((100, 100), dtype=bool)
    other[10:90, 62:99] = True  # adjacent, disjoint mask
    kept2 = pre._remove_overlapping_objects(
        [[10, 10, 60, 90], [62, 10, 99, 90]], ["elephant_1", "elephant_2"], [0.9, 0.9],
        masks=[{"segmentation": body}, {"segmentation": other}],
        same_class_fragment_containment=0.85,
    )
    assert len(kept2[1]) == 2, "two distinct elephants must both survive"
