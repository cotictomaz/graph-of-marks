import pytest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gom.viz.visualizer as visualizer_module
from gom.viz.visualizer import Visualizer, VisualizerConfig


def test_outline_render_preserves_pixels_and_dimensions(tmp_path):
    image = Image.new("RGB", (73, 41), (210, 205, 200))
    mask = np.zeros((41, 73), dtype=bool)
    mask[8:33, 15:58] = True
    output = tmp_path / "outline.png"
    cfg = VisualizerConfig(
        display_labels=False,
        display_relationships=False,
        show_segmentation=True,
        fill_segmentation=False,
        show_bboxes=False,
        use_vectorized_masks=True,
    )

    Visualizer(cfg).draw(
        image,
        boxes=[[15, 8, 58, 33]],
        labels=["object"],
        scores=[1.0],
        relationships=[],
        masks=[{"segmentation": mask}],
        save_path=str(output),
        dpi=160,
    )

    rendered = Image.open(output).convert("RGB")
    assert rendered.size == image.size
    # The center is inside the mask but not on its contour, so outline-only
    # rendering must leave the source image unchanged there.
    assert rendered.getpixel((36, 20)) == image.getpixel((36, 20))


def test_filled_render_applies_fill_alpha_exactly_once(tmp_path):
    import matplotlib.colors as mcolors

    image = Image.new("RGB", (73, 41), (200, 200, 200))
    mask = np.zeros((41, 73), dtype=bool)
    mask[8:33, 15:58] = True
    output = tmp_path / "filled.png"
    cfg = VisualizerConfig(
        display_labels=False,
        display_relationships=False,
        show_segmentation=True,
        fill_segmentation=True,
        seg_fill_alpha=0.25,
        show_bboxes=False,
        use_vectorized_masks=True,
    )
    viz = Visualizer(cfg)
    color = mcolors.to_rgb(viz._assign_colors(["object"])[0])

    viz.draw(
        image,
        boxes=[[15, 8, 58, 33]],
        labels=["object"],
        scores=[1.0],
        relationships=[],
        masks=[{"segmentation": mask}],
        save_path=str(output),
        dpi=160,
    )

    rendered = Image.open(output).convert("RGB")
    # A double-applied fill would land at 0.5625*bg + 0.4375*color instead.
    expected = tuple(0.75 * 200 + 0.25 * 255 * channel for channel in color)
    got = rendered.getpixel((36, 20))
    assert all(abs(g - e) <= 2 for g, e in zip(got, expected)), (got, expected)


def test_draw_segmentation_fill_flag_controls_fill_artists():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    viz = Visualizer(
        VisualizerConfig(show_segmentation=True, fill_segmentation=True, seg_fill_alpha=0.25)
    )
    fig, ax = plt.subplots()
    viz._draw_segmentation(ax, mask, "#ff0000", 2.0, fill=False)
    assert len(ax.patches) + len(ax.images) == 0
    viz._draw_segmentation(ax, mask, "#ff0000", 2.0, fill=True)
    assert len(ax.patches) + len(ax.images) > 0
    plt.close(fig)


def test_unsafe_legacy_relation_is_not_rendered():
    viz = Visualizer(VisualizerConfig())
    assert viz._preprocess_relations(
        [{"src_idx": 0, "tgt_idx": 1, "relation": "cnet_synonym"}],
        [[0, 0, 2, 2], [3, 3, 5, 5]],
    ) == []


def test_effective_font_sizes_respect_configurable_floors():
    cfg = VisualizerConfig(
        obj_fontsize_inside=8,
        obj_fontsize_outside=8,
        rel_fontsize=7,
        auto_scale_styles=True,
        style_ref_px=1000,
        style_scale_min=0.5,
        obj_fontsize_inside_min=4,
        obj_fontsize_outside_min=4,
        rel_fontsize_min=4,
    )

    effective = Visualizer(cfg).effective_style((500, 400))

    assert effective["obj_fontsize_inside"] == 4
    assert effective["obj_fontsize_outside"] == 4
    assert effective["rel_fontsize"] == 4


def test_fixed_font_profile_is_exact():
    cfg = VisualizerConfig(
        obj_fontsize_inside=9,
        obj_fontsize_outside=10,
        rel_fontsize=8,
        auto_scale_styles=False,
    )

    effective = Visualizer(cfg).effective_style((473, 500))

    assert effective["obj_fontsize_inside"] == 9
    assert effective["obj_fontsize_outside"] == 10
    assert effective["rel_fontsize"] == 8


def test_overlap_fallback_separates_relation_and_object_labels(monkeypatch):
    monkeypatch.setattr(visualizer_module, "adjust_text", None)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    fixed = ax.text(50, 50, "person_1", fontsize=10, bbox={"facecolor": "red"})
    movable = ax.text(
        50, 50, "Touching Above", fontsize=8, bbox={"facecolor": "white"}
    )
    visualizer = Visualizer(VisualizerConfig(micro_push_iters=12))
    visualizer._resolve_overlaps(ax, [movable], [(50, 50)], [fixed], [])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not movable.get_window_extent(renderer).overlaps(
        fixed.get_window_extent(renderer)
    )
    plt.close(fig)


def _dense_scene():
    """A deliberately hostile scene: adjacent small boxes in a tight grid."""
    boxes, labels, rels = [], [], []
    for row in range(3):
        for col in range(4):
            x, y = 20 + col * 70, 20 + row * 60
            boxes.append([x, y, x + 46, y + 38])
            labels.append(f"object_{row * 4 + col + 1}")
    for i in range(8):
        rels.append({"src_idx": i, "tgt_idx": i + 1, "relation": "left_of"})
    return boxes, labels, rels


def test_deterministic_placement_never_overlaps_labels():
    boxes, labels, rels = _dense_scene()
    image = Image.new("RGB", (340, 230), (128, 128, 128))
    cfg = VisualizerConfig(
        display_labels=True,
        display_relationships=True,
        display_relation_labels=True,
        deterministic_label_placement=True,
        show_segmentation=False,
        show_bboxes=True,
        fill_segmentation=False,
    )

    fig, _ = Visualizer(cfg).draw(
        image,
        boxes=boxes,
        labels=labels,
        scores=[0.9] * len(boxes),
        relationships=rels,
        masks=None,
    )
    try:
        assert fig._gom_label_overlap_count == 0, (
            f"{fig._gom_label_overlap_count} overlapping label pairs on a dense scene"
        )
        # Placement must actually have happened, not been skipped.
        assert len(fig._gom_label_boxes) >= len(boxes)
    finally:
        plt.close(fig)


def test_deterministic_placement_keeps_label_bound_to_its_own_object():
    """Guards the label<->object binding: each label must sit nearest its own box."""
    image = Image.new("RGB", (400, 200), (90, 90, 90))
    boxes = [[10, 60, 90, 140], [150, 60, 230, 140], [300, 60, 380, 140]]
    labels = ["left_1", "middle_1", "right_1"]
    cfg = VisualizerConfig(
        display_labels=True,
        display_relationships=False,
        deterministic_label_placement=True,
        show_segmentation=False,
        show_bboxes=True,
    )
    fig, ax = Visualizer(cfg).draw(
        image,
        boxes=boxes,
        labels=labels,
        scores=[0.9] * 3,
        relationships=[],
        masks=None,
    )
    try:
        by_text = {t.get_text(): t.get_position()[0] for t in ax.texts}
        assert by_text["left_1"] < by_text["middle_1"] < by_text["right_1"]
        for label, box in zip(labels, boxes):
            centre_x = (box[0] + box[2]) / 2
            assert abs(by_text[label] - centre_x) < 120, f"{label} drifted off its object"
    finally:
        plt.close(fig)


def test_arrowheads_stay_visible_and_shafts_stay_drawable():
    """R1: no label box may cover an arrow head, AND the shaft must be visible.

    gom_v4 kept the head clear by clipping the arrow to the target box *and* by
    reserving the head in the label registry. The clip was redundant -- the
    registry does that work -- and it deleted the shaft on 52.9% of arrows
    run-wide. The head is therefore no longer required to end outside its target;
    what is required is that a reader can see which way the arrow points, which
    needs a shaft.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.transforms import Bbox

    def overlap_area(a, b):  # mirrors _registry_overlap: touching is fine
        return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(
            0.0, min(a.y1, b.y1) - max(a.y0, b.y0)
        )

    boxes, labels, rels = _dense_scene()
    image = Image.new("RGB", (340, 230), (128, 128, 128))
    viz = Visualizer(
        VisualizerConfig(
            display_labels=True,
            display_relationships=True,
            display_relation_labels=True,
            deterministic_label_placement=True,
            show_segmentation=False,
            show_bboxes=True,
            fill_segmentation=False,
        )
    )
    fig, ax = viz.draw(
        image, boxes=boxes, labels=labels, scores=[0.9] * len(boxes),
        relationships=rels, masks=None,
    )
    try:
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == len(rels), f"{len(arrows)} arrows for {len(rels)} relations"
        label_bbs = [Bbox.from_extents(*e) for e in fig._gom_label_boxes]
        for arrow in arrows:
            head = viz._arrow_head_bbox_px(arrow)
            for bb in label_bbs:
                assert overlap_area(head, bb) == 0.0, "a label box covers an arrow head"
            assert viz._arrow_shaft_length_px(arrow) >= 25.0, (
                "arrow shaft is too short to read direction from"
            )
        assert fig._gom_label_overlap_count == 0  # R4 unchanged
    finally:
        plt.close(fig)


def test_relation_labels_stay_bound_to_their_arc():
    """R2: every relation label sits on its arc or carries a leader back to it."""
    import matplotlib.pyplot as _plt

    boxes, labels, rels = _dense_scene()
    image = Image.new("RGB", (340, 230), (128, 128, 128))
    viz = Visualizer(
        VisualizerConfig(
            display_labels=True,
            display_relationships=True,
            display_relation_labels=True,
            deterministic_label_placement=True,
            show_segmentation=False,
            show_bboxes=True,
            fill_segmentation=False,
        )
    )
    fig, ax = viz.draw(
        image, boxes=boxes, labels=labels, scores=[0.9] * len(boxes),
        relationships=rels, masks=None,
    )
    try:
        seated = [t for t in ax.texts if hasattr(t, "_gom_arc_anchor")]
        assert seated, "no relation label was seated"
        leader_ends = {
            tuple(round(v, 3) for v in a.xy)
            for a in ax.texts
            if a.get_text() == "" and a.arrow_patch is not None
        }
        for t in seated:
            anchor = np.asarray(t._gom_arc_anchor)
            pos = np.asarray(ax.transData.transform(t.get_position()))
            if float(np.hypot(*(pos - anchor))) <= 30.0:
                continue
            xy = tuple(
                round(v, 3)
                for v in ax.transData.inverted().transform(anchor)
            )
            assert xy in leader_ends, "far relation label has no leader to its arc"
    finally:
        _plt.close(fig)


def test_mask_outline_ignores_holes_and_specks():
    """Only object boundaries get stroked, not hole boundaries or fragments.

    cv2.RETR_CCOMP returns interior hole boundaries as well as outer ones, and the
    draw loop had no area floor, so a mask with a hole and a few specks was stroked
    as several closed curves -- which is what made a flock of sheep render as
    scribbles. The legacy monolith used RETR_EXTERNAL + largest contour
    (all_in_one_gom.py:1983); the refactor lost it.
    """
    pytest.importorskip("cv2")  # the no-cv2 fallback path draws differently
    import matplotlib.pyplot as _plt

    # one solid blob with a hole in it, plus two specks elsewhere
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 40:160] = True
    mask[90:110, 90:110] = False      # hole -> an interior contour under RETR_CCOMP
    mask[5:9, 5:9] = True             # speck
    mask[190:194, 190:194] = True     # speck

    viz = Visualizer(
        VisualizerConfig(
            display_labels=False, display_relationships=False,
            show_segmentation=True, fill_segmentation=False, show_bboxes=False,
            use_vectorized_masks=False,
        )
    )
    fig, ax = viz.draw(
        Image.new("RGB", (200, 200), (120, 120, 120)),
        boxes=[[40, 40, 160, 160]], labels=["thing"], scores=[0.9],
        relationships=[], masks=[{"segmentation": mask}],
    )
    try:
        # exactly one stroked outline: the object boundary
        assert viz._contours_drawn == [1], (
            f"expected 1 contour stroked, got {viz._contours_drawn} "
            "(hole boundaries or specks are being drawn)"
        )
    finally:
        _plt.close(fig)


def test_mask_outline_keeps_a_genuinely_two_part_object():
    """An area floor, not largest-only: a sheep split by a fence post keeps both
    halves. Largest-only would silently discard the smaller one."""
    pytest.importorskip("cv2")  # the no-cv2 fallback path draws differently
    import matplotlib.pyplot as _plt

    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 30:90] = True        # left half
    mask[50:150, 110:170] = True      # right half, comparable size
    viz = Visualizer(
        VisualizerConfig(
            display_labels=False, display_relationships=False,
            show_segmentation=True, fill_segmentation=False, show_bboxes=False,
            use_vectorized_masks=False,
        )
    )
    fig, ax = viz.draw(
        Image.new("RGB", (200, 200), (120, 120, 120)),
        boxes=[[30, 50, 170, 150]], labels=["sheep"], scores=[0.9],
        relationships=[], masks=[{"segmentation": mask}],
    )
    try:
        assert viz._contours_drawn == [2], (
            f"expected both parts stroked, got {viz._contours_drawn}"
        )
    finally:
        _plt.close(fig)
