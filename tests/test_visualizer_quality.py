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
