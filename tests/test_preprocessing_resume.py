from types import SimpleNamespace

from gom.pipeline.preprocessor import ImageGraphPreprocessor


def test_resume_requires_every_render_variant(tmp_path):
    preprocessor = ImageGraphPreprocessor.__new__(ImageGraphPreprocessor)
    preprocessor.cfg = SimpleNamespace(
        output_folder=str(tmp_path),
        skip_graph=False,
        skip_prompt=False,
        skip_visualization=False,
        output_format="jpg",
        render_variants={"small": {}, "large": {}},
    )
    name = "image_q1"
    (tmp_path / f"{name}_graph.json").write_text("{}")
    (tmp_path / f"{name}_graph_triples.txt").write_text("")
    (tmp_path / f"{name}_render_variants.json").write_text("{}")
    for variant in ("small", "large"):
        directory = tmp_path / "renders" / variant
        directory.mkdir(parents=True)
        (directory / f"{name}_output.jpg").write_bytes(b"jpg")

    assert preprocessor._outputs_complete(name)
    (tmp_path / "renders" / "large" / f"{name}_output.jpg").unlink()
    assert not preprocessor._outputs_complete(name)
