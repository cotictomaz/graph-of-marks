"""Guard against the wheel silently dropping a subpackage.

`pyproject.toml` uses `[tool.setuptools.packages.find]` (classic ``find_packages``),
which only discovers directories that contain an ``__init__.py``. A subpackage added
without one is omitted from the wheel with no error -- that shipped a broken
``graph-of-mark`` 1.0.1/1.1.0 (missing gom.pipeline/gom.viz/gom.relations/gom.vqa/...).
This test asserts every source directory holding a ``.py`` file is actually packaged.

Torch-free: runs on the host, no models loaded.
"""
from pathlib import Path

from setuptools import find_packages

SRC = Path(__file__).resolve().parent.parent / "src"


def test_every_gom_source_dir_is_a_discoverable_package():
    shipped = set(find_packages(where=str(SRC), include=["gom*"], exclude=["tests*", "scripts*"]))

    missing = []
    for py in (SRC / "gom").rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        pkg = ".".join(py.parent.relative_to(SRC).parts)
        if pkg not in shipped:
            missing.append(f"{pkg} (holds {py.name}) has no __init__.py -> dropped from wheel")

    assert not missing, "Subpackages missing from the built wheel:\n  " + "\n  ".join(sorted(set(missing)))
