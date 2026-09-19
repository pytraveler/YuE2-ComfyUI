"""Make yue2_comfy importable without ComfyUI and without installing anything.

The package is registered by hand rather than imported. The repository root
cannot go on sys.path -- it holds ComfyUI's entry point __init__.py, which only
makes sense with a parent package, and any tool that walks sys.path tries to
import it and fails. Handing the package a __path__ lets every submodule import
normally while nothing above it is ever touched.

Nothing here imports ComfyUI.
"""

import pathlib
import sys
import types

import pytest

PACKAGE = "yue2_comfy"
ROOT = pathlib.Path(__file__).resolve().parent.parent

if PACKAGE not in sys.modules:
    module = types.ModuleType(PACKAGE)
    module.__path__ = [str(ROOT / PACKAGE)]
    sys.modules[PACKAGE] = module


@pytest.fixture(autouse=True)
def songs_held_in_memory(monkeypatch):
    """Every test starts with an empty song memory that never touches the disk.

    Outside ComfyUI the pack's user folder is inside the pack itself, so a test
    that sang through a node would otherwise leave song files in the working
    tree. A test about the disk makes its own store in a temporary folder.
    """
    from yue2_comfy import songs

    monkeypatch.setattr(songs, "_store", songs.Store(None))
