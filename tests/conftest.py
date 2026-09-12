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

PACKAGE = "yue2_comfy"
ROOT = pathlib.Path(__file__).resolve().parent.parent

if PACKAGE not in sys.modules:
    module = types.ModuleType(PACKAGE)
    module.__path__ = [str(ROOT / PACKAGE)]
    sys.modules[PACKAGE] = module
