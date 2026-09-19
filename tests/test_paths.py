"""Where the pack keeps what is not a model, with the pack imported the way ComfyUI imports it.

Every other test imports the package as ``yue2_comfy``. ComfyUI does not: it
imports a custom node under the node's full path, dots replaced with "_x_", so
the package is that path and its modules are named below it. A folder derived
from the module name was right in every test and wrong in every ComfyUI, and
this loads ``paths`` the ComfyUI way to keep it right.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_user_folder_is_inside_comfyuis_user_directory(tmp_path, monkeypatch):
    user = tmp_path / "ComfyUI" / "user"
    node = str(tmp_path / "ComfyUI" / "custom_nodes" / "YuE2-ComfyUI").replace(".", "_x_")
    top = types.ModuleType(node)
    top.__path__ = [str(ROOT)]
    package = types.ModuleType(node + ".yue2_comfy")
    package.__path__ = [str(ROOT / "yue2_comfy")]
    monkeypatch.setitem(sys.modules, node, top)
    monkeypatch.setitem(sys.modules, node + ".yue2_comfy", package)
    monkeypatch.setitem(sys.modules, "folder_paths",
                        types.SimpleNamespace(get_user_directory=lambda: str(user)))
    try:
        paths = importlib.import_module(node + ".yue2_comfy.paths")
        assert paths.user_dir() == os.path.join(str(user), "yue2_comfy")
    finally:
        for name in [name for name in sys.modules if name.startswith(node + ".")]:
            del sys.modules[name]


def test_outside_comfyui_the_user_folder_stays_inside_the_pack(monkeypatch):
    """Without folder_paths there is no ComfyUI user directory to ask."""
    from yue2_comfy import paths

    monkeypatch.setitem(sys.modules, "folder_paths", None)
    assert paths.user_dir() == str(ROOT / "yue2_comfy" / "_user" / "yue2_comfy")
