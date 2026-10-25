"""``paths.shown``: a path as a browser on another machine may be told it."""

from __future__ import annotations

import os
import types

import pytest

from yue2_comfy import paths


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    base = tmp_path / "ComfyUI"
    (base / "models" / "LLM").mkdir(parents=True)
    monkeypatch.setattr(paths, "_folder_paths", lambda: types.SimpleNamespace(base_path=str(base)))
    return base


def test_inside_comfyui_the_part_below_its_folder_is_given(comfy):
    assert paths.shown(str(comfy / "models" / "LLM" / "x.gguf")) == "models/LLM/x.gguf"
    assert paths.shown(str(comfy / "user" / "yue2_comfy" / "writers.json")) == "user/yue2_comfy/writers.json"


def test_anywhere_else_only_the_name(comfy, tmp_path):
    assert paths.shown(str(tmp_path / "elsewhere" / "x.gguf")) == "x.gguf"
    assert paths.shown(str(tmp_path / "ComfyUI-other" / "x.gguf")) == "x.gguf"
    assert paths.shown("\\\\host\\share\\models\\x.gguf") == "x.gguf"
    assert paths.shown(str(comfy) + os.sep) == "ComfyUI"
    assert paths.shown("") == "" and paths.shown(None) == ""


def test_outside_comfyui_there_is_no_folder_to_count_from(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_folder_paths", lambda: None)
    assert paths.shown(str(tmp_path / "models" / "LLM" / "x.gguf")) == "x.gguf"
