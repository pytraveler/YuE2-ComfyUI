"""The list of LoRAs the node offers: only YuE2 ones, by stable names, read once per file."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types

import pytest

import lora_files as lf
from yue2_comfy import paths
from yue2_comfy.lora import catalogue, formats


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """Two LoRA roots: an ordinary one with an image LoRA beside YuE2 ones, and a second one."""
    first, second = tmp_path / "loras", tmp_path / "more_loras"
    (first / "yue2" / "rock").mkdir(parents=True)
    second.mkdir()
    lf.write(first / "yue2" / "rock" / "ar.safetensors", lf.map_layer(0))
    lf.write(first / "jpop.safetensors", lf.comfy_layer(0, rank=4), {"trigger_word": "jpstyle26"})
    lf.write(first / "sdxl_style.safetensors", {
        "lora_unet_mid_block_attentions_0_proj_in.lora_down.weight": ("F16", (4, 320)),
        "lora_unet_mid_block_attentions_0_proj_in.lora_up.weight": ("F16", (320, 4))})
    hum = lf.map_layer(0, half="nar", prefix="")
    hum["hum_proj.0.weight"] = ("F32", (2048, 64))
    lf.write(first / "hum.safetensors", hum)
    lf.write(second / "jpop.safetensors", lf.map_layer(1))
    lf.write(second / "extra.safetensors", lf.map_layer(2, half="nar"))
    monkeypatch.setattr(catalogue, "roots", lambda: [str(first), str(second)])
    return first, second


def test_only_yue2_loras_are_listed_by_forward_slashed_names(folders):
    names = [entry.name for entry in catalogue.entries()]
    assert names == ["extra.safetensors", "hum.safetensors", "jpop.safetensors",
                     "yue2/rock/ar.safetensors"]


def test_a_name_in_two_folders_is_the_first_folders_file(folders):
    first, _second = folders
    entry = catalogue.find("jpop.safetensors")
    assert entry.path == str(first / "jpop.safetensors")
    assert entry.summary["triggers"] == ["jpstyle26"]


def test_a_file_that_cannot_be_folded_is_listed_with_its_reason(folders):
    entry = catalogue.find("hum.safetensors")
    assert not entry.usable and "hum_proj" in entry.summary["problem"]


def test_the_listing_is_plain_json_with_what_each_row_shows(folders):
    listing = catalogue.listing()
    json.dumps(listing)
    row = next(item for item in listing if item["name"] == "yue2/rock/ar.safetensors")
    assert row["halves"] == {"ar": {"tensors": 7, "rank": 4}}
    assert row["problem"] == ""


def test_a_saved_name_is_found_whichever_slashes_it_was_saved_with(folders):
    assert catalogue.find("yue2\\rock\\ar.safetensors").name == "yue2/rock/ar.safetensors"
    assert catalogue.find(" YUE2/Rock/AR.safetensors ").name == "yue2/rock/ar.safetensors"


def test_a_missing_file_is_refused_with_what_is_there_now(folders):
    with pytest.raises(FileNotFoundError) as caught:
        catalogue.find("gone.safetensors")
    assert "gone.safetensors" in str(caught.value) and "jpop.safetensors" in str(caught.value)


def test_each_file_is_read_once_until_it_changes(folders, monkeypatch):
    first, _second = folders
    reads = []
    original = formats.read

    def counted(path):
        reads.append(os.path.basename(path))
        return original(path)

    monkeypatch.setattr(formats, "read", counted)
    catalogue.entries()
    before = len(reads)
    catalogue.entries()
    assert len(reads) == before
    target = first / "jpop.safetensors"
    stat = os.stat(target)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    catalogue.entries()
    assert reads[before:] == ["jpop.safetensors"]


def test_the_kept_verdicts_survive_a_restart_and_name_no_file(folders, monkeypatch):
    catalogue.entries()
    kept = catalogue._cache_path()
    text = open(kept, encoding="utf-8").read()
    assert "jpop" not in text and "loras" not in text and "sdxl" not in text
    saved = json.loads(text)
    assert saved["format"] == catalogue.CACHE_FORMAT and len(saved["entries"]) == 5

    reads = []
    monkeypatch.setattr(formats, "read", lambda path: reads.append(path))
    catalogue._MEMORY.clear()
    catalogue._DISK.update(loaded=False, entries={}, dirty=False)
    assert len(catalogue.entries()) == 4
    assert reads == []


def test_a_damaged_verdict_file_is_an_empty_one(folders):
    kept = catalogue._cache_path()
    os.makedirs(os.path.dirname(kept), exist_ok=True)
    with open(kept, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert len(catalogue.entries()) == 4


def test_the_identity_of_a_file_is_its_sha256(folders):
    first, _second = folders
    path = str(first / "jpop.safetensors")
    with open(path, "rb") as handle:
        expected = hashlib.sha256(handle.read()).hexdigest()
    assert catalogue.identity(path) == expected
    assert catalogue.identity(path) == expected


def test_the_lora_folders_are_comfyuis_then_the_one_beside_the_checkout(tmp_path, monkeypatch):
    registered, models = tmp_path / "elsewhere" / "loras", tmp_path / "models"
    registered.mkdir(parents=True)
    (models / "loras").mkdir(parents=True)
    sibling = tmp_path / "checkout_models"
    (sibling / "loras").mkdir(parents=True)
    stub = types.SimpleNamespace(models_dir=str(models),
                                 get_folder_paths=lambda name: [str(registered)])
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.setattr(paths, "checkout_sibling_root", lambda: str(sibling))
    assert paths.lora_roots() == [str(registered), str(models / "loras"), str(sibling / "loras")]


def test_the_models_root_override_is_the_only_lora_folder(tmp_path, monkeypatch):
    (tmp_path / "loras").mkdir()
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    assert paths.lora_roots() == [str(tmp_path / "loras")]
