"""Loader checks that need neither ComfyUI, nor torch, nor the weights.

Everything here works on files the test writes itself. The one thing that
cannot be faked -- that 628 and 217 real tensors load into the real classes --
is checked by running it, not by a unit test.
"""

import json
import os
import struct

import pytest

from yue2_comfy import loader


def write_safetensors(path, names):
    """A minimal valid safetensors file: header only, one float32 scalar each."""
    header = {}
    offset = 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    header["__metadata__"] = {"format": "pt"}
    body = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        handle.write(b"\0" * offset)
    return str(path)


def test_read_header_returns_the_tensor_table(tmp_path):
    path = write_safetensors(tmp_path / "m.safetensors", ["a.weight", "b.weight"])
    header = loader.read_header(path)
    assert header["a.weight"]["shape"] == [1]
    assert "__metadata__" in header


def test_tensor_names_drops_the_metadata_entry(tmp_path):
    path = write_safetensors(tmp_path / "m.safetensors", ["a.weight", "b.weight"])
    assert loader.tensor_names(path) == ["a.weight", "b.weight"]


def test_header_reading_never_maps_the_tensors(tmp_path):
    """The point of reading the header by hand is not paying for the payload.

    A file that claims tensors it does not contain still has a readable header,
    which is what makes this cheap enough to run over every candidate file.
    """
    path = str(tmp_path / "claims.safetensors")
    body = json.dumps({"big": {"dtype": "F32", "shape": [1 << 30],
                               "data_offsets": [0, 4 << 30]}}).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
    assert loader.tensor_names(path) == ["big"]
    assert os.path.getsize(path) < 1024


@pytest.mark.parametrize("content", [
    b"",
    b"not a safetensors file at all",
    struct.pack("<Q", 1 << 40),
    struct.pack("<Q", 4) + b"[1]\x00",
])
def test_read_header_refuses_what_is_not_a_safetensors_file(tmp_path, content):
    path = str(tmp_path / "junk.bin")
    with open(path, "wb") as handle:
        handle.write(content)
    with pytest.raises(ValueError):
        loader.read_header(path)


def test_config_is_optional(tmp_path):
    """No config.json means the vendored architecture defaults, not an error.

    That is what lets someone drop a single .safetensors anywhere and have it
    work, which is the whole reason this pack does not use from_pretrained.
    """
    weights = write_safetensors(tmp_path / "m.safetensors", ["a"])
    assert loader._config_dict(weights) == {}


def test_config_is_read_from_beside_the_weights(tmp_path):
    weights = write_safetensors(tmp_path / "m.safetensors", ["a"])
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 2048}))
    assert loader._config_dict(weights) == {"hidden_size": 2048}


@pytest.mark.parametrize("text", ["{ not json", "[1, 2, 3]"])
def test_a_broken_config_falls_back_instead_of_raising(tmp_path, text):
    weights = write_safetensors(tmp_path / "m.safetensors", ["a"])
    (tmp_path / "config.json").write_text(text)
    assert loader._config_dict(weights) == {}


def test_the_cache_key_follows_the_file_not_the_name(tmp_path):
    """Replacing a file under the same path has to invalidate the cache.

    Someone who overwrites model.safetensors with a different checkpoint gets
    the new one, without having to know that a cache exists.
    """
    files = loader.Files(
        lm=write_safetensors(tmp_path / "lm.safetensors", ["a"]),
        vae=write_safetensors(tmp_path / "vae.safetensors", ["decoder.a"]),
        merges=write_safetensors(tmp_path / "merges.bin", ["m"]),
    )
    first = loader._cache_key(files, "cuda:0", "standard")
    assert loader._cache_key(files, "cuda:0", "standard") == first
    assert loader._cache_key(files, "cpu", "standard") != first
    assert loader._cache_key(files, "cuda:0", "legacy") != first

    write_safetensors(tmp_path / "lm.safetensors", ["a", "b"])
    assert loader._cache_key(files, "cuda:0", "standard") != first


def test_locate_honours_the_environment_override(tmp_path, monkeypatch):
    for directory, name in (("YuE2-3B", "model.safetensors"),
                            ("YuE2-3B", "qwen.tiktoken"),
                            ("YuE2-Vae", "model.safetensors")):
        target = tmp_path / directory
        target.mkdir(exist_ok=True)
        (target / name).write_bytes(b"x")
    monkeypatch.setenv("YUE2_MODELS_ROOT", str(tmp_path))
    found = loader.locate("standard")
    assert found.lm.endswith(os.path.join("YuE2-3B", "model.safetensors"))
    assert found.vae.endswith(os.path.join("YuE2-Vae", "model.safetensors"))
    assert found.merges.endswith("qwen.tiktoken")


def test_locate_names_every_missing_file(tmp_path, monkeypatch):
    """A refusal has to name all three paths, not just the first one missing.

    Being told about one missing file at a time turns one fix into three runs.
    """
    monkeypatch.setenv("YUE2_MODELS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError) as error:
        loader.locate("legacy")
    message = str(error.value)
    assert message.count("model.safetensors") == 2
    assert "qwen.tiktoken" in message
    assert "YuE2-Vae-legacy" in message


def test_unload_is_safe_when_nothing_is_loaded():
    loader.unload()
    loader.unload()
    assert not loader.is_loaded()


def test_the_vendor_package_stays_lazy():
    """Importing the vendored package must not import transformers.

    The two modeling modules are the only vendored files that pull names out of
    transformers, and those names move between major versions. Importing them
    at registration time would turn a future transformers change into a pack
    that does not load at all.
    """
    import sys

    from yue2_comfy.vendor import yue2

    assert yue2.UPSTREAM_VERSION
    assert "yue2_comfy.vendor.yue2.modeling_yue2" not in sys.modules
    assert "yue2_comfy.vendor.yue2.modeling_vae" not in sys.modules
    with pytest.raises(AttributeError):
        yue2.YuE2Pipeline
