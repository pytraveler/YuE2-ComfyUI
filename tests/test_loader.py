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


def write_safetensors(path, names, shapes=None):
    """A minimal valid safetensors file of float32 zeros, one scalar each unless ``shapes`` names one."""
    header = {}
    offset = 0
    for name in names:
        shape = list((shapes or {}).get(name, [1]))
        size = 4
        for extent in shape:
            size *= extent
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
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
    assert os.path.join("YuE2-3B", "model.safetensors") in message
    assert os.path.join("YuE2-Vae-legacy", "model.safetensors") in message
    assert "qwen.tiktoken" in message


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

    Checked in a fresh interpreter. Tests that build the model import the
    modeling modules into this one, and this process's sys.modules would then
    say whatever the order the tests ran in says.
    """
    import pathlib
    import subprocess
    import sys

    package = pathlib.Path(loader.__file__).resolve().parent
    program = "\n".join([
        "import sys, types",
        "package = types.ModuleType('yue2_comfy')",
        "package.__path__ = [{!r}]".format(str(package)),
        "sys.modules['yue2_comfy'] = package",
        "from yue2_comfy.vendor import yue2",
        "assert yue2.UPSTREAM_VERSION",
        "assert 'yue2_comfy.vendor.yue2.modeling_yue2' not in sys.modules",
        "assert 'yue2_comfy.vendor.yue2.modeling_vae' not in sys.modules",
        "assert not hasattr(yue2, 'YuE2Pipeline')",
    ])
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_narrow_decoder_is_refused_by_name(tmp_path):
    """Issue #1: the refusal used to give a dtype and no file.

    What was actually wrong there was the file, not its dtype -- the search had
    picked up another model's VAE -- and the message sent the reporter looking
    at YuE2's decoder instead. The path is the first thing it has to say.
    """
    torch = pytest.importorskip("torch")

    state = {"decoder.layers.0.weight_g": torch.zeros(1, dtype=torch.float16),
             "decoder.layers.0.bias": torch.zeros(1)}

    with pytest.raises(ValueError) as error:
        loader._refuse_narrow(state, str(tmp_path / "video_vae_fp16.safetensors"))

    message = str(error.value)
    assert "video_vae_fp16.safetensors" in message
    assert "decoder.layers.0.weight_g" in message
    assert "decoder.layers.0.bias" not in message


def test_a_repack_is_never_built_into_a_legacy_decoder(tmp_path):
    """Same 217 tensors, same shapes, all different values: the strict load cannot tell."""
    with pytest.raises(ValueError) as refusal:
        loader.load_repack(str(tmp_path / "absent.safetensors"), "cpu", "legacy")
    assert "legacy" in str(refusal.value)


def test_a_decoder_beside_a_repack_is_part_of_what_is_loaded(tmp_path):
    from yue2_comfy import discovery

    repack = write_safetensors(tmp_path / "r.safetensors", ["a"])
    decoder = write_safetensors(tmp_path / "d.safetensors", ["b"])
    alone = loader._cache_key(discovery.Files(repack=repack), "cpu", "standard")
    beside = loader._cache_key(discovery.Files(repack=repack, vae=decoder), "cpu", "legacy")
    assert alone != beside
    assert beside[1] == loader._stamp(decoder)


def test_a_rewritten_decoder_beside_a_repack_is_loaded_again(tmp_path):
    """The stamp is what changes when the decoder is replaced; the key has to follow it."""
    import os

    from yue2_comfy import discovery

    repack = write_safetensors(tmp_path / "r.safetensors", ["a"])
    decoder = tmp_path / "d.safetensors"
    write_safetensors(decoder, ["b"])
    files = discovery.Files(repack=repack, vae=str(decoder))
    before = loader._cache_key(files, "cpu", "legacy")
    write_safetensors(decoder, ["b", "c"])
    os.utime(decoder, ns=(1, 1))
    assert loader._cache_key(files, "cpu", "legacy") != before


def test_the_repack_takes_its_decoder_from_the_file_beside_it(tmp_path, monkeypatch):
    """With a decoder path, load_vae builds it and the repack's own decoder is never built."""
    pytest.importorskip("torch")
    from yue2_comfy import repack

    calls = []
    monkeypatch.setattr(loader, "tensor_names", lambda path: [])
    monkeypatch.setattr(loader, "_read_state", lambda path, prefix="": {})
    monkeypatch.setattr(repack, "lm_state", lambda state: {})
    monkeypatch.setattr(loader, "_build_lm", lambda state, settings, device, file_backed=False:
                        calls.append(("lm", file_backed)) or "lm")
    monkeypatch.setattr(loader, "load_vae", lambda path, variant: calls.append(("vae", path, variant))
                        or "vae")
    monkeypatch.setattr(loader, "_build_vae", lambda *args: calls.append(("repack vae",)))
    monkeypatch.setattr(repack, "merges_for", lambda path: "merges")
    monkeypatch.setattr(loader, "load_tokenizer", lambda path: "tokenizer")

    result = loader.load_repack(str(tmp_path / "r.safetensors"), "cpu", "legacy",
                                decoder_path="legacy.safetensors")

    assert result == ("lm", "vae", "tokenizer")
    assert calls == [("lm", True), ("vae", "legacy.safetensors", "legacy")]


def test_acquire_hands_the_decoder_beside_a_repack_to_the_loader(tmp_path, monkeypatch):
    from yue2_comfy import discovery

    seen = []
    monkeypatch.setattr(loader, "load_repack", lambda path, device, variant, progress, decoder:
                        seen.append((path, variant, decoder)) or ("lm", "vae", "tokenizer"))
    monkeypatch.setattr(loader, "_free_comfy_vram", lambda spec: None)
    monkeypatch.setattr(loader.devices, "resolve", lambda spec: "cpu")
    repack = write_safetensors(tmp_path / "r.safetensors", ["a"])
    decoder = write_safetensors(tmp_path / "d.safetensors", ["b"])
    try:
        loader.acquire(discovery.Files(repack=repack, vae=decoder), "cpu", "legacy")
    finally:
        loader.unload()
    assert seen == [(repack, "legacy", decoder)]


def test_the_released_backbone_is_marked_as_still_in_its_file(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from yue2_comfy import placement

    marked = []
    monkeypatch.setattr(loader, "_read_state", lambda path, prefix="": {})
    monkeypatch.setattr(placement, "load", lambda model, device, file_backed=False:
                        marked.append(file_backed) or model)

    class Model:
        def load_state_dict(self, state, strict=True, assign=False):
            pass

        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

    from yue2_comfy.vendor.yue2 import modeling_yue2
    monkeypatch.setattr(modeling_yue2, "YuE2ForCausalLM", lambda config: Model())
    loader.load_lm(str(tmp_path / "m.safetensors"), "cpu")
    assert marked == [True]


def _without_tiktoken(monkeypatch):
    """The parse fails before tiktoken is used, so an empty module stands in for it."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "tiktoken", types.ModuleType("tiktoken"))


def test_a_placeholder_vocabulary_is_refused_by_its_path(tmp_path, monkeypatch):
    """The two-byte file from 2026-09-19 has to say where it is and what to do."""
    _without_tiktoken(monkeypatch)
    folder = tmp_path / "YuE2-3B"
    folder.mkdir()
    placeholder = folder / "qwen.tiktoken"
    placeholder.write_bytes(b"{}")

    with pytest.raises(ValueError) as refusal:
        loader.load_tokenizer(str(placeholder))
    message = str(refusal.value)
    assert str(placeholder) in message
    assert "2 bytes" in message and "2,561,218" in message
    assert "not enough values to unpack" in message
    assert "m-a-p/YuE2-3B" in message and loader.paths.ENV_ROOT in message
    assert "writes it again" not in message


def test_a_broken_copy_in_the_packs_own_cache_is_simply_deleted(tmp_path, monkeypatch):
    _without_tiktoken(monkeypatch)
    cache = tmp_path / ".vocabulary"
    cache.mkdir()
    short = cache / "qwen-7799983228-1.tiktoken"
    short.write_bytes(b"IQ== 0\nIg== 1\n")

    with pytest.raises(ValueError) as refusal:
        loader.load_tokenizer(str(short))
    message = str(refusal.value)
    assert str(short) in message
    assert "151643" in message
    assert "delete it, and the next run writes it again" in message
