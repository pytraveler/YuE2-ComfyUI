"""Finding the weights, on files the test writes itself.

None of this needs ComfyUI, torch or the real 7 GB. What it does need is an
isolated search, so every test here sets YUE2_MODELS_ROOT: without it the
search legitimately reaches the ComfyUI model folders and the Hugging Face
cache of whatever machine is running the suite, and a test that passes or
fails depending on that is worse than no test.
"""

import json
import os

import pytest

from yue2_comfy import discovery, paths
from yue2_comfy.constants import (
    LM_DIRNAME, MERGES_NAME, VAE_DIRNAME, VAE_LEGACY_DIRNAME, VAE_MARKERS, WEIGHTS_NAME,
)

from test_loader import write_safetensors

VAE_NAMES = list(VAE_MARKERS) + ["decoder.layers.0.bias"]
"""A decoder shaped like the released one, which is what discovery looks for."""

VIDEO_VAE_NAMES = ["decoder.conv_in.conv.weight", "decoder.conv_in.conv.bias",
                   "decoder.up_blocks.0.resnets.0.conv1.conv.weight",
                   "encoder.conv_in.conv.weight"]
"""A diffusers-shaped video VAE, the kind that sits in models/vae on a machine
that generates video. It has decoder tensors, and none of YuE2's."""


@pytest.fixture(autouse=True)
def forget_identifications():
    """The identification cache is keyed on path, and tmp_path is reused."""
    discovery._identified.clear()
    yield
    discovery._identified.clear()


def publish(root, lm_names=("lm_head.weight", "vae2llm.weight"),
            vae_names=tuple(VAE_NAMES), vae_dirname=VAE_DIRNAME):
    """Lay out the three files the way the Hub serves them."""
    lm_dir = root / LM_DIRNAME
    vae_dir = root / vae_dirname
    lm_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)
    write_safetensors(lm_dir / WEIGHTS_NAME, list(lm_names))
    write_safetensors(vae_dir / WEIGHTS_NAME, list(vae_names))
    (lm_dir / MERGES_NAME).write_bytes(b"placeholder")
    return lm_dir, vae_dir


def test_the_published_layout_is_found_without_reading_a_file(tmp_path, monkeypatch):
    """The common case must not depend on the files being readable at all.

    A layout that matches the published folder names is believed on sight, so
    a truncated or still-downloading file is reported by the loader, which can
    say something useful about it, rather than silently skipped here.
    """
    lm_dir, vae_dir = publish(tmp_path)
    (lm_dir / WEIGHTS_NAME).write_bytes(b"not a safetensors file")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    found = discovery.locate("standard")

    assert found.lm == str(lm_dir / WEIGHTS_NAME)
    assert found.vae == str(vae_dir / WEIGHTS_NAME)
    assert found.merges == str(lm_dir / MERGES_NAME)


def test_the_legacy_variant_picks_the_other_vae(tmp_path, monkeypatch):
    publish(tmp_path)
    _, legacy_dir = publish(tmp_path, vae_dirname=VAE_LEGACY_DIRNAME)
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    assert discovery.locate("legacy").vae == str(legacy_dir / WEIGHTS_NAME)
    assert discovery.locate("standard").vae == str(tmp_path / VAE_DIRNAME / WEIGHTS_NAME)


def test_a_hugging_face_snapshot_is_a_home_too(tmp_path, monkeypatch):
    """Files sit at the snapshot's own top level, named by revision.

    The folder that says what the files are is two levels up, so the layout
    pass has to look there instead of at the directory it is standing in.
    """
    hub = tmp_path / "hub"
    lm = hub / "models--m-a-p--YuE2-3B" / "snapshots" / "0123456789abcdef"
    vae = hub / "models--m-a-p--YuE2-Vae" / "snapshots" / "fedcba9876543210"
    lm.mkdir(parents=True)
    vae.mkdir(parents=True)
    write_safetensors(lm / WEIGHTS_NAME, ["lm_head.weight", "vae2llm.weight"])
    write_safetensors(vae / WEIGHTS_NAME, VAE_NAMES)
    (lm / MERGES_NAME).write_bytes(b"placeholder")

    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(paths, "comfy_roots", lambda: [])
    monkeypatch.setattr(paths, "checkout_sibling_root", lambda: str(tmp_path / "nothing"))

    found = discovery.locate("standard")

    assert found.lm == str(lm / WEIGHTS_NAME)
    assert found.vae == str(vae / WEIGHTS_NAME)


def test_a_renamed_file_is_found_by_what_is_inside_it(tmp_path, monkeypatch):
    """The whole point of the second pass: the name carries no authority."""
    loose = tmp_path / "diffusion_models"
    loose.mkdir()
    write_safetensors(loose / "yue2-music-model.safetensors",
                      ["lm_head.weight", "vae2llm.weight"])
    write_safetensors(loose / "some-audio-vae.safetensors", VAE_NAMES)
    (loose / "vocab.tiktoken").write_bytes(b"placeholder")
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    found = discovery.locate("standard")

    assert found.lm.endswith("yue2-music-model.safetensors")
    assert found.vae.endswith("some-audio-vae.safetensors")
    assert found.merges.endswith("vocab.tiktoken")


def test_another_models_decoder_is_not_taken_for_the_vae(tmp_path, monkeypatch):
    """Issue #1: a video VAE in models/vae was picked up as the YuE2 decoder.

    The machine has the released backbone but no standard decoder beside it,
    which is what a run leaves behind after fetching the legacy one. The sweep
    then goes looking, and every video VAE ever published has tensors whose
    names start with "decoder.". Finding none of YuE2's own names in this one,
    the search has to come up empty rather than hand the loader a file that
    decodes something else.
    """
    published = tmp_path / LM_DIRNAME
    published.mkdir()
    write_safetensors(published / WEIGHTS_NAME, ["lm_head.weight", "vae2llm.weight"])
    (published / MERGES_NAME).write_bytes(b"placeholder")
    loose = tmp_path / "vae"
    loose.mkdir()
    stranger = write_safetensors(loose / "video_vae_fp16.safetensors", VIDEO_VAE_NAMES)
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    assert discovery.identify(stranger) == ""
    with pytest.raises(FileNotFoundError):
        discovery.locate("standard")


def test_an_unrelated_checkpoint_is_not_mistaken_for_the_model(tmp_path, monkeypatch):
    """A Qwen-shaped checkpoint in the same folder has lm_head but no vae2llm."""
    loose = tmp_path / "diffusion_models"
    loose.mkdir()
    write_safetensors(loose / "some-llm.safetensors", ["lm_head.weight", "model.layers.0.q"])
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    with pytest.raises(FileNotFoundError):
        discovery.locate("standard")


def test_identify_reads_the_discriminator_not_the_shape(tmp_path):
    lm = write_safetensors(tmp_path / "a.safetensors", ["lm_head.weight", "vae2llm.weight"])
    vae = write_safetensors(tmp_path / "b.safetensors", VAE_NAMES)
    other = write_safetensors(tmp_path / "c.safetensors", ["lm_head.weight"])
    junk = tmp_path / "d.safetensors"
    junk.write_bytes(b"nonsense")

    assert discovery.identify(lm) == "lm"
    assert discovery.identify(vae) == "vae"
    assert discovery.identify(other) == ""
    assert discovery.identify(str(junk)) == ""


def test_the_vae_variant_comes_from_the_config_before_the_folder(tmp_path):
    """Both releases are the same size, so only the publisher's word separates them."""
    directory = tmp_path / VAE_DIRNAME
    directory.mkdir()
    weights = write_safetensors(directory / WEIGHTS_NAME, ["decoder.a"])
    assert discovery.vae_variant(weights) == "standard"

    (directory / "config.json").write_text(json.dumps({"release_variant": "legacy"}),
                                           encoding="utf-8")
    assert discovery.vae_variant(weights) == "legacy"


def test_the_override_is_the_only_root_when_it_is_set(tmp_path, monkeypatch):
    """Otherwise it cannot be used to stop the search guessing."""
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(paths, "comfy_roots", lambda: ["should-not-be-reached"])

    assert paths.search_roots() == [str(tmp_path)]


def test_a_mistyped_override_warns_instead_of_bricking_the_pack(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path / "typo"))
    monkeypatch.setattr(paths, "comfy_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(paths, "hf_cache_roots", lambda: [])

    assert str(tmp_path) in paths.search_roots()


def test_the_refusal_names_all_three_files_and_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    with pytest.raises(FileNotFoundError) as error:
        discovery.locate("legacy")

    message = str(error.value)
    assert message.count(os.path.join(LM_DIRNAME, WEIGHTS_NAME)) == 1
    assert message.count(os.path.join(VAE_LEGACY_DIRNAME, WEIGHTS_NAME)) == 1
    assert MERGES_NAME in message
    assert str(tmp_path) in message
    assert message.count("https://huggingface.co/") == 3


def test_a_network_root_is_refused_rather_than_contacted(monkeypatch):
    """Looking at a UNC path authenticates against whatever host it names."""
    monkeypatch.setenv(paths.ENV_ROOT, r"\\attacker\share\models")

    with pytest.raises(RuntimeError) as error:
        paths.search_roots()

    assert "network path" in str(error.value)


def test_an_extended_local_path_is_not_a_network_path():
    assert not paths.is_network_path(r"\\?\C:\models\YuE2")
    assert paths.is_network_path(r"\\nas\models")
    assert not paths.is_network_path(r"C:\models\YuE2")


def test_folder_kind_reads_the_repository_name_of_a_snapshot(tmp_path):
    snapshot = tmp_path / "models--m-a-p--YuE2-Vae-legacy" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert discovery.folder_kind(str(snapshot)) == "legacy"
    assert discovery.folder_kind(str(tmp_path / LM_DIRNAME)) == "lm"
    assert discovery.folder_kind(str(tmp_path / "unrelated")) == ""


def test_the_sweep_does_not_descend_past_the_second_level(tmp_path, monkeypatch):
    """A deep sweep of a large model tree costs real time for nothing."""
    deep = tmp_path / "one" / "two" / "three"
    deep.mkdir(parents=True)
    write_safetensors(deep / "model.safetensors", ["lm_head.weight", "vae2llm.weight"])
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))

    with pytest.raises(FileNotFoundError):
        discovery.locate("standard")


def test_models_root_is_created_and_registered(tmp_path, monkeypatch):
    """Without ComfyUI the fallback still has to name a directory that exists."""
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path / "made-on-demand"))
    monkeypatch.setattr(paths, "_folder_paths", lambda: None)

    root = paths.models_root()

    assert os.path.isdir(root)
