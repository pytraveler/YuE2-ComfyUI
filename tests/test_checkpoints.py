"""Checkpoint selection stays explicit and a saved plan never falls back to another file."""

import json
import struct

import pytest

from yue2_comfy import checkpoints, constants, discovery, download, nodes, paths, staged


def checkpoint(path, quantized=False):
    header = {"text_encoders.model.test": {}, "model.diffusion_model.test": {},
              "vae.test": {}, "text_encoders.yue2_tokenizer_json": {}}
    if quantized:
        header["model.diffusion_model.test.comfy_quant"] = {}
    body = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    return str(path)


def test_picker_and_selection_prefer_the_explicit_finetune(monkeypatch, tmp_path):
    base = checkpoint(tmp_path / "yue2_3b_bf16.safetensors")
    fine = checkpoint(tmp_path / "fine.safetensors")
    (tmp_path / "image.safetensors").write_bytes(b"other model")
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(tmp_path)])
    labels = checkpoints.catalogue()
    assert set(labels.values()) == {base, fine}
    choice = next(name for name, path in labels.items() if path == fine)
    settings = dict(constants.DEFAULT_OPTIONS, checkpoint=choice, download="off", quantization="int8")
    assert download.ensure(settings) == discovery.Files(repack=fine)
    before = nodes.YuE2Options.IS_CHANGED(checkpoint=choice)
    with open(fine, "ab") as handle:
        handle.write(b"replacement")
    assert nodes.YuE2Options.IS_CHANGED(checkpoint=choice) != before


def test_missing_selection_never_uses_auto_or_downloads(monkeypatch):
    monkeypatch.setattr(discovery, "locate", lambda *args: pytest.fail("must not fall back"))
    monkeypatch.setattr(download, "fetch_repack", lambda *args: pytest.fail("must not download a base"))
    with pytest.raises(ValueError, match="not available"):
        download.ensure(dict(constants.DEFAULT_OPTIONS, checkpoint="gone.safetensors"))


def test_identical_relative_names_do_not_change_identity_when_root_order_changes(monkeypatch, tmp_path):
    folders = [tmp_path / "first" / "checkpoints", tmp_path / "second" / "checkpoints"]
    for folder in folders:
        folder.mkdir(parents=True)
        checkpoint(folder / "fine.safetensors")
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(folder) for folder in folders])
    first = checkpoints.catalogue()
    folders.reverse()
    assert checkpoints.catalogue() == first
    assert len(first) == 2


def test_auto_plan_keeps_its_source_and_detects_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(tmp_path)])
    path = checkpoint(tmp_path / "fine.safetensors")
    settings = dict(constants.DEFAULT_OPTIONS)
    checkpoints.remember(settings, discovery.Files(repack=path))
    monkeypatch.setattr(discovery, "locate", lambda *args: pytest.fail("must reuse the saved source"))
    assert download.ensure(settings).repack == path
    with open(path, "ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        download.ensure(settings)


def test_default_render_options_preserve_the_plans_explicit_choice(tmp_path):
    settings = dict(constants.DEFAULT_OPTIONS, checkpoint="fine.safetensors")
    checkpoints.remember(settings, discovery.Files(repack=checkpoint(tmp_path / "fine.safetensors")))
    plan = {"settings": settings}
    merged = staged._settings(plan, dict(constants.DEFAULT_OPTIONS, ode_steps=24), None)
    assert merged["checkpoint"] == "fine.safetensors"
    assert merged["ode_steps"] == 24
    changed = staged._settings(plan, {"checkpoint": "other.safetensors"}, None)
    assert changed["checkpoint"] == "other.safetensors"
    assert checkpoints.pinned(changed) is None


def test_saved_source_rejects_a_network_path_before_touching_it(monkeypatch):
    settings = dict(constants.DEFAULT_OPTIONS, checkpoint_source={
        "choice": "auto", "files": {"repack": "\\\\untrusted\\share\\file.safetensors"}})
    monkeypatch.setattr(checkpoints, "_stamp", lambda path: pytest.fail("must not stat a network path"))
    with pytest.raises(ValueError, match="outside"):
        checkpoints.pinned(settings)


def test_an_explicit_int8_file_uses_the_existing_repack_loader(monkeypatch, tmp_path):
    path = checkpoint(tmp_path / "fine_int8.safetensors", quantized=True)
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(tmp_path)])
    choice = next(iter(checkpoints.catalogue()))
    assert discovery._identify_by_header(path) == "repack_int8"
    assert download.ensure(dict(constants.DEFAULT_OPTIONS, checkpoint=choice)).repack == path


def test_changing_the_decoder_keeps_the_pinned_backbone(monkeypatch, tmp_path):
    path = checkpoint(tmp_path / "fine.safetensors")
    legacy = tmp_path / constants.VAE_LEGACY_DIRNAME
    legacy.mkdir()
    (legacy / constants.WEIGHTS_NAME).write_bytes(b"decoder supplied by this test")
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(paths, "search_roots", lambda: [str(tmp_path)])
    settings = dict(constants.DEFAULT_OPTIONS)
    checkpoints.remember(settings, discovery.Files(repack=path))
    settings["vae"] = "legacy"
    assert download.ensure(settings) == discovery.Files(repack=path, vae=str(legacy / constants.WEIGHTS_NAME))


def test_explicit_native_weights_keep_their_own_backbone_and_vocabulary(monkeypatch, tmp_path):
    home = tmp_path / "finetune"
    home.mkdir()
    body = json.dumps({"lm_head.weight": {}, "vae2llm.weight": {}}).encode()
    lm = home / "renamed.safetensors"
    lm.write_bytes(struct.pack("<Q", len(body)) + body)
    vocab = home / constants.MERGES_NAME
    vocab.write_bytes(b"vocabulary supplied by this test")
    decoder = tmp_path / constants.VAE_DIRNAME
    decoder.mkdir()
    (decoder / constants.WEIGHTS_NAME).write_bytes(b"decoder supplied by this test")
    monkeypatch.setattr(checkpoints, "roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(paths, "search_roots", lambda: [str(tmp_path)])
    choice = next(iter(checkpoints.catalogue()))
    files = download.ensure(dict(constants.DEFAULT_OPTIONS, checkpoint=choice, download="off"))
    assert files == discovery.Files(lm=str(lm), vae=str(decoder / constants.WEIGHTS_NAME), merges=str(vocab))
