"""Finding and fetching the voice separator's weights, without the 0.85 GB of them."""

from __future__ import annotations

import json
import os
import struct

import pytest

from yue2_comfy import constants, discovery, download, paths


@pytest.fixture(autouse=True)
def nothing_remembered():
    discovery._identified.clear()
    yield
    discovery._identified.clear()


def header_only(path, names):
    """A safetensors file holding these tensor names and no data."""
    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]} for name in names}
    header["__metadata__"] = {"format": "pt"}
    body = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)) + body)
    return path


def released_file(folder, monkeypatch, size=16):
    """The released .ckpt under its own name, with the pack told that is its size."""
    monkeypatch.setattr(discovery, "VOCALS_BYTES", size)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / constants.VOCALS_NAME
    path.write_bytes(b"\0" * size)
    return path


def test_the_released_file_is_found_under_the_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    path = released_file(tmp_path, monkeypatch)
    assert discovery.find_vocals() == str(path)


def test_a_file_with_that_name_but_another_size_is_not_taken(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    path = released_file(tmp_path, monkeypatch)
    path.write_bytes(b"\0" * 7)
    assert discovery.find_vocals() == ""


def test_a_conversion_is_found_by_its_tensors_whatever_it_is_called(tmp_path, monkeypatch):
    """kijai's fp16 file sits in models/diffusion_models under a name of its own."""
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    path = header_only(tmp_path / "separators" / "MelBandRoformer_fp16.safetensors", constants.VOCALS_MARKERS)
    assert discovery.find_vocals() == str(path)
    assert discovery.identify(str(path)) == "vocals"


def test_a_file_missing_one_marker_is_not_taken(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    header_only(tmp_path / "other.safetensors", constants.VOCALS_MARKERS[:-1])
    assert discovery.find_vocals() == ""


def test_the_released_file_is_preferred_over_a_conversion(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    header_only(tmp_path / "a_first" / "MelBandRoformer_fp16.safetensors", constants.VOCALS_MARKERS)
    path = released_file(tmp_path / "z_last", monkeypatch)
    assert discovery.find_vocals() == str(path)


def test_the_refusal_names_the_pinned_link_the_file_and_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    message = discovery.vocals_missing_message([str(tmp_path)])
    assert ("https://huggingface.co/KimberleyJSN/melbandroformer/resolve/"
            + constants.VOCALS_REVISION + "/MelBandRoformer.ckpt") in message
    assert os.path.join(str(tmp_path), constants.VOCALS_NAME) in message
    assert "Looked in 1 place" in message and "MIT" in message


def test_the_download_is_pinned_and_lands_in_the_models_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "models_root", lambda: str(tmp_path))
    fetched = []
    monkeypatch.setattr(download, "fetch", lambda repo, wanted, title, progress=None, revision="main":
                        fetched.append((repo, wanted, title, revision)))
    assert download.fetch_vocals() == str(tmp_path / constants.VOCALS_NAME)
    repo, wanted, title, revision = fetched[0]
    assert repo == "KimberleyJSN/melbandroformer"
    assert revision == constants.VOCALS_REVISION
    assert wanted == {constants.VOCALS_NAME: str(tmp_path / constants.VOCALS_NAME)}
    assert "Mel-Band RoFormer" in title


def test_nothing_is_fetched_when_the_weights_are_here(monkeypatch):
    monkeypatch.setattr(discovery, "find_vocals", lambda roots=None: "here")
    monkeypatch.setattr(download, "fetch_vocals", lambda progress=None: pytest.fail("fetched"))
    assert download.ensure_vocals({"download": "off"}) == "here"


def test_downloading_off_explains_instead_of_fetching(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_vocals", lambda progress=None: pytest.fail("fetched"))
    with pytest.raises(FileNotFoundError) as refused:
        download.ensure_vocals({"download": "off"})
    assert constants.VOCALS_NAME in str(refused.value) and "'download' in YuE2 Options" in str(refused.value)


@pytest.mark.parametrize("choice", ["auto", "comfy-org", "original"])
def test_every_other_download_choice_fetches_the_released_file(monkeypatch, tmp_path, choice):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_vocals", lambda progress=None: "fetched")
    assert download.ensure_vocals({"download": choice}) == "fetched"
