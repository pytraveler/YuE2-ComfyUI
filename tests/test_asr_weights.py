"""Finding and fetching Qwen3-ASR-1.7B, without the 3.8 GB of weights."""

from __future__ import annotations

import json
import os
import struct

import pytest

from yue2_comfy import constants, discovery, download, paths


def asr_folder(folder, width=2048, tokenizer=True):
    """A folder that looks like the release: a header-only weights file and a tokenizer."""
    folder.mkdir(parents=True, exist_ok=True)
    header = {constants.ASR_MARKER: {"dtype": "BF16", "shape": [width, 1024], "data_offsets": [0, 0]},
              "__metadata__": {"format": "pt"}}
    body = json.dumps(header).encode("utf-8")
    with open(folder / constants.WEIGHTS_NAME, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)) + body)
    if tokenizer:
        (folder / constants.ASR_TOKENIZER_NAME).write_text("{}", encoding="utf-8")
    return folder


def test_the_downloaded_folder_is_found_under_the_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    folder = asr_folder(tmp_path / constants.ASR_DIRNAME)
    assert discovery.find_asr() == str(folder)


def test_a_renamed_folder_is_known_by_its_projector(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    folder = asr_folder(tmp_path / "speech")
    assert discovery.find_asr() == str(folder)


def test_the_smaller_build_is_not_taken_for_it(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    asr_folder(tmp_path / "Qwen3-ASR-0.6B", width=1024)
    assert discovery.find_asr() == ""


def test_weights_without_their_tokenizer_are_not_enough(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    asr_folder(tmp_path / constants.ASR_DIRNAME, tokenizer=False)
    assert discovery.find_asr() == ""


def test_a_collection_beside_the_checkout_is_searched(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.setattr(paths, "_folder_paths", lambda: None)
    monkeypatch.setattr(paths, "hf_cache_roots", lambda: [])
    monkeypatch.setattr(paths, "checkout_sibling_root", lambda: str(tmp_path))
    folder = asr_folder(tmp_path / "Qwen-ASR-collection" / "Qwen-ASR-1.7B-hf")
    assert discovery.find_asr() == str(folder)


def test_the_refusal_names_the_links_the_folder_and_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    message = discovery.asr_missing_message([str(tmp_path)])
    for name in constants.ASR_FILES:
        assert "https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/resolve/main/" + name in message
    assert os.path.join(str(tmp_path), constants.ASR_DIRNAME) in message
    assert "Looked in 1 place" in message and "Apache-2.0" in message


def test_the_download_lands_in_its_own_folder_under_the_models_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "models_root", lambda: str(tmp_path))
    fetched = []
    monkeypatch.setattr(download, "fetch", lambda repo, wanted, title, progress=None: fetched.append((repo, wanted, title)))
    assert download.fetch_asr() == str(tmp_path / constants.ASR_DIRNAME)
    repo, wanted, title = fetched[0]
    assert repo == "Qwen/Qwen3-ASR-1.7B-hf"
    assert wanted == {name: str(tmp_path / constants.ASR_DIRNAME / name) for name in constants.ASR_FILES}
    assert "3.80 GB" in title or "3.8" in title


def test_nothing_is_fetched_when_the_folder_is_here(monkeypatch):
    monkeypatch.setattr(discovery, "find_asr", lambda roots=None: "here")
    monkeypatch.setattr(download, "fetch_asr", lambda progress=None: pytest.fail("fetched"))
    assert download.ensure_asr({"download": "off"}) == "here"


def test_downloading_off_explains_instead_of_fetching(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_asr", lambda progress=None: pytest.fail("fetched"))
    with pytest.raises(FileNotFoundError) as refused:
        download.ensure_asr({"download": "off"})
    assert constants.ASR_DIRNAME in str(refused.value) and "'download' in YuE2 Options" in str(refused.value)


@pytest.mark.parametrize("choice", ["auto", "comfy-org", "original"])
def test_every_other_download_choice_fetches_qwens_release(monkeypatch, tmp_path, choice):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_asr", lambda progress=None: "fetched")
    assert download.ensure_asr({"download": choice}) == "fetched"
