"""Finding and fetching SheetSage2's weights, without the 1.3 GB file."""

from __future__ import annotations

import os

import pytest

from yue2_comfy import constants, discovery, download, paths
from test_loader import write_safetensors

SHEETSAGE_KEYS = list(constants.SHEETSAGE_MARKERS) + ["decoder.layers.0.fc1.weight", "encoder.layers.0.attn.key_proj.weight"]


@pytest.fixture(autouse=True)
def forget_identifications(monkeypatch):
    monkeypatch.setattr(discovery, "_identified", {})


def test_the_published_name_is_found_in_an_audio_encoders_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    folder = tmp_path / "audio_encoders"
    folder.mkdir()
    (folder / constants.SHEETSAGE_NAME).write_bytes(b"not read")
    assert discovery.find_sheetsage() == str(folder / constants.SHEETSAGE_NAME)


def test_a_renamed_file_is_found_by_its_tensors(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    write_safetensors(tmp_path / "transcriber.safetensors", SHEETSAGE_KEYS)
    write_safetensors(tmp_path / "other.safetensors", ["model.weight"])
    assert discovery.find_sheetsage() == str(tmp_path / "transcriber.safetensors")


def test_sheetsage_dropped_into_a_vae_folder_is_not_taken_for_the_yue2_decoder(tmp_path):
    """Its decoder keys would pass the VAE check, which runs after this one."""
    write_safetensors(tmp_path / "vae.safetensors", SHEETSAGE_KEYS)
    assert discovery.identify(str(tmp_path / "vae.safetensors")) == "sheetsage"


def test_nothing_found_is_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    write_safetensors(tmp_path / "other.safetensors", ["decoder.layers.0.weight"])
    assert discovery.find_sheetsage() == ""


def test_a_downloaded_comfy_org_repository_beside_the_checkout_is_searched(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.setattr(paths, "_folder_paths", lambda: None)
    monkeypatch.setattr(paths, "hf_cache_roots", lambda: [])
    monkeypatch.setattr(paths, "checkout_sibling_root", lambda: str(tmp_path))
    folder = tmp_path / "ComfyOrg-YuE2" / "audio_encoders"
    folder.mkdir(parents=True)
    (folder / constants.SHEETSAGE_NAME).write_bytes(b"x")
    assert str(folder) in paths.sheetsage_roots()
    assert discovery.find_sheetsage() == str(folder / constants.SHEETSAGE_NAME)


def test_the_refusal_names_the_link_the_folder_and_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    message = discovery.sheetsage_missing_message([str(tmp_path)])
    assert "https://huggingface.co/Comfy-Org/YuE2/resolve/main/audio_encoders/sheetsage2_bf16.safetensors" in message
    assert os.path.join(str(tmp_path), "audio_encoders", constants.SHEETSAGE_NAME) in message
    assert "Looked in 1 place" in message and "CC BY-NC 4.0" in message


def test_the_download_lands_in_audio_encoders(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "audio_encoders_root", lambda: str(tmp_path))
    fetched = []
    monkeypatch.setattr(download, "fetch", lambda repo, wanted, title, progress=None: fetched.append((repo, wanted, title)))
    assert download.fetch_sheetsage() == str(tmp_path / constants.SHEETSAGE_NAME)
    repo, wanted, title = fetched[0]
    assert repo == "Comfy-Org/YuE2"
    assert wanted == {"audio_encoders/sheetsage2_bf16.safetensors": str(tmp_path / constants.SHEETSAGE_NAME)}
    assert "1.29 GB" in title or "1.3" in title


def test_nothing_is_fetched_when_the_file_is_here(monkeypatch):
    monkeypatch.setattr(discovery, "find_sheetsage", lambda roots=None: "here.safetensors")
    monkeypatch.setattr(download, "fetch_sheetsage", lambda progress=None: pytest.fail("fetched"))
    assert download.ensure_sheetsage({"download": "off"}) == "here.safetensors"


def test_downloading_off_explains_instead_of_fetching(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_sheetsage", lambda progress=None: pytest.fail("fetched"))
    with pytest.raises(FileNotFoundError) as refused:
        download.ensure_sheetsage({"download": "off"})
    assert "audio_encoders" in str(refused.value) and "'download' in YuE2 Options" in str(refused.value)


@pytest.mark.parametrize("choice", ["auto", "comfy-org", "original"])
def test_every_other_download_choice_fetches_the_comfy_org_file(monkeypatch, tmp_path, choice):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.setattr(download, "fetch_sheetsage", lambda progress=None: "fetched.safetensors")
    assert download.ensure_sheetsage({"download": choice}) == "fetched.safetensors"
