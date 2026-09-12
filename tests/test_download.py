"""Fetching the weights: the decisions, not the transfer.

The ranged HTTP download is not mocked here. A fake server would only prove
that the fake answers, and the parts of it that go wrong in real life -- an
expired CDN link, a server that ignores Range, a transparently gzipped
response -- cannot be reproduced by one. What is tested is everything that
decides whether to transfer at all, which is where a mistake costs somebody
eight gigabytes or a wrong folder.
"""

import os

import pytest

from yue2_comfy import constants, discovery, download, paths


def test_auto_means_the_repack_because_one_download_serves_both():
    """It is the file ComfyUI's own nodes read, so fetching it twice is silly."""
    assert download.source_for({}) == "comfy-org"
    assert download.source_for({"download": "auto"}) == "comfy-org"


def test_auto_switches_to_the_released_files_for_the_legacy_decoder():
    """Comfy-Org does not publish that decoder, so auto has to go elsewhere."""
    assert download.source_for({"download": "auto", "vae": "legacy"}) == "original"


def test_an_explicit_choice_is_never_second_guessed():
    assert download.source_for({"download": "original", "vae": "standard"}) == "original"
    assert download.source_for({"download": "comfy-org", "vae": "legacy"}) == "comfy-org"
    assert download.source_for({"download": "off"}) == "off"


def test_the_quantization_switch_picks_the_file_and_the_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "checkpoints_root", lambda: str(tmp_path))

    bf16 = download.repack_wanted("bf16")
    int8 = download.repack_wanted("int8")

    assert list(bf16) == [constants.REPACK_BF16_PATH]
    assert list(int8) == [constants.REPACK_INT8_PATH]
    assert list(bf16.values()) == [os.path.join(str(tmp_path), constants.REPACK_BF16_NAME)]
    assert list(int8.values()) == [os.path.join(str(tmp_path), constants.REPACK_INT8_NAME)]
    assert download.repack_bytes("int8") < download.repack_bytes("bf16")


def test_a_repository_path_that_escapes_its_folder_is_refused():
    """The paths come from a server, so they are input, not configuration."""
    for hostile in ("../../etc/passwd", "C:/windows/system32", "a/../../b", ""):
        with pytest.raises(download.DownloadError):
            download._safe_name(hostile)
    assert download._safe_name("checkpoints/yue2.safetensors") == "yue2.safetensors"


def test_space_is_checked_before_the_first_byte(monkeypatch, tmp_path):
    """Failing seven gigabytes in costs the whole transfer, so fail at zero."""
    monkeypatch.setattr(download, "free_space", lambda directory: 1000)

    download.check_space(str(tmp_path), 900)
    with pytest.raises(download.DownloadError) as error:
        download.check_space(str(tmp_path), 999)
    assert "is free" in str(error.value)


def test_the_token_is_a_header_and_nothing_arrives_compressed():
    headers = download._headers("secret")
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Accept-Encoding"] == "identity"
    assert "Authorization" not in download._headers(None)


def test_a_file_that_is_already_whole_is_not_fetched_again(monkeypatch, tmp_path):
    target = tmp_path / "weights.safetensors"
    target.write_bytes(b"12345")
    monkeypatch.setattr(download, "list_repo_files", lambda *args, **kwargs: {"w": 5})
    monkeypatch.setattr(download, "access_token", lambda: None)

    def refuse(*args, **kwargs):
        raise AssertionError("the transfer should not have started")

    monkeypatch.setattr(download, "fetch_item", refuse)

    assert download.fetch("org/repo", {"w": str(target)}, "test") == 5


def test_a_file_of_the_wrong_size_is_removed_rather_than_resumed(monkeypatch, tmp_path):
    """Under the final name, a short file is not a download, it is a wreck."""
    target = tmp_path / "weights.safetensors"
    target.write_bytes(b"12")
    monkeypatch.setattr(download, "list_repo_files", lambda *args, **kwargs: {"w": 5})
    monkeypatch.setattr(download, "access_token", lambda: None)
    fetched = []

    def record(repo_id, item, revision, token, base, on_progress):
        fetched.append(item.repo_path)
        return item.size

    monkeypatch.setattr(download, "fetch_item", record)
    download.fetch("org/repo", {"w": str(target)}, "test")

    assert fetched == ["w"]
    assert not target.exists()


def test_a_repository_path_that_has_vanished_is_named(monkeypatch):
    monkeypatch.setattr(download, "list_repo_files", lambda *args, **kwargs: {"other": 1})

    with pytest.raises(download.DownloadError) as error:
        download.plan("org/repo", {"gone.safetensors": "nowhere"})
    assert "gone.safetensors" in str(error.value)


def test_nothing_is_downloaded_when_the_weights_are_already_here(monkeypatch):
    found = discovery.Files(repack="already here")
    monkeypatch.setattr(discovery, "locate", lambda variant, quantization: found)

    def refuse(*args, **kwargs):
        raise AssertionError("nothing should have been fetched")

    monkeypatch.setattr(download, "fetch_repack", refuse)
    monkeypatch.setattr(download, "fetch_original", refuse)

    assert download.ensure({}) is found


def test_downloading_off_keeps_the_instructions_and_adds_the_way_out(monkeypatch):
    def missing(variant, quantization):
        raise FileNotFoundError("here is where to put them")

    monkeypatch.setattr(discovery, "locate", missing)

    with pytest.raises(FileNotFoundError) as error:
        download.ensure({"download": "off"})
    message = str(error.value)
    assert "here is where to put them" in message
    assert "'auto'" in message


def test_the_source_decides_which_repository_is_fetched(monkeypatch):
    calls = []
    attempts = {"n": 0}

    def missing_once(variant, quantization):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FileNotFoundError("not yet")
        return discovery.Files(repack="fetched")

    monkeypatch.setattr(discovery, "locate", missing_once)
    monkeypatch.setattr(download, "fetch_repack",
                        lambda quantization, progress=None: calls.append(("repack", quantization)))
    monkeypatch.setattr(download, "fetch_original",
                        lambda variant, progress=None: calls.append(("original", variant)))

    assert download.ensure({"quantization": "int8"}).repack == "fetched"
    assert calls == [("repack", "int8")]

    attempts["n"] = 0
    calls.clear()
    download.ensure({"download": "original", "vae": "legacy"})
    assert calls == [("original", "legacy")]
