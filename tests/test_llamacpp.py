"""Finding an llama.cpp runtime, and refusing to unpack one that misbehaves.

Nothing here touches the network. The archives are built in the test, because
what is worth testing about them is what happens when a member points somewhere
it should not -- and that is easier to write than to download.

The release itself was checked by running it: on 2026-09-12 all four Windows
assets of b10310 answered a HEAD with a size (17.51 MB cpu, 32.53 MB vulkan,
512.59 MB cuda including cudart), the vulkan archive unpacked to 52 files in
3.2 s, and the binary in it wrote a song on the Vulkan device of an RTX 5090.
"""

import io
import os
import sys
import tarfile
import zipfile

import pytest

from yue2_comfy import llamacpp, paths


@pytest.fixture
def user(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "user_dir", lambda: str(tmp_path))
    monkeypatch.delenv(llamacpp.BIN_ENV, raising=False)
    return tmp_path


def make_binary(directory, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or llamacpp.BINARIES[0])
    path.write_bytes(b"MZ")
    return str(path)


def test_auto_takes_cuda_on_a_card_the_archive_has_code_for(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(llamacpp, "nvidia_capability", lambda: (12, 0))
    assert llamacpp.resolve_backend("auto") == "cuda"


def test_auto_takes_vulkan_on_a_card_the_archive_has_nothing_for(monkeypatch):
    """No PTX in that archive, so an unlisted card has nothing to fall back on."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(llamacpp, "nvidia_capability", lambda: (7, 5))
    assert llamacpp.resolve_backend("auto") == "vulkan"


def test_auto_takes_vulkan_with_no_card_at_all(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(llamacpp, "nvidia_capability", lambda: None)
    assert llamacpp.resolve_backend("auto") == "vulkan"


def test_a_named_backend_is_taken_as_given(monkeypatch):
    monkeypatch.setattr(llamacpp, "nvidia_capability", lambda: (12, 0))
    assert llamacpp.resolve_backend("cpu") == "cpu"


def test_cuda_on_linux_says_what_to_do_instead(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError) as error:
        llamacpp.assets("cuda")
    message = str(error.value)
    assert "no CUDA build" in message
    assert llamacpp.BIN_ENV in message
    assert "GGML_CUDA=ON" in message


def test_an_unpublished_backend_names_the_published_ones(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError) as error:
        llamacpp.assets("rocm")
    assert "cpu, vulkan" in str(error.value)


def test_the_binary_is_found_at_any_depth(tmp_path):
    wanted = make_binary(tmp_path / "build" / "bin")
    assert llamacpp.find_binary(str(tmp_path)) == wanted


def test_a_directory_given_by_name_is_searched(user):
    wanted = make_binary(user / "mine")
    assert llamacpp._binary_at(str(user / "mine"), "test") == wanted


def test_a_sibling_of_the_named_file_is_accepted(user):
    """Somebody points at llama-server; the completion binary is right there."""
    wanted = make_binary(user / "mine")
    beside = user / "mine" / ("llama-server" + llamacpp.EXE)
    beside.write_bytes(b"MZ")
    assert llamacpp._binary_at(str(beside), "test") == wanted


def test_a_path_that_points_nowhere_raises_rather_than_downloads(user):
    """Naming a build is an instruction, and a typo should not cost 512 MB."""
    with pytest.raises(RuntimeError) as error:
        llamacpp._binary_at(str(user / "typo"), "YUE2_LLAMA_BIN")
    assert "does not exist" in str(error.value)


def test_a_folder_without_a_binary_in_it_raises(user):
    (user / "empty").mkdir()
    with pytest.raises(RuntimeError) as error:
        llamacpp._binary_at(str(user / "empty"), "YUE2_LLAMA_BIN")
    assert "holds neither" in str(error.value)


def test_nothing_given_is_not_an_error(user):
    assert llamacpp._binary_at("", "YUE2_LLAMA_BIN") == ""


def test_the_environment_variable_wins_over_an_unpacked_runtime(user, monkeypatch):
    mine = make_binary(user / "mine")
    make_binary(user / llamacpp.SUBDIR / (llamacpp.RELEASE + "-vulkan"))
    monkeypatch.setenv(llamacpp.BIN_ENV, str(user / "mine"))
    assert llamacpp.ensure("vulkan", False, None) == mine


def test_the_written_file_is_read_when_the_variable_is_not_set(user):
    mine = make_binary(user / "mine")
    (user / llamacpp.BIN_FILE).write_text(
        "# where my build lives\n\n" + str(user / "mine") + "\n", encoding="utf-8")
    assert llamacpp.ensure("vulkan", False, None) == mine


def test_an_unpacked_runtime_is_used_without_asking_the_network(user):
    wanted = make_binary(user / llamacpp.SUBDIR / (llamacpp.RELEASE + "-vulkan"))
    assert llamacpp.installed("vulkan") == wanted
    assert llamacpp.available()
    assert llamacpp.ensure("vulkan", False, None) == wanted


def test_nothing_anywhere_and_downloading_off_says_where_it_looked(user, monkeypatch):
    monkeypatch.setattr(llamacpp, "_path_binary", lambda: "")
    with pytest.raises(RuntimeError) as error:
        llamacpp.ensure("vulkan", False, None)
    message = str(error.value)
    assert "downloading is off" in message
    assert llamacpp.install_dir("vulkan") in message
    assert str(os.getpid()) in message
    assert llamacpp.BIN_FILE in message


def test_by_hand_names_the_archives_and_the_folder(user, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    text = llamacpp.by_hand("vulkan")
    assert llamacpp.DOWNLOAD_URL + "/llama-" + llamacpp.RELEASE in text
    assert llamacpp.install_dir("vulkan") in text


def test_a_zip_member_outside_the_target_is_refused(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "no")
    with pytest.raises(RuntimeError) as error:
        llamacpp._safe_extract(str(archive), str(tmp_path / "out"))
    assert "outside the target" in str(error.value)
    assert not (tmp_path / "escaped.txt").exists()


def _tar_with(path, members):
    with tarfile.open(path, "w") as handle:
        for info, payload in members:
            handle.addfile(info, io.BytesIO(payload) if payload is not None else None)


def test_a_tar_symlink_pointing_outside_is_refused(tmp_path):
    archive = tmp_path / "evil.tar"
    link = tarfile.TarInfo("build/bin/libllama.so")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../../../secret"
    _tar_with(archive, [(link, None)])
    with pytest.raises(RuntimeError) as error:
        llamacpp._safe_extract(str(archive), str(tmp_path / "out"))
    assert "pointing outside" in str(error.value)


def test_a_soname_symlink_inside_the_archive_is_kept(tmp_path):
    """Every Linux release ships these, and the binaries link against them."""
    archive = tmp_path / "good.tar"
    real = tarfile.TarInfo("build/bin/libllama.so.0.0.10310")
    real.size = 2
    link = tarfile.TarInfo("build/bin/libllama.so.0")
    link.type = tarfile.SYMTYPE
    link.linkname = "libllama.so.0.0.10310"
    _tar_with(archive, [(real, b"so"), (link, None)])
    llamacpp._safe_extract(str(archive), str(tmp_path / "out"))
    assert (tmp_path / "out" / "build" / "bin" / "libllama.so.0.0.10310").is_file()


def test_a_hardlink_is_measured_from_the_root_not_from_its_own_folder(tmp_path):
    """The two kinds resolve from different places; one base for both waves this through."""
    archive = tmp_path / "evil.tar"
    link = tarfile.TarInfo("build/bin/x")
    link.type = tarfile.LNKTYPE
    link.linkname = "../secret"
    _tar_with(archive, [(link, None)])
    with pytest.raises(RuntimeError) as error:
        llamacpp._safe_extract(str(archive), str(tmp_path / "out"))
    assert "pointing outside" in str(error.value)
