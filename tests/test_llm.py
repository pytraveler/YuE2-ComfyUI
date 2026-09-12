"""Finding a writer model, and refusing the files that cannot be one.

The GGUFs here are built byte by byte rather than downloaded, so the tests say
what the header reader does with an adapter, an mmproj and a truncated file
without keeping three gigabytes in the repository.

``general.type`` really is the separator: measured across seventeen files in a
working model folder it read ``model``, ``adapter`` or ``mmproj`` outright, and
only the models carried a chat template.
"""

import os
import struct

import pytest

from yue2_comfy import gguf_meta, llm, paths

STRING = 8
PADDING = 4096


def _kv(key: str, value: str) -> bytes:
    name = key.encode("utf-8")
    text = value.encode("utf-8")
    return (struct.pack("<Q", len(name)) + name + struct.pack("<I", STRING)
            + struct.pack("<Q", len(text)) + text)


def write_gguf(path, kind="model", arch="qwen35", chat=True, truncated=False):
    """A GGUF with a valid header, no tensors, and nothing else in it."""
    pairs = [_kv("general.architecture", arch), _kv("general.type", kind)]
    if chat:
        pairs.append(_kv("tokenizer.chat_template", "{{ messages }}"))
    header = (gguf_meta.MAGIC + struct.pack("<I", 3)
              + struct.pack("<QQ", 0, len(pairs)) + b"".join(pairs))
    with open(path, "wb") as handle:
        handle.write(header)
        if not truncated:
            handle.write(b"\0" * PADDING)
    return str(path)


@pytest.fixture(autouse=True)
def forget_models():
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": []})
    yield
    llm._HEADERS.clear()
    llm._CATALOGUE.update({"roots": None, "entries": []})


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "gguf_roots", lambda: [str(tmp_path)])
    return tmp_path


def test_a_model_with_a_chat_template_is_runnable(tmp_path):
    assert llm.runnable(write_gguf(tmp_path / "writer.gguf"))


def test_an_adapter_is_not_offered(tmp_path):
    """A LoRA would load, answer as the base model, and look like a bad writer."""
    assert not llm.runnable(write_gguf(tmp_path / "lora.gguf", kind="adapter", chat=False))


def test_an_mmproj_is_not_offered(tmp_path):
    assert not llm.runnable(
        write_gguf(tmp_path / "mmproj.gguf", kind="mmproj", arch="clip", chat=False))


def test_a_model_without_a_chat_template_is_not_offered(tmp_path):
    assert not llm.runnable(write_gguf(tmp_path / "bare.gguf", chat=False))


def test_a_half_downloaded_file_is_not_offered(tmp_path):
    assert not llm.runnable(write_gguf(tmp_path / "cut.gguf", truncated=True))


def test_something_that_is_not_a_gguf_is_not_offered(tmp_path):
    path = tmp_path / "notes.gguf"
    path.write_bytes(b"this is not a model")
    assert not llm.runnable(str(path))


def test_a_missing_file_is_not_offered(tmp_path):
    assert not llm.runnable(str(tmp_path / "never-existed.gguf"))


def test_the_catalogue_lists_only_what_can_write(folder):
    write_gguf(folder / "writer.gguf")
    write_gguf(folder / "lora.gguf", kind="adapter", chat=False)
    write_gguf(folder / "mmproj.gguf", kind="mmproj", arch="clip", chat=False)
    assert [name for name, _path in llm.catalogue()] == ["writer.gguf"]


def test_the_catalogue_looks_inside_subfolders(folder):
    (folder / "nested").mkdir()
    write_gguf(folder / "nested" / "writer.gguf")
    assert [name for name, _path in llm.catalogue()] == ["writer.gguf"]


def test_two_files_of_the_same_name_stay_tellable_apart(folder):
    for parent in ("q4", "q8"):
        (folder / parent).mkdir()
        write_gguf(folder / parent / "same.gguf")
    assert sorted(name for name, _path in llm.catalogue()) == ["q4/same.gguf", "q8/same.gguf"]


def test_the_search_is_bounded_in_depth(folder):
    """`checkpoints` is in the search now, and it is a large tree."""
    deep = folder / "one" / "two" / "three"
    deep.mkdir(parents=True)
    write_gguf(deep / "buried.gguf")
    write_gguf(folder / "one" / "shallow.gguf")
    assert [name for name, _path in llm.catalogue()] == ["shallow.gguf"]


def test_only_the_first_shard_of_a_split_model_is_offered(folder):
    """Measured: llama.cpp loads the siblings itself, given the first shard.

    The later shards carry no chat template, which is what keeps them out --
    offering them would hand somebody half a model.
    """
    write_gguf(folder / "model-00001-of-00002.gguf")
    write_gguf(folder / "model-00002-of-00002.gguf", chat=False)
    assert [name for name, _path in llm.catalogue()] == ["model-00001-of-00002.gguf"]


def test_a_cache_copy_is_named_by_repository_not_by_commit(tmp_path, monkeypatch):
    """A snapshot directory is a commit hash, which is no use as a label."""
    snapshot = tmp_path / "models--Qwen--Qwen3-VL-8B-Instruct-GGUF" / "snapshots" / "f982a075"
    snapshot.mkdir(parents=True)
    write_gguf(snapshot / "same.gguf")
    (tmp_path / "LLM").mkdir()
    write_gguf(tmp_path / "LLM" / "same.gguf")
    monkeypatch.setattr(paths, "gguf_roots", lambda: [str(tmp_path / "LLM"), str(snapshot)])
    assert sorted(name for name, _path in llm.catalogue()) == [
        "LLM/same.gguf", "Qwen/Qwen3-VL-8B-Instruct-GGUF/same.gguf"]


def test_choices_start_with_auto(folder):
    write_gguf(folder / "writer.gguf")
    assert llm.choices() == ["auto", "writer.gguf"]


def test_auto_prefers_the_default_model(folder):
    from yue2_comfy.constants import WRITER_NAME

    write_gguf(folder / "aaa-first-alphabetically.gguf")
    wanted = write_gguf(folder / WRITER_NAME)
    assert llm.resolve("auto", {}, None) == wanted


def test_auto_takes_what_is_there_when_the_default_is_absent(folder):
    only = write_gguf(folder / "something-else.gguf")
    assert llm.resolve("auto", {}, None) == only


def test_a_named_model_resolves_to_its_path(folder):
    wanted = write_gguf(folder / "writer.gguf")
    write_gguf(folder / "other.gguf")
    assert llm.resolve("writer.gguf", {}, None) == wanted


def test_a_deleted_model_says_what_is_there_instead(folder):
    write_gguf(folder / "still-here.gguf")
    with pytest.raises(FileNotFoundError) as error:
        llm.resolve("deleted.gguf", {}, None)
    message = str(error.value)
    assert "deleted.gguf" in message
    assert "still-here.gguf" in message
    assert "auto" in message


def test_auto_with_nothing_on_disk_and_downloading_off_gives_the_link(folder):
    with pytest.raises(FileNotFoundError) as error:
        llm.resolve("auto", {"download": "off"}, None)
    message = str(error.value)
    assert "huggingface.co/unsloth/Qwen3.5-4B-GGUF" in message
    assert "Qwen3.5-4B-Q4_K_M.gguf" in message


def test_the_install_hint_names_this_interpreter():
    import sys

    hint = llm.install_hint()
    assert "llama-cpp-python" in hint
    assert os.path.basename(sys.executable or "python") in hint


def test_headers_are_read_once_per_file(folder, monkeypatch):
    """A folder of large quants should cost a stat each after the first pass."""
    path = write_gguf(folder / "writer.gguf")
    reads = []
    real = gguf_meta.keys
    monkeypatch.setattr(gguf_meta, "keys",
                        lambda *args, **kwargs: reads.append(args[0]) or real(*args, **kwargs))
    assert llm.runnable(path)
    assert llm.runnable(path)
    assert reads == [path]
